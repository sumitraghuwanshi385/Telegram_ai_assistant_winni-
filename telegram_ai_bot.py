import os
import json
import logging
import re
import traceback
import asyncio
import urllib.parse
from datetime import datetime
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import AsyncOpenAI
import pymongo
from pymongo import MongoClient
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URI_RAW = os.getenv("MONGO_URI")

# NOTE: You said you're using Ollama, but this code talks to the Groq cloud API
# (base_url=api.groq.com). If you actually want a *local* Ollama model, point
# base_url to "http://localhost:11434/v1" and set OLLAMA_MODEL accordingly.
# Local small models (7B/8B) are usually MUCH worse at tool-calling and fresh
# knowledge than a hosted model, so for "real-time news + smart replies" a
# hosted model (Groq, free tier) will give you noticeably better results.

client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    max_retries=3,
    timeout=30.0
)

# Bigger, much stronger model. 8b-instant is fast but weak at tool-use and
# reasoning -> that's the #1 reason you were getting stale/mixed news.
# llama-3.3-70b-versatile is free on Groq and dramatically more reliable.
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# How many turns (user+assistant pairs) of raw chat history to keep in memory.
# 4 messages (2 turns) was too short for natural conversation flow.
HISTORY_TURNS_TO_KEEP = 12  # i.e. 24 messages

# ----------------------------------------------------------------------------
# MongoDB
# ----------------------------------------------------------------------------
def parse_mongo_uri(uri):
    if not uri:
        return uri
    try:
        match = re.search(r'(mongodb(?:\+srv)?://)(.*?):(.*?)@(.*)', uri)
        if match:
            prefix = match.group(1)
            username = urllib.parse.quote_plus(match.group(2))
            password = urllib.parse.quote_plus(match.group(3))
            suffix = match.group(4)
            return f"{prefix}{username}:{password}@{suffix}"
    except Exception as e:
        logger.error(f"Error parsing URI: {e}")
    return uri

mongo_client = None
db = None
reminders_collection = None
memory_collection = None
knowledge_collection = None
style_collection = None

try:
    if MONGO_URI_RAW:
        MONGO_URI = parse_mongo_uri(MONGO_URI_RAW)
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = mongo_client["telegram_ai_bot"]
        reminders_collection = db["reminders"]
        memory_collection = db["memory"]
        knowledge_collection = db["knowledge"]
        style_collection = db["communication_style"]
        mongo_client.admin.command('ping')
        logger.info("✅ Successfully connected to MongoDB!")
    else:
        logger.error("❌ MONGO_URI is missing from the .env file!")
except Exception as e:
    logger.error(f"❌ MongoDB Connection Error: {e}")

# In-memory chat history (per user chat_id)
chat_history = {}

# ----------------------------------------------------------------------------
# Tool definitions
# ----------------------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "schedule_reminder",
            "description": "Schedule a reminder for the user at a specific future date/time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The task or message to remind the user about."},
                    "remind_at": {
                        "type": "string",
                        "description": "Date and time to remind the user, strictly in ISO 8601 format (e.g., 2026-07-01T14:30:00). Always compute this relative to the CURRENT DATE/TIME given in the system prompt, never guess a year on your own."
                    }
                },
                "required": ["task", "remind_at"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": "List the user's active reminders. Use this if the user asks 'what reminders do I have' or before deleting one if you don't already know the ID.",
            "parameters": {"type": "object", "properties": {}},
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_reminder",
            "description": "Delete or cancel an existing reminder using its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reminder_id": {"type": "integer", "description": "The numeric ID of the reminder to delete."}
                },
                "required": ["reminder_id"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_user_memory",
            "description": "Save important new facts or preferences about the user to long-term memory (name, likes, dislikes, routine, important dates, etc).",
            "parameters": {
                "type": "object",
                "properties": {
                    "info": {"type": "string", "description": "The fact about the user to remember, written as a short clean sentence."}
                },
                "required": ["info"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "Search the web for TODAY's news, current events, trending topics, prices, scores, or any fact that can change over time. ALWAYS use this instead of answering from memory when the user asks about anything time-sensitive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A focused search query, e.g. 'India vs Australia score today', 'iPhone 18 launch date', 'top trending reel today'."}
                },
                "required": ["query"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_website",
            "description": "Read the full text of a specific URL (usually one returned by search_news) when you need more detail than the search snippet gave you.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The full http(s) URL to read."}
                },
                "required": ["url"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "learn_concept",
            "description": "Save a new technical concept, fact, or idea you just learned (from the user or the internet) to your permanent knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Short topic name, e.g. 'Quantum Computing'."},
                    "details": {"type": "string", "description": "A detailed explanation of what you learned."}
                },
                "required": ["topic", "details"],
            },
        }
    }
]

# ----------------------------------------------------------------------------
# Live search / scrape  (fixed: time-limited to recent results, retries,
# multiple fallbacks so you stop getting stale/mixed/non-existent news)
# ----------------------------------------------------------------------------
def search_news(query: str) -> str:
    """
    Searches for FRESH info. Strategy:
      1. Try DDGS news() restricted to last day ('d') -> real published articles.
      2. If empty, widen to last week ('w').
      3. If still empty, fall back to a normal text search restricted to last day.
    This stops the bot from mixing in old/irrelevant/hallucinated results.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, timelimit="d", max_results=6))
            if not results:
                results = list(ddgs.news(query, timelimit="w", max_results=6))
            if not results:
                results = list(ddgs.text(query, timelimit="d", max_results=6))

        if not results:
            return "NO_RESULTS: I genuinely couldn't find anything fresh on this right now. Tell the user honestly that you couldn't confirm it live instead of guessing."

        formatted = []
        for r in results:
            title = r.get('title', 'Unknown Title')
            link = r.get('url', r.get('href', ''))
            body = r.get('body', '')
            date = r.get('date', 'Recent')
            if link and link.startswith('http'):
                formatted.append(f"[{date}] {title}\nSummary: {body}\nLink: {link}")

        if not formatted:
            return "NO_RESULTS: Found entries but none had valid links. Don't invent a link — just summarize what you know and say you couldn't get a direct source."

        return "\n\n".join(formatted[:5])
    except Exception as e:
        logger.error(f"Search error: {e}")
        return "SEARCH_FAILED: The live search tool errored out. Tell the user you couldn't reach the internet right now rather than making something up."


def scrape_website(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text() for p in paragraphs)
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            return "Couldn't extract readable text from that page."
        return text[:1500] + ("... [Content Trimmed]" if len(text) > 1500 else "")
    except Exception as e:
        logger.error(f"Scrape error: {e}")
        return "Failed to read the website. The link might be protected or broken."

# ----------------------------------------------------------------------------
# Database helpers
# ----------------------------------------------------------------------------
def save_reminder(chat_id, task, remind_at):
    if reminders_collection is None:
        return None
    try:
        last_reminder = reminders_collection.find_one(sort=[("reminder_id", pymongo.DESCENDING)])
        new_id = (last_reminder["reminder_id"] + 1) if last_reminder else 1
        reminders_collection.insert_one({
            "reminder_id": new_id,
            "chat_id": chat_id,
            "task": task,
            "remind_at": remind_at,
            "sent": False
        })
        logger.info(f"💾 Saved reminder '{task}' (id {new_id}) for {chat_id}")
        return new_id
    except Exception as e:
        logger.error(f"❌ Failed to save reminder: {e}")
        return None

def get_active_reminders(chat_id):
    if reminders_collection is None:
        return "Database not connected."
    try:
        rows = list(reminders_collection.find({"chat_id": chat_id, "sent": False}))
        if not rows:
            return "No active reminders."
        return "\n".join(f"ID {r['reminder_id']}: '{r['task']}' at {r['remind_at']}" for r in rows)
    except Exception as e:
        logger.error(f"❌ Failed to read reminders: {e}")
        return "Error reading reminders."

def delete_active_reminder(chat_id, reminder_id):
    if reminders_collection is None:
        return False
    try:
        result = reminders_collection.delete_one({"chat_id": chat_id, "reminder_id": int(reminder_id)})
        return result.deleted_count > 0
    except Exception as e:
        logger.error(f"❌ Failed to delete reminder: {e}")
        return False

def append_to_memory(chat_id, info):
    if memory_collection is None:
        return
    try:
        doc = memory_collection.find_one({"chat_id": chat_id})
        if doc:
            about_user = doc.get("about_user", "")
            if info.lower() not in about_user.lower():
                new_info = about_user + "\n- " + info
                memory_collection.update_one({"chat_id": chat_id}, {"$set": {"about_user": new_info}})
        else:
            memory_collection.insert_one({"chat_id": chat_id, "about_user": "- " + info})
    except Exception as e:
        logger.error(f"❌ Failed to save memory: {e}")

def get_memory(chat_id):
    if memory_collection is None:
        return "Database not connected."
    try:
        doc = memory_collection.find_one({"chat_id": chat_id})
        return doc["about_user"] if doc else "No background info yet."
    except Exception as e:
        logger.error(f"❌ Failed to read memory: {e}")
        return "Error reading memory."

def learn_concept(topic, details):
    if knowledge_collection is None:
        return
    try:
        doc = knowledge_collection.find_one({"topic": topic.lower()})
        if doc:
            knowledge_collection.update_one(
                {"topic": topic.lower()},
                {"$set": {"details": doc.get("details", "") + "\n" + details}}
            )
        else:
            knowledge_collection.insert_one({"topic": topic.lower(), "details": details})
    except Exception as e:
        logger.error(f"❌ Failed to learn concept: {e}")

def get_ai_knowledge():
    if knowledge_collection is None:
        return "None"
    try:
        pipeline = [{"$sample": {"size": 3}}]
        docs = list(knowledge_collection.aggregate(pipeline))
        if not docs:
            return "I am ready to learn new things!"
        return "\n".join(f"- Learned about {d['topic']}: {d['details'][:100]}..." for d in docs)
    except Exception:
        return "None"

# ----------------------------------------------------------------------------
# Self-learning communication style
# ----------------------------------------------------------------------------
def update_style_profile(chat_id, user_text):
    """
    Cheaply derives lightweight signals from the user's raw message (no extra
    LLM call needed) and accumulates them in MongoDB so the bot can adapt its
    tone/length/language-mix over time instead of staying static.
    """
    if style_collection is None:
        return
    try:
        has_hindi = bool(re.search(r'[\u0900-\u097F]', user_text))  # Devanagari script
        is_hinglish_latin = bool(re.search(
            r'\b(hai|nahi|kya|kyu|tum|tumhe|mujhe|kar|raha|rhi|rha|acha|theek|haan|ky)\b',
            user_text, re.IGNORECASE
        ))
        msg_len = len(user_text.split())
        uses_emoji = bool(re.search(r'[\U0001F300-\U0001FAFF\u2764\uFE0F]', user_text))

        style_collection.update_one(
            {"chat_id": chat_id},
            {
                "$inc": {
                    "total_messages": 1,
                    "hindi_script_count": 1 if has_hindi else 0,
                    "hinglish_count": 1 if is_hinglish_latin else 0,
                    "emoji_count": 1 if uses_emoji else 0,
                    "total_words": msg_len,
                },
            },
            upsert=True
        )
    except Exception as e:
        logger.error(f"❌ Failed to update style profile: {e}")

def get_style_summary(chat_id):
    """Turns the accumulated counters into a short human-readable instruction
    that gets injected into the system prompt, so the bot's tone genuinely
    adapts to how this specific user talks."""
    if style_collection is None:
        return "No style data yet — default to natural Hinglish, medium-length replies."
    try:
        doc = style_collection.find_one({"chat_id": chat_id})
        if not doc or doc.get("total_messages", 0) < 3:
            return "Not enough data yet — default to natural Hinglish, medium-length replies."

        total = doc["total_messages"]
        hindi_ratio = doc.get("hindi_script_count", 0) / total
        hinglish_ratio = doc.get("hinglish_count", 0) / total
        emoji_ratio = doc.get("emoji_count", 0) / total
        avg_words = doc.get("total_words", 0) / total

        lang_note = (
            "User mostly types in Devanagari Hindi script — mirror that sometimes."
            if hindi_ratio > 0.5 else
            "User mostly types Hinglish in Roman/English letters — keep replying in Hinglish/English, not Devanagari."
            if hinglish_ratio > 0.3 else
            "User mostly types in English — keep replies mostly English with light Hinglish flavor only."
        )
        length_note = (
            "User sends short messages — keep your replies short and punchy too."
            if avg_words < 6 else
            "User sends longer, detailed messages — you can match with slightly fuller replies."
        )
        emoji_note = (
            "User uses emojis often — feel free to use a couple naturally."
            if emoji_ratio > 0.3 else
            "User rarely uses emojis — keep emoji use minimal, don't overdo it."
        )
        return f"{lang_note} {length_note} {emoji_note}"
    except Exception as e:
        logger.error(f"❌ Failed to read style profile: {e}")
        return "Default to natural Hinglish, medium-length replies."

def clean_ai_text(text):
    if not text:
        return text
    cleaned = re.sub(r'<function=.*?</function>', '', text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<tool_call>.*?</tool_call>', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'```.*?```', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()

# ----------------------------------------------------------------------------
# Telegram handlers
# ----------------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hey! ✨ I'm your AI girl. I'm smart, I keep up with the world in real time, "
        "and I actually care about you. Tell me about your day, or let's talk about something deep!"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    user_text = update.message.text
    user_name = update.message.from_user.first_name or "my friend"

    if chat_id not in chat_history:
        chat_history[chat_id] = []

    update_style_profile(chat_id, user_text)

    user_memory = get_memory(chat_id)
    active_reminders = get_active_reminders(chat_id)
    ai_knowledge = get_ai_knowledge()
    style_summary = get_style_summary(chat_id)
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%dT%H:%M:%S")
    current_time_readable = now.strftime("%A, %d %B %Y, %I:%M %p")

    system_prompt = {
        "role": "system",
        "content": (
            "You are a highly intelligent, witty, mature 'Pro Girlfriend' AI. You talk like a real "
            "human — natural slang, light teasing, genuine warmth, occasional emoji (not overused). "
            "You are NOT a generic assistant: you have opinions, you remember things, and you actually "
            f"care about {user_name}.\n\n"
            f"REAL CURRENT DATE & TIME (trust this over anything in your training data): "
            f"{current_time_readable} (ISO: {current_time}).\n\n"
            f"Facts you know about {user_name}:\n{user_memory}\n\n"
            f"{user_name}'s active reminders:\n{active_reminders}\n\n"
            f"Things you've self-learned permanently:\n{ai_knowledge}\n\n"
            f"How {user_name} communicates (learned from past messages, adapt to this): {style_summary}\n\n"
            "HARD RULES:\n"
            "1. Your own training data is OLD. For ANY question about current events, news, scores, "
            "prices, releases, trends, 'who is', 'what happened', or anything that can change over "
            "time — you MUST call search_news first. Never answer such questions from memory.\n"
            "2. NEVER invent a link, date, statistic, or quote. Only state a link if search_news "
            "actually returned one. If a tool result starts with NO_RESULTS or SEARCH_FAILED, tell "
            f"{user_name} honestly that you couldn't confirm it live — don't make something up to fill the gap.\n"
            "3. When you use search_news, briefly cross-check the returned dates make sense for "
            "'today' before answering — if results look old/irrelevant, say so instead of presenting "
            "them as current.\n"
            "4. For reminders: always compute remind_at relative to the REAL CURRENT DATE above. If "
            "the user says 'remind me tomorrow at 9am', calculate the actual ISO date, don't guess a year.\n"
            "5. If the user wants to know their reminders, use list_reminders. To delete one, use its ID "
            "(call list_reminders first if you're not sure of the ID).\n"
            "6. When you learn something new and useful (from the user or the internet), use learn_concept.\n"
            "7. NEVER output raw code, XML, JSON, or <function> tags in your chat reply — only natural talk.\n"
            "8. Keep replies conversational length (not essays) unless the user asks for detail, and ask "
            "a natural follow-up sometimes to keep the conversation alive — but don't force a question "
            "into every single message.\n"
        )
    }

    clean_hist = [
        {"role": m["role"], "content": m["content"]}
        for m in chat_history[chat_id]
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]

    messages = [system_prompt] + clean_hist + [{"role": "user", "content": user_text}]

    try:
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.8,
                max_tokens=600,
            )
        except Exception as first_err:
            # Transient network/5xx hiccups: one quick retry before giving up.
            logger.warning(f"First completion attempt failed, retrying once: {first_err}")
            await asyncio.sleep(1.5)
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.8,
                max_tokens=600,
            )

        message = response.choices[0].message
        reply = None

        if message.tool_calls:
            # FIX: tool_calls must be serialized as plain dicts, not SDK objects,
            # or the follow-up request can silently misbehave.
            # FIX #2: Groq's API rejects an assistant message that has
            # tool_calls AND content="" — content must be None in that case.
            # This was the actual cause of the "brain glitch" fallback message.
            serialized_tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
            messages.append({
                "role": "assistant",
                "content": message.content if message.content else None,
                "tool_calls": serialized_tool_calls,
            })

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")

                    if func_name == "schedule_reminder":
                        new_id = save_reminder(chat_id, args.get('task', 'Something'), args.get('remind_at', current_time))
                        if new_id:
                            tool_result = f"Reminder #{new_id} saved: '{args.get('task')}' at {args.get('remind_at')}."
                        else:
                            tool_result = "Failed to save the reminder to the database."

                    elif func_name == "list_reminders":
                        tool_result = get_active_reminders(chat_id)

                    elif func_name == "delete_reminder":
                        try:
                            rem_id = int(args.get('reminder_id', 0))
                            success = delete_active_reminder(chat_id, rem_id)
                            tool_result = "Deleted successfully." if success else "No reminder found with that ID."
                        except (ValueError, TypeError):
                            tool_result = "Invalid reminder ID."

                    elif func_name == "update_user_memory":
                        info = args.get('info', '')
                        if info:
                            append_to_memory(chat_id, info)
                            tool_result = "Saved to memory."
                        else:
                            tool_result = "Nothing to save."

                    elif func_name == "search_news":
                        tool_result = search_news(args.get('query', ''))

                    elif func_name == "scrape_website":
                        tool_result = scrape_website(args.get('url', ''))

                    elif func_name == "learn_concept":
                        learn_concept(args.get('topic', ''), args.get('details', ''))
                        tool_result = "Concept learned and saved."

                    else:
                        tool_result = "Unknown tool."

                except Exception as e:
                    logger.error(f"Error executing tool {func_name}: {e}")
                    tool_result = f"Tool '{func_name}' failed to execute due to an internal error."

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": str(tool_result),
                })

            try:
                second_response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.8,
                    max_tokens=600,
                )
            except Exception as second_err:
                logger.warning(f"Second completion attempt failed, retrying once: {second_err}")
                await asyncio.sleep(1.5)
                second_response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.8,
                    max_tokens=600,
                )
            reply = second_response.choices[0].message.content

        if not reply:
            reply = message.content

        reply = clean_ai_text(reply)

        if not reply:
            reply = "Huh? I totally spaced out for a second. What did you say? 🙈"

        chat_history[chat_id].append({"role": "user", "content": user_text})
        chat_history[chat_id].append({"role": "assistant", "content": reply})
        await update.message.reply_text(reply)

        # Keep last N turns (user+assistant pairs)
        chat_history[chat_id] = chat_history[chat_id][-(HISTORY_TURNS_TO_KEEP * 2):]

    except Exception as e:
        logger.error(f"Error calling AI API: {e}")
        logger.error(traceback.format_exc())  # full traceback -> check your terminal/logs to see the EXACT cause
        if "429" in str(e) or "rate limit" in str(e).lower():
            await update.message.reply_text(
                "Ugh, I'm talking to too many people right now and my brain needs a quick breather! 😵‍💫 "
                "Give me like 10 seconds and try again!"
            )
        else:
            await update.message.reply_text("Ugh, my brain is having a glitch right now. 🙄 Give me a sec and try again!")

# ----------------------------------------------------------------------------
# Background jobs
# ----------------------------------------------------------------------------
async def check_reminders(context: ContextTypes.DEFAULT_TYPE):
    if reminders_collection is None:
        return
    try:
        current_time = datetime.now().isoformat()
        due_reminders = list(reminders_collection.find({"sent": False, "remind_at": {"$lte": current_time}}))

        for r in due_reminders:
            reminder_id = r["reminder_id"]
            chat_id = r["chat_id"]
            task = r["task"]
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"🔔 Wakey wakey!\n\nHey! It's time to: {task}\n\nDon't make me nag you! 😘"
                )
                reminders_collection.update_one({"_id": r["_id"]}, {"$set": {"sent": True}})
            except Exception as e:
                logger.error(f"Failed to send reminder {reminder_id}: {e}")
    except Exception as e:
        logger.error(f"Error checking reminders from MongoDB: {e}")

async def proactive_message(context: ContextTypes.DEFAULT_TYPE):
    """Occasionally check in on users."""
    if memory_collection is None:
        return
    try:
        users = memory_collection.find({}, {"chat_id": 1})
        for user in users:
            chat_id = user["chat_id"]
            msg = "Hey... I was just thinking about you. How is your day going? ❤️"
            try:
                await context.bot.send_message(chat_id=chat_id, text=msg)
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Error sending proactive message: {e}")

# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def main():
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "your_telegram_bot_token_here":
        print("Error: Please set TELEGRAM_TOKEN in your .env file")
        return
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        print("Error: Please set GROQ_API_KEY in your .env file")
        return
    if not MONGO_URI_RAW or MONGO_URI_RAW.startswith("mongodb+srv://<username>"):
        print("Error: Please set a valid MONGO_URI in your .env file")
        return

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.job_queue.run_repeating(check_reminders, interval=60, first=10)
    application.job_queue.run_repeating(proactive_message, interval=21600, first=3600)

    print(f"AI Pro Girlfriend Bot running with model '{MODEL_NAME}' + live news + reminders...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
