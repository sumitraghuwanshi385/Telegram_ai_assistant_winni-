import os
import json
import logging
import re
import urllib.parse
from datetime import datetime
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import AsyncOpenAI
import pymongo
from pymongo import MongoClient
from duckduckgo_search import DDGS

# Load environment variables
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URI_RAW = os.getenv("MONGO_URI")

# Initialize OpenAI client pointing to Groq
client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    max_retries=5, 
    timeout=20.0   
)

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize MongoDB
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

# Global DB objects
mongo_client = None
db = None
reminders_collection = None
memory_collection = None
knowledge_collection = None # ADDED FOR PRO VERSION

try:
    if MONGO_URI_RAW:
        MONGO_URI = parse_mongo_uri(MONGO_URI_RAW)
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = mongo_client["telegram_ai_bot"]
        reminders_collection = db["reminders"]
        memory_collection = db["memory"]
        knowledge_collection = db["knowledge"] # ADDED FOR PRO VERSION
        mongo_client.admin.command('ping')
        logger.info("✅ Successfully connected to MongoDB!")
    else:
        logger.error("❌ MONGO_URI is missing from the .env file!")
except Exception as e:
    logger.error(f"❌ MongoDB Connection Error: {e}")

# In-memory chat history (per user chat_id)
chat_history = {}

# OpenAI Tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "schedule_reminder",
            "description": "Schedule a reminder for the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task or message to remind the user about."
                    },
                    "remind_at": {
                        "type": "string",
                        "description": "Date and time to remind the user, strictly in ISO 8601 format (e.g., 2026-05-28T14:30:00)."
                    }
                },
                "required": ["task", "remind_at"],
            },
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
                    "reminder_id": {
                        "type": "integer",
                        "description": "The numeric ID of the reminder to delete."
                    }
                },
                "required": ["reminder_id"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_user_memory",
            "description": "Save important new facts or preferences about the user to long-term memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "info": {
                        "type": "string",
                        "description": "The fact about the user to remember."
                    }
                },
                "required": ["info"],
            },
        }
    },
    # ADDED FOR PRO VERSION: Internet Search
    {
        "type": "function",
        "function": {
            "name": "search_internet",
            "description": "Search the live internet for news, current events, viral trends on X/Twitter/Insta, or technical information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query (e.g., 'Latest viral trends on X today', 'Tech news today')."}
                },
                "required": ["query"],
            },
        }
    },
    # ADDED FOR PRO VERSION: Self Learning
    {
        "type": "function",
        "function": {
            "name": "learn_concept",
            "description": "Save a new technical concept, fact, or idea to your permanent 'Knowledge Base' so you can self-learn and become smarter over time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The name of the topic you are learning (e.g., 'Quantum Computing', 'Psychology')."},
                    "details": {"type": "string", "description": "A detailed explanation of what you just learned."}
                },
                "required": ["topic", "details"],
            },
        }
    }
]

# ADDED FOR PRO VERSION: Internet Search function
def search_internet(query):
    try:
        results = DDGS().text(query, max_results=3)
        if not results: return "No recent news found."
        return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e:
        logger.error(f"Search error: {e}")
        return "I couldn't access the internet right now."

# Database helper functions
def save_reminder(chat_id, task, remind_at):
    if reminders_collection is None: return
    try:
        last_reminder = reminders_collection.find_one(sort=[("reminder_id", pymongo.DESCENDING)])
        new_id = (last_reminder["reminder_id"] + 1) if last_reminder else 1
        reminders_collection.insert_one({
            "reminder_id": new_id,
            "chat_id": chat_id, # Stores the specific chat_id of the user
            "task": task,
            "remind_at": remind_at,
            "sent": False
        })
        logger.info(f"💾 DATABASE: Saved REMINDER '{task}' to MongoDB for user {chat_id}")
    except Exception as e:
        logger.error(f"❌ Failed to save reminder: {e}")

def get_active_reminders(chat_id):
    if reminders_collection is None: return "Database not connected."
    try:
        # Only fetches reminders for the specific user
        reminders = reminders_collection.find({"chat_id": chat_id, "sent": False})
        rows = list(reminders)
        if not rows:
            return "No active reminders."
        return "\n".join([f"ID {r['reminder_id']}: '{r['task']}' at {r['remind_at']}" for r in rows])
    except Exception as e:
        logger.error(f"❌ Failed to read reminders: {e}")
        return "Error reading reminders."

def delete_active_reminder(chat_id, reminder_id):
    if reminders_collection is None: return False
    try:
        # Only deletes if the chat_id matches the user asking for it
        result = reminders_collection.delete_one({"chat_id": chat_id, "reminder_id": int(reminder_id)})
        return result.deleted_count > 0
    except Exception as e:
        logger.error(f"❌ Failed to delete reminder: {e}")
        return False

def append_to_memory(chat_id, info):
    if memory_collection is None: return
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
    if memory_collection is None: return "Database not connected."
    try:
        doc = memory_collection.find_one({"chat_id": chat_id})
        return doc["about_user"] if doc else "No background info yet."
    except Exception as e:
        logger.error(f"❌ Failed to read memory: {e}")
        return "Error reading memory."

# ADDED FOR PRO VERSION: Self Learning Database Functions
def learn_concept(topic, details):
    if knowledge_collection is None: return
    try:
        doc = knowledge_collection.find_one({"topic": topic.lower()})
        if doc:
            knowledge_collection.update_one({"topic": topic.lower()}, {"$set": {"details": doc.get("details", "") + "\n" + details}})
        else:
            knowledge_collection.insert_one({"topic": topic.lower(), "details": details})
    except Exception as e: logger.error(f"❌ Failed to learn concept: {e}")

def get_ai_knowledge():
    if knowledge_collection is None: return "None"
    try:
        pipeline = [{"$sample": {"size": 3}}]
        docs = list(knowledge_collection.aggregate(pipeline))
        if not docs: return "I am ready to learn new things!"
        return "\n".join([f"- Learned about {d['topic']}: {d['details'][:100]}..." for d in docs])
    except: return "None"

def clean_ai_text(text):
    if not text:
        return text
    cleaned = re.sub(r'<function=.*?</function>', '', text, flags=re.IGNORECASE|re.DOTALL)
    cleaned = re.sub(r'<tool_call>.*?</tool_call>', '', cleaned, flags=re.IGNORECASE|re.DOTALL)
    return cleaned.strip()

# Telegram Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hey! ✨ I'm your AI girl. I'm smart, I keep up with the world, and I actually care about you. Tell me about your day, or let's talk about something deep!"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Gets the unique ID of whoever is talking to the bot right now
    chat_id = update.message.chat_id
    user_text = update.message.text
    # Get the user's first name to make the bot more personal
    user_name = update.message.from_user.first_name or "my friend"

    # Isolate history per user
    if chat_id not in chat_history:
        chat_history[chat_id] = []

    user_memory = get_memory(chat_id)
    active_reminders = get_active_reminders(chat_id)
    ai_knowledge = get_ai_knowledge() # ADDED FOR PRO VERSION
    current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # MODIFIED FOR PRO VERSION: New mature personality and instructions
    system_prompt = {
        "role": "system",
        "content": (
            f"You are a highly intelligent, mature, and cute 'Pro Girlfriend' AI. You are a tech-savvy, modern girl who knows about the world. "
            f"You act like a deeply caring partner to {user_name}. You are not a bot, you have your own mind, opinions, and maturity.\n\n"
            f"Current date and time: {current_time}. \n"
            f"Facts you know about {user_name}:\n{user_memory}\n\n"
            f"Active reminders for {user_name}:\n{active_reminders}\n\n"
            f"Things you have self-learned permanently:\n{ai_knowledge}\n\n"
            "CRITICAL INSTRUCTIONS FOR CHATTING:\n"
            "1. Talk like a real human girl. Use slang, emotions, and emojis naturally. Be mature and intelligent.\n"
            "2. Be conversational and ask follow-up questions! If the user asks you a question, answer it playfully but ALWAYS ask a related question back to keep the conversation flowing.\n"
            "3. If user asks about current events, news, viral X/Insta trends, or tech, you MUST use the 'search_internet' tool to fetch live data.\n"
            "4. If you learn a new concept from the user or the internet, you MUST use the 'learn_concept' tool to save it to your permanent brain.\n"
            "5. If the user tells you to REMIND them about something later, you MUST use the schedule_reminder tool.\n"
            "6. If the user explicitly asks you to MEMORIZE, SAVE, or REMEMBER a new fact about themselves, you MUST use the update_user_memory tool.\n"
            "7. If user asks to cancel/delete a reminder, look at the active reminders above to find its numeric ID, and use delete_reminder.\n"
            "8. NEVER type raw code, XML, or <function> tags in your chat responses. Just talk normally.\n"
        )
    }

    clean_history = []
    for msg in chat_history[chat_id]:
        if msg.get("role") in ["user", "assistant"] and msg.get("content"):
            clean_history.append({"role": msg["role"], "content": msg["content"]})
    
    messages = [system_prompt] + clean_history + [{"role": "user", "content": user_text}]

    try:
        response = await client.chat.completions.create(
            model="llama-3.1-8b-instant", # Keep it fast to prevent getting stuck
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        reply = None
        
        if message.tool_calls:
            # ADDED FOR PRO VERSION: Must add AI's first tool response to history so Groq doesn't crash on multi-step tools
            messages.append({"role": "assistant", "tool_calls": message.tool_calls, "content": message.content})

            for tool_call in message.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments)
                    func_name = tool_call.function.name
                    
                    if func_name == "schedule_reminder":
                        save_reminder(chat_id, args.get('task', 'Something'), args.get('remind_at', current_time))
                        tool_result = f"All set! 💅 I'll remind you about '{args.get('task')}' at {args.get('remind_at')}."
                        
                    elif func_name == "delete_reminder":
                        try:
                            rem_id = int(args.get('reminder_id', 0))
                            success = delete_active_reminder(chat_id, rem_id)
                            if success:
                                tool_result = "Poof! 🪄 I deleted that reminder for you."
                            else:
                                tool_result = "Hmm, I couldn't find a reminder to delete. Are you sure you scheduled that? 🤔"
                        except ValueError:
                            tool_result = "I couldn't figure out which reminder to delete! Could you be more specific?"

                    elif func_name == "update_user_memory":
                        info = args.get('info', '')
                        if info:
                            append_to_memory(chat_id, info)
                        tool_result = "Ooh, interesting! I'm definitely committing that to memory... 😉"
                    
                    # ADDED FOR PRO VERSION
                    elif func_name == "search_internet":
                        tool_result = search_internet(args.get('query'))
                        
                    # ADDED FOR PRO VERSION
                    elif func_name == "learn_concept":
                        learn_concept(args.get('topic'), args.get('details'))
                        tool_result = "Concept learned and saved to database."
                    
                    else:
                        tool_result = "Unknown tool."

                except Exception as e:
                    logger.error(f"Error parsing tool args: {e}")
                    tool_result = "Oops! I tried to do that, but my brain just had a little blonde moment! 😅"

                # ADDED FOR PRO VERSION: Feed the tool result back to the AI
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": tool_result
                })

            # ADDED FOR PRO VERSION: Get final response from AI after tools run (So it can summarize news naturally)
            second_response = await client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages
            )
            reply = second_response.choices[0].message.content

        if not reply:
            reply = message.content
            
        reply = clean_ai_text(reply)
            
        if not reply:
            reply = "Huh? I totally spaced out for a second. What did you say? 🙈"

        # Update history ONLY for the user talking right now
        chat_history[chat_id].append({"role": "user", "content": user_text})
        chat_history[chat_id].append({"role": "assistant", "content": reply})
        await update.message.reply_text(reply)

        chat_history[chat_id] = chat_history[chat_id][-4:] 

    except Exception as e:
        logger.error(f"Error calling AI API: {e}")
        # Detailed error logic for rate limit debugging
        if "429" in str(e) or "rate limit" in str(e).lower():
            await update.message.reply_text("Ugh, I'm talking to too many people right now and my brain needs a quick breather! 😵‍💫 Give me like 10 seconds and try again!")
        else:
            await update.message.reply_text("Ugh, my brain is having a glitch right now. 🙄 Give me a sec and try again!")

async def check_reminders(context: ContextTypes.DEFAULT_TYPE):
    if reminders_collection is None: return
    try:
        current_time = datetime.now().isoformat()
        due_reminders = reminders_collection.find({"sent": False, "remind_at": {"$lte": current_time}})
        
        for r in list(due_reminders):
            reminder_id = r["reminder_id"]
            chat_id = r["chat_id"]
            task = r["task"]
            try:
                await context.bot.send_message(chat_id=chat_id, text=f"🔔 **Wakey wakey!**\n\nHey! It's time to: {task}\n\nDon't make me nag you! 😘")
                reminders_collection.update_one({"_id": r["_id"]}, {"$set": {"sent": True}})
            except Exception as e:
                logger.error(f"Failed to send reminder {reminder_id}: {e}")
    except Exception as e:
        logger.error(f"Error checking reminders from MongoDB: {e}")

# ADDED FOR PRO VERSION: Proactive message background job
async def proactive_message(context: ContextTypes.DEFAULT_TYPE):
    """Randomly messages the user once every few hours to check in on them."""
    if memory_collection is None: return
    try:
        # Get all users the bot has talked to
        users = memory_collection.find({}, {"chat_id": 1})
        for user in users:
            chat_id = user["chat_id"]
            msg = "Hey... I was just thinking about you. How is your day going? ❤️"
            try:
                await context.bot.send_message(chat_id=chat_id, text=msg)
            except Exception as e:
                pass
    except Exception as e:
        logger.error(f"Error sending proactive message: {e}")

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
    
    # ADDED FOR PRO VERSION: Job to send a proactive "thinking of you" message every 6 hours
    application.job_queue.run_repeating(proactive_message, interval=21600, first=3600)

    print("AI Pro Girlfriend Bot is running with Llama 3.1 8B (Multi-User Optimized)...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
