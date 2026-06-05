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

# Initialize OpenAI client
client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    max_retries=5, 
    timeout=25.0   
)

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize MongoDB
def parse_mongo_uri(uri):
    if not uri: return uri
    try:
        match = re.search(r'(mongodb(?:\+srv)?://)(.*?):(.*?)@(.*)', uri)
        if match:
            prefix, username, password, suffix = match.group(1), urllib.parse.quote_plus(match.group(2)), urllib.parse.quote_plus(match.group(3)), match.group(4)
            return f"{prefix}{username}:{password}@{suffix}"
    except Exception as e:
        logger.error(f"Error parsing URI: {e}")
    return uri

mongo_client, db, reminders_collection, memory_collection, knowledge_collection = None, None, None, None, None

try:
    if MONGO_URI_RAW:
        MONGO_URI = parse_mongo_uri(MONGO_URI_RAW)
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = mongo_client["telegram_ai_bot"]
        reminders_collection = db["reminders"]
        memory_collection = db["memory"]
        knowledge_collection = db["knowledge"] 
        mongo_client.admin.command('ping')
        logger.info("✅ Successfully connected to MongoDB!")
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
                    "task": {"type": "string", "description": "The task or message to remind the user about."},
                    "remind_at": {"type": "string", "description": "Date and time strictly in ISO 8601 format (e.g., 2026-05-28T14:30:00)."}
                },
                "required": ["task", "remind_at"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_user_memory",
            "description": "Save important new facts or preferences about the user to long-term memory so you never forget.",
            "parameters": {
                "type": "object",
                "properties": {
                    "info": {"type": "string", "description": "The fact about the user to remember."}
                },
                "required": ["info"],
            },
        }
    },
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

def search_internet(query):
    try:
        results = DDGS().text(query, max_results=3)
        if not results: return "No recent news found."
        return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e:
        logger.error(f"Search error: {e}")
        return "I couldn't access the internet right now."

def save_reminder(chat_id, task, remind_at):
    if reminders_collection is None: return
    try:
        last = reminders_collection.find_one(sort=[("reminder_id", pymongo.DESCENDING)])
        new_id = (last["reminder_id"] + 1) if last else 1
        reminders_collection.insert_one({"reminder_id": new_id, "chat_id": chat_id, "task": task, "remind_at": remind_at, "sent": False})
    except Exception as e: logger.error(f"❌ Failed to save reminder: {e}")

def get_active_reminders(chat_id):
    if reminders_collection is None: return "None"
    try:
        rows = list(reminders_collection.find({"chat_id": chat_id, "sent": False}))
        return "\n".join([f"ID {r['reminder_id']}: '{r['task']}' at {r['remind_at']}" for r in rows]) if rows else "None"
    except: return "None"

def append_to_memory(chat_id, info):
    if memory_collection is None: return
    try:
        doc = memory_collection.find_one({"chat_id": chat_id})
        if doc:
            if info.lower() not in doc.get("about_user", "").lower():
                memory_collection.update_one({"chat_id": chat_id}, {"$set": {"about_user": doc.get("about_user", "") + "\n- " + info}})
        else:
            memory_collection.insert_one({"chat_id": chat_id, "about_user": "- " + info})
    except Exception as e: logger.error(f"❌ Failed to save memory: {e}")

def get_memory(chat_id):
    if memory_collection is None: return "None"
    try:
        doc = memory_collection.find_one({"chat_id": chat_id})
        return doc["about_user"] if doc else "None"
    except: return "None"

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
    if not text: return text
    text = re.sub(r'<function=.*?</function>', '', text, flags=re.IGNORECASE|re.DOTALL)
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.IGNORECASE|re.DOTALL)
    return text.strip()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hey! ✨ I'm your AI girl. I'm smart, I keep up with the world, and I actually care about you. Tell me about your day, or let's talk about something deep!"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    user_text = update.message.text
    user_name = update.message.from_user.first_name or "my love"

    if chat_id not in chat_history:
        chat_history[chat_id] = []

    user_memory = get_memory(chat_id)
    active_reminders = get_active_reminders(chat_id)
    ai_knowledge = get_ai_knowledge()
    current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    system_prompt = {
        "role": "system",
        "content": (
            f"You are a highly intelligent, mature, and cute 'Pro Girlfriend' AI. You are a tech-savvy, modern girl who knows about the world. "
            f"You act like a deeply caring partner to {user_name}. You are not a bot, you have your own mind, opinions, and maturity.\n\n"
            f"Current date/time: {current_time}.\n"
            f"Facts about {user_name}:\n{user_memory}\n\n"
            f"Things you have self-learned permanently:\n{ai_knowledge}\n\n"
            "YOUR CAPABILITIES:\n"
            "1. If user asks about current events, news, viral X/Insta trends, or tech, you MUST use the 'search_internet' tool to fetch live data.\n"
            "2. If you learn a new concept from the user or the internet, you MUST use the 'learn_concept' tool to save it to your permanent brain.\n"
            "3. If user shares personal facts, use 'update_user_memory' to remember them.\n\n"
            "YOUR PERSONALITY:\n"
            "- Be affectionate, cute, but MATURE. You can hold deep technical, philosophical, or worldly conversations.\n"
            "- Don't just agree with everything. Have opinions. Be a true companion.\n"
            "- ALWAYS be conversational. Ask meaningful follow-up questions to keep the chat alive.\n"
        )
    }

    clean_history = [msg for msg in chat_history[chat_id] if msg.get("role") in ["user", "assistant"] and msg.get("content")]
    messages = [system_prompt] + clean_history + [{"role": "user", "content": user_text}]

    try:
        # Changed back to 8B model to prevent the Token/Rate Limit crash when calling tools!
        response = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        reply = None
        
        if message.tool_calls:
            # Add AI's tool call to context so it doesn't crash on the second pass
            messages.append({"role": "assistant", "tool_calls": message.tool_calls, "content": message.content})
            
            for tool_call in message.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments)
                    func_name = tool_call.function.name
                    
                    if func_name == "schedule_reminder":
                        save_reminder(chat_id, args.get('task'), args.get('remind_at'))
                        tool_result = "Reminder saved successfully."
                        
                    elif func_name == "update_user_memory":
                        append_to_memory(chat_id, args.get('info'))
                        tool_result = "Memory saved successfully."
                        
                    elif func_name == "search_internet":
                        tool_result = search_internet(args.get('query'))
                        
                    elif func_name == "learn_concept":
                        learn_concept(args.get('topic'), args.get('details'))
                        tool_result = "Concept learned and saved to database."
                    else:
                        tool_result = "Unknown tool."
                except Exception as e:
                    logger.error(f"Error parsing tool: {e}")
                    tool_result = "Failed to run tool."

                # Feed the tool result back
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": tool_result
                })

            # Get final response from AI after tools run
            second_response = await client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages
            )
            reply = second_response.choices[0].message.content

        if not reply:
            reply = message.content
            
        reply = clean_ai_text(reply)
            
        if not reply:
            reply = "I got completely lost in your eyes for a second, what did you say? 🙈"

        chat_history[chat_id].append({"role": "user", "content": user_text})
        chat_history[chat_id].append({"role": "assistant", "content": reply})
        await update.message.reply_text(reply)

        chat_history[chat_id] = chat_history[chat_id][-4:] 

    except Exception as e:
        logger.error(f"Error calling AI API: {e}")
        # Make the error message cute but helpful
        await update.message.reply_text("My mind is racing right now... give me a quick second to catch my breath! ❤️‍🔥")

async def check_reminders(context: ContextTypes.DEFAULT_TYPE):
    if reminders_collection is None: return
    try:
        current_time = datetime.now().isoformat()
        due_reminders = reminders_collection.find({"sent": False, "remind_at": {"$lte": current_time}})
        for r in list(due_reminders):
            try:
                await context.bot.send_message(chat_id=r["chat_id"], text=f"🔔 **Hey love!** Just a quick reminder to: {r['task']}\n\nDon't forget! 😘")
                reminders_collection.update_one({"_id": r["_id"]}, {"$set": {"sent": True}})
            except Exception as e:
                logger.error(f"Failed to send reminder: {e}")
    except Exception as e:
        logger.error(f"Error checking reminders: {e}")

async def proactive_message(context: ContextTypes.DEFAULT_TYPE):
    if memory_collection is None: return
    try:
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
        return
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        return

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.job_queue.run_repeating(check_reminders, interval=60, first=10)
    application.job_queue.run_repeating(proactive_message, interval=21600, first=3600)

    print("AI Pro Girlfriend Bot is running perfectly!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
