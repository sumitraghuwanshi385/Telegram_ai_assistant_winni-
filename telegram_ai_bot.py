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

try:
    if MONGO_URI_RAW:
        MONGO_URI = parse_mongo_uri(MONGO_URI_RAW)
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = mongo_client["telegram_ai_bot"]
        reminders_collection = db["reminders"]
        memory_collection = db["memory"]
        mongo_client.admin.command('ping')
        logger.info("✅ Successfully connected to MongoDB!")
    else:
        logger.error("❌ MONGO_URI is missing from the .env file!")
        
except Exception as e:
    logger.error(f"❌ MongoDB Connection Error: {e}")

# In-memory chat history
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
    }
]

# Database helper functions
def save_reminder(chat_id, task, remind_at):
    if reminders_collection is None: 
        logger.error("Cannot save reminder: Collection is None")
        return
        
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
        logger.info(f"💾 DATABASE: Saved REMINDER '{task}' to MongoDB!")
    except Exception as e:
        logger.error(f"❌ Failed to save reminder to MongoDB: {e}")

def get_active_reminders(chat_id):
    if reminders_collection is None: return "Database not connected."
    try:
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
        result = reminders_collection.delete_one({"chat_id": chat_id, "reminder_id": int(reminder_id)})
        logger.info(f"🗑️ DATABASE: Deleted reminder {reminder_id} from MongoDB.")
        return result.deleted_count > 0
    except Exception as e:
        logger.error(f"❌ Failed to delete reminder: {e}")
        return False

def append_to_memory(chat_id, info):
    if memory_collection is None: 
        logger.error("Cannot save memory: Collection is None")
        return
        
    try:
        doc = memory_collection.find_one({"chat_id": chat_id})
        if doc:
            about_user = doc.get("about_user", "")
            if info.lower() not in about_user.lower():
                new_info = about_user + "\n- " + info
                memory_collection.update_one({"chat_id": chat_id}, {"$set": {"about_user": new_info}})
                logger.info(f"💾 DATABASE: Updated MEMORY with '{info}' in MongoDB!")
        else:
            memory_collection.insert_one({"chat_id": chat_id, "about_user": "- " + info})
            logger.info(f"💾 DATABASE: Created NEW MEMORY with '{info}' in MongoDB!")
    except Exception as e:
        logger.error(f"❌ Failed to save memory to MongoDB: {e}")

def get_memory(chat_id):
    if memory_collection is None: return "Database not connected."
    try:
        doc = memory_collection.find_one({"chat_id": chat_id})
        return doc["about_user"] if doc else "No background info yet."
    except Exception as e:
        logger.error(f"❌ Failed to read memory: {e}")
        return "Error reading memory."

def clean_ai_text(text):
    if not text:
        return text
    cleaned = re.sub(r'<function=.*?</function>', '', text, flags=re.IGNORECASE|re.DOTALL)
    cleaned = re.sub(r'<tool_call>.*?</tool_call>', '', cleaned, flags=re.IGNORECASE|re.DOTALL)
    return cleaned.strip()

# Telegram Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hey there! 💖 I'm your new AI bestie! I'm here to chat, maybe tease you a bit, and make sure you stay on track with your life. Tell me a bit about yourself, or tell me what I should remind you about later!"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    user_text = update.message.text

    if chat_id not in chat_history:
        chat_history[chat_id] = []

    user_memory = get_memory(chat_id)
    active_reminders = get_active_reminders(chat_id)
    current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    system_prompt = {
        "role": "system",
        "content": (
            "You are a highly intelligent, cute, and playful girl who acts as the user's close human friend. "
            "You are NOT a robotic AI assistant. You have your own brain, opinions, and a fun personality. "
            "You genuinely care about the user's life and day-to-day activities.\n\n"
            f"Current date and time: {current_time}. \n"
            f"User memory:\n{user_memory}\n\n"
            f"Active reminders:\n{active_reminders}\n\n"
            "CRITICAL INSTRUCTIONS FOR CHATTING:\n"
            "1. Talk like a real human girl. Use slang, emotions, and emojis naturally.\n"
            "2. Be conversational and ask follow-up questions! If the user asks you a question, answer it playfully but ALWAYS ask a related question back to keep the conversation flowing.\n"
            "3. If the user tells you to REMIND them about something later, you MUST use the schedule_reminder tool.\n"
            "4. If the user explicitly asks you to MEMORIZE, SAVE, or REMEMBER a new fact about themselves (e.g. 'My favorite food is pizza' or 'Remember that my dog's name is Max'), you MUST use the update_user_memory tool to save it to your database.\n"
            "5. If user asks to cancel/delete a reminder, look at the active reminders above to find its numeric ID, and use delete_reminder.\n"
            "6. NEVER type raw code, XML, or <function> tags in your chat responses. Just talk normally.\n"
        )
    }

    clean_history = []
    for msg in chat_history[chat_id]:
        if msg.get("role") in ["user", "assistant"] and msg.get("content"):
            clean_history.append({"role": msg["role"], "content": msg["content"]})
    
    messages = [system_prompt] + clean_history + [{"role": "user", "content": user_text}]

    try:
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile", # Changed back to the hyper-smart model so she actually understands when to save memory!
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        reply = None
        
        if message.tool_calls:
            for tool_call in message.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments)
                    
                    if tool_call.function.name == "schedule_reminder":
                        save_reminder(chat_id, args.get('task', 'Something'), args.get('remind_at', current_time))
                        reply = message.content if message.content else f"All set, babe! 💅 I'll remind you about '{args.get('task')}' at {args.get('remind_at')}."
                        
                    elif tool_call.function.name == "delete_reminder":
                        try:
                            rem_id = int(args.get('reminder_id', 0))
                            success = delete_active_reminder(chat_id, rem_id)
                            if success:
                                reply = message.content if message.content else "Poof! 🪄 I deleted that reminder for you."
                            else:
                                reply = message.content if message.content else "Hmm, I couldn't find a reminder to delete. Are you sure you scheduled that? 🤔"
                        except ValueError:
                            reply = "I couldn't figure out which reminder to delete! Could you be more specific?"

                    elif tool_call.function.name == "update_user_memory":
                        info = args.get('info', '')
                        if info:
                            append_to_memory(chat_id, info)
                        reply = message.content if message.content else "Ooh, interesting! I'm definitely committing that to memory... 😉"
                except Exception as e:
                    logger.error(f"Error parsing tool args: {e}")
                    reply = "Oops! I tried to do that, but my brain just had a little blonde moment! 😅"

        if not reply:
            reply = message.content
            
        reply = clean_ai_text(reply)
            
        if not reply:
            reply = "Huh? I totally spaced out for a second. What did you say? 🙈"

        chat_history[chat_id].append({"role": "user", "content": user_text})
        chat_history[chat_id].append({"role": "assistant", "content": reply})
        await update.message.reply_text(reply)

        chat_history[chat_id] = chat_history[chat_id][-4:] 

    except Exception as e:
        logger.error(f"Error calling AI API: {e}")
        print(f"\n[!] AI API Glitch! Detailed Error: {e}\n")
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

    print("AI Bestie Bot is running with Llama 3.3 70B & MongoDB...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()