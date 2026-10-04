from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import sqlite3
import os


# --------------------------------
# 1. Load environment variables
# --------------------------------

load_dotenv()


# --------------------------------
# 2. Create OpenAI client
# --------------------------------

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# --------------------------------
# 3. Create FastAPI application
# --------------------------------

app = FastAPI()


# --------------------------------
# 4. Database file
# --------------------------------

DATABASE_NAME = "chatbot.db"


# --------------------------------
# 5. Create database table
# --------------------------------

def initialize_database():

    connection = sqlite3.connect(DATABASE_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            conversation_id TEXT NOT NULL,

            role TEXT NOT NULL,

            content TEXT NOT NULL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
    """)

    connection.commit()

    connection.close()


# Run when application starts
initialize_database()


# --------------------------------
# 6. Request models
# --------------------------------

class ChatRequest(BaseModel):
    conversation_id: str
    message: str


class ClearRequest(BaseModel):
    conversation_id: str


# --------------------------------
# 7. Save message to database
# --------------------------------

def save_message(
    conversation_id: str,
    role: str,
    content: str
):

    connection = sqlite3.connect(DATABASE_NAME)

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO messages (
            conversation_id,
            role,
            content
        )
        VALUES (?, ?, ?)
        """,
        (
            conversation_id,
            role,
            content
        )
    )

    connection.commit()

    connection.close()


# --------------------------------
# 8. Get conversation history
# --------------------------------

def get_conversation_history(
    conversation_id: str
):

    connection = sqlite3.connect(DATABASE_NAME)

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT role, content
        FROM messages
        WHERE conversation_id = ?
        ORDER BY id
        """,
        (conversation_id,)
    )

    rows = cursor.fetchall()

    connection.close()


    conversation_history = []


    for row in rows:

        role = row[0]

        content = row[1]


        conversation_history.append({
            "role": role,
            "content": content
        })


    return conversation_history


# --------------------------------
# 9. Serve frontend
# --------------------------------

@app.get("/")
def home():

    return FileResponse("index.html")


# --------------------------------
# 10. Chat endpoint
# --------------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    conversation_id = request.conversation_id


    # Get previous messages from SQLite
    conversation_history = (
        get_conversation_history(
            conversation_id
        )
    )


    # Add current user message
    conversation_history.append({
        "role": "user",
        "content": request.message
    })


    print("\nConversation ID:")
    print(conversation_id)


    print("\nHistory being sent to AI:")
    print(conversation_history)


    # Send conversation to LLM
    response = client.responses.create(

        model="gpt-6-luna",

        instructions="""
        You are a helpful AI assistant.
        Give simple, clear and friendly answers.
        """,

        input=conversation_history
    )


    # Extract AI response
    ai_reply = response.output_text


    # Save user's message permanently
    save_message(
        conversation_id,
        "user",
        request.message
    )


    # Save AI response permanently
    save_message(
        conversation_id,
        "assistant",
        ai_reply
    )


    return {

        "conversation_id":
            conversation_id,

        "user_message":
            request.message,

        "bot_response":
            ai_reply
    }


# --------------------------------
# 11. Clear one conversation
# --------------------------------

@app.post("/clear")
def clear_chat(request: ClearRequest):

    connection = sqlite3.connect(
        DATABASE_NAME
    )

    cursor = connection.cursor()


    cursor.execute(
        """
        DELETE FROM messages
        WHERE conversation_id = ?
        """,
        (request.conversation_id,)
    )


    connection.commit()

    connection.close()


    print(
        f"\nConversation "
        f"{request.conversation_id} cleared!"
    )


    return {
        "message":
        "Conversation cleared successfully"
    }