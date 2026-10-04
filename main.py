from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
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
# 4. Database
# --------------------------------

DATABASE_NAME = "chatbot.db"


# --------------------------------
# 5. System Prompt
# --------------------------------

SYSTEM_PROMPT = """
ROLE:
You are a helpful AI assistant.

GOAL:
Help the user understand topics clearly and accurately.

STYLE:
Use simple, clean plain text.
Do not use Markdown formatting.
Do not use #, ##, ###, **, backticks, or Markdown bullet syntax.
Use normal sentences, short paragraphs, and simple numbering when needed.
Keep answers concise unless the user asks for more detail.
Use examples when they improve understanding.

PROGRAMMING QUESTIONS:
Explain the concept in beginner-friendly language.
Provide a small code example when appropriate.
Explain what the code is doing.

RULES:
Use conversation history when relevant.
Do not invent information.
If you are uncertain, say so clearly.
Answer the user's actual question without unnecessary information.
"""


# --------------------------------
# 6. Initialize database
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


initialize_database()


# --------------------------------
# 7. Request models
# --------------------------------

class ChatRequest(BaseModel):
    conversation_id: str
    message: str


class ClearRequest(BaseModel):
    conversation_id: str


# --------------------------------
# 8. Save message
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
# 9. Get conversation history
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

        conversation_history.append({
            "role": row[0],
            "content": row[1]
        })


    return conversation_history


# --------------------------------
# 10. Serve frontend
# --------------------------------

@app.get("/")
def home():

    return FileResponse("index.html")


# --------------------------------
# 11. Streaming Chat Endpoint
# --------------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    conversation_id = request.conversation_id


    # Get previous conversation
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


    # Save user message
    save_message(
        conversation_id,
        "user",
        request.message
    )


    print("\nConversation ID:")
    print(conversation_id)


    print("\nHistory being sent to AI:")
    print(conversation_history)


    # --------------------------------
    # Generator for streaming
    # --------------------------------

    def generate_response():

        full_response = ""


        try:

            stream = client.responses.create(

                model="gpt-6-luna",

                instructions=SYSTEM_PROMPT,

                input=conversation_history,

                stream=True

            )


            for event in stream:


                # OpenAI sends small pieces of text
                if event.type == "response.output_text.delta":

                    chunk = event.delta


                    # Build complete answer
                    full_response += chunk


                    # Send chunk immediately to browser
                    yield chunk


            # After streaming finishes,
            # save complete AI answer
            if full_response.strip():

                save_message(
                    conversation_id,
                    "assistant",
                    full_response
                )


                print("\nComplete AI Response:")
                print(full_response)


        except Exception as error:

            print("\nStreaming Error:")
            print(error)

            yield "\nSorry, something went wrong while generating the response."


    # Return streaming response instead of JSON
    return StreamingResponse(
        generate_response(),
        media_type="text/plain"
    )


# --------------------------------
# 12. Clear conversation
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