from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os


# -----------------------------
# 1. Load environment variables
# -----------------------------

load_dotenv()


# -----------------------------
# 2. Create OpenAI client
# -----------------------------

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# -----------------------------
# 3. Create FastAPI application
# -----------------------------

app = FastAPI()


# -----------------------------
# 4. Conversation memory
# -----------------------------

conversation_history = []


# -----------------------------
# 5. Request model
# -----------------------------

class ChatRequest(BaseModel):
    message: str


# -----------------------------
# 6. Show chatbot webpage
# -----------------------------

@app.get("/")
def home():

    return FileResponse("index.html")


# -----------------------------
# 7. Chat endpoint
# -----------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    # Add the user's new message to memory
    conversation_history.append({
        "role": "user",
        "content": request.message
    })


    # Optional:
    # lets us see memory in the VS Code terminal
    print("\nConversation History:")
    print(conversation_history)


    # Send the entire conversation to the LLM
    response = client.responses.create(

        model="gpt-6-luna",

        instructions="""
        You are a helpful AI assistant.
        Give simple, clear and friendly answers.
        """,

        input=conversation_history
    )


    # Extract AI answer
    ai_reply = response.output_text


    # Add AI answer to memory
    conversation_history.append({
        "role": "assistant",
        "content": ai_reply
    })


    # Return response to frontend
    return {
        "user_message": request.message,
        "bot_response": ai_reply
    }

@app.post("/clear")
def clear_chat():

    conversation_history.clear()

    print("\nConversation memory cleared!")

    return {
        "message": "Conversation cleared successfully"
    }