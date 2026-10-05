from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from typing import Literal
import sqlite3
import os
import json


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
# 7. Request / Response Models
# --------------------------------

class ChatRequest(BaseModel):
    conversation_id: str
    message: str


class ClearRequest(BaseModel):
    conversation_id: str


class StructuredChatResponse(BaseModel):
    topic: str

    difficulty: Literal[
        "beginner",
        "intermediate",
        "advanced"
    ]

    answer: str

    key_points: list[str]


# --------------------------------
# 8. Save message to database
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


### calculator tool

def calculate(
    operation: str,
    a: float,
    b: float
):
    if operation == "add":
        return a+b
    elif operation =="subtract":
        return a-b
    elif operation == "multiply":
        return a*b
    elif operation == "divide":
        if b==0:
            return "cannot divide by zero."
        return a/b
    else:
        return "Unsoppurted Operation"

# Tool Definition:
# --------------------------------
# Tool Definitions
# --------------------------------

TOOLS = [
    {
        "type": "function",

        "name": "calculator",

        "description": (
            "Perform basic arithmetic calculations. "
            "Use this tool for addition, subtraction, "
            "multiplication, and division."
        ),

        "parameters": {
            "type": "object",

            "properties": {

                "operation": {
                    "type": "string",

                    "enum": [
                        "add",
                        "subtract",
                        "multiply",
                        "divide"
                    ]
                },

                "a": {
                    "type": "number"
                },

                "b": {
                    "type": "number"
                }
            },

            "required": [
                "operation",
                "a",
                "b"
            ],

            "additionalProperties": False
        },

        "strict": True
    }
]

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


    # Get previous conversation history
    conversation_history = get_conversation_history(
        conversation_id
    )


    # Add current user message to history
    conversation_history.append({
        "role": "user",
        "content": request.message
    })


    # Save user message to database
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
    # Streaming generator
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

                if event.type == "response.output_text.delta":

                    chunk = event.delta

                    # Build full answer
                    full_response += chunk

                    # Immediately send chunk to browser
                    yield chunk


            # Save complete assistant response
            # after streaming finishes
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

            yield (
                "\nSorry, something went wrong "
                "while generating the response."
            )


    return StreamingResponse(
        generate_response(),
        media_type="text/plain"
    )


# --------------------------------
# 12. Structured Chat Endpoint
# --------------------------------

@app.post("/chat/structured")
def structured_chat(request: ChatRequest):

    conversation_id = request.conversation_id


    # Get previous conversation history
    conversation_history = get_conversation_history(
        conversation_id
    )


    # Add current user message
    conversation_history.append({
        "role": "user",
        "content": request.message
    })


    # Ask model for structured output
    response = client.responses.parse(

        model="gpt-6-luna",

        instructions="""
        You are a helpful AI tutor.

        Analyze the user's question and answer it clearly.

        Determine whether the topic is appropriate for a
        beginner, intermediate, or advanced learner.

        Provide the main answer and a few important key points.

        Keep the answer beginner-friendly whenever possible.
        """,

        input=conversation_history,

        text_format=StructuredChatResponse
    )


    structured_answer = response.output_parsed


    # Make sure structured output exists
    if structured_answer is None:

        raise HTTPException(
            status_code=500,
            detail="The AI did not return structured output."
        )


    # Save user message
    save_message(
        conversation_id,
        "user",
        request.message
    )


    # Convert structured result into readable
    # text for conversation history
    assistant_text = (
        structured_answer.answer
        + "\n\nKey points:\n"
        + "\n".join(
            structured_answer.key_points
        )
    )


    # Save assistant response
    save_message(
        conversation_id,
        "assistant",
        assistant_text
    )


    # Return structured JSON response
    return structured_answer.model_dump()

# --------------------------------
# Tool Calling Chat Endpoint
# --------------------------------

@app.post("/chat/tools")
def tool_chat(request: ChatRequest):

    conversation_id = request.conversation_id


    # --------------------------------
    # Get previous conversation
    # --------------------------------

    conversation_history = get_conversation_history(
        conversation_id
    )


    # --------------------------------
    # Add current user message
    # --------------------------------

    conversation_history.append({
        "role": "user",
        "content": request.message
    })


    # --------------------------------
    # First LLM request
    # --------------------------------

    response = client.responses.create(

        model="gpt-6-luna",

        instructions="""
        You are a helpful AI assistant.

        You have access to a calculator tool.

        Whenever arithmetic is required,
        use the calculator tool instead of
        calculating the result yourself.

        If no tool is needed, answer normally.

        Use simple plain text.
        """,

        input=conversation_history,

        tools=TOOLS,

        tool_choice="auto"
    )


    # --------------------------------
    # Preserve model output
    # --------------------------------

    conversation_history.extend(
        item.model_dump(exclude_none=True)
        for item in response.output
    )


    used_tools = []


    # --------------------------------
    # Look for function calls
    # --------------------------------

    for item in response.output:


        if item.type != "function_call":
            continue


        # --------------------------------
        # Calculator requested
        # --------------------------------

        if item.name == "calculator":

            arguments = json.loads(
                item.arguments
            )


            operation = arguments[
                "operation"
            ]

            a = arguments["a"]

            b = arguments["b"]


            print("\nCalculator tool requested:")

            print(
                "Operation:",
                operation
            )

            print(
                "A:",
                a
            )

            print(
                "B:",
                b
            )


            # --------------------------------
            # Execute real Python function
            # --------------------------------

            result = calculate(
                operation,
                a,
                b
            )


            print(
                "\nCalculator result:",
                result
            )


            used_tools.append(
                "calculator"
            )


            # --------------------------------
            # Send tool result back to LLM
            # --------------------------------

            conversation_history.append({
                "type":
                    "function_call_output",

                "call_id":
                    item.call_id,

                "output":
                    json.dumps({
                        "result": result
                    })
            })


    # --------------------------------
    # Tool was used
    # --------------------------------

    if used_tools:

        final_response = client.responses.create(

            model="gpt-6-luna",

            instructions="""
            Give the user a clear final answer
            using the result returned by the tool.

            Use simple plain text.
            """,

            input=conversation_history,

            tools=TOOLS,

            tool_choice="none"
        )


        final_answer = (
            final_response.output_text
        )


    # --------------------------------
    # No tool needed
    # --------------------------------

    else:

        final_answer = (
            response.output_text
        )


    # --------------------------------
    # Save conversation to SQLite
    # --------------------------------

    save_message(
        conversation_id,
        "user",
        request.message
    )


    save_message(
        conversation_id,
        "assistant",
        final_answer
    )


    # --------------------------------
    # Return response
    # --------------------------------

    return {

        "conversation_id":
            conversation_id,

        "used_tools":
            used_tools,

        "bot_response":
            final_answer
    }


# --------------------------------
# 13. Clear Conversation Endpoint
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
        "message": "Conversation cleared successfully"
    }