from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from typing import Literal

import sqlite3
import json
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

TOOLS:
You have access to a calculator tool.

Whenever arithmetic calculation is required,
use the calculator tool instead of calculating
the answer yourself.

If a tool is needed, call the tool without first
writing a user-facing answer.

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
# 8. Save Message
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
# 9. Get Conversation History
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
# 10. Calculator Python Function
# --------------------------------

def calculate(
    operation: str,
    a: float,
    b: float
):

    if operation == "add":

        return a + b


    elif operation == "subtract":

        return a - b


    elif operation == "multiply":

        return a * b


    elif operation == "divide":

        if b == 0:

            return "Cannot divide by zero."

        return a / b


    else:

        return "Unsupported operation."


# --------------------------------
# 11. Tool Definition
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
# 12. Serve Frontend
# --------------------------------

@app.get("/")
def home():

    return FileResponse("index.html")


# --------------------------------
# 13. Main Chat Endpoint
#
# Normal Chat + Streaming
# + Automatic Tool Calling
# --------------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    conversation_id = request.conversation_id


    # --------------------------------
    # Load conversation history
    # --------------------------------

    conversation_history = (
        get_conversation_history(
            conversation_id
        )
    )


    # --------------------------------
    # Add current user message
    # --------------------------------

    conversation_history.append({
        "role": "user",
        "content": request.message
    })


    # --------------------------------
    # Save user message
    # --------------------------------

    save_message(
        conversation_id,
        "user",
        request.message
    )


    print("\n--------------------------------")

    print("Conversation ID:")
    print(conversation_id)

    print("\nUser Message:")
    print(request.message)

    print("--------------------------------")


    # --------------------------------
    # Streaming Generator
    # --------------------------------

    def generate_response():

        final_answer = ""

        tool_calls = {}

        model_output_items = []


        try:

            # --------------------------------
            # FIRST LLM CALL
            #
            # Model decides:
            #
            # 1. Answer normally
            #
            # OR
            #
            # 2. Request a tool
            # --------------------------------

            stream = client.responses.create(

                model="gpt-6-luna",

                instructions=SYSTEM_PROMPT,

                input=conversation_history,

                tools=TOOLS,

                tool_choice="auto",

                stream=True
            )


            # --------------------------------
            # Process first stream
            # --------------------------------

            for event in stream:


                # --------------------------------
                # Normal text response
                # --------------------------------

                if event.type == "response.output_text.delta":

                    chunk = event.delta

                    final_answer += chunk

                    yield chunk


                # --------------------------------
                # Tool call has started
                # --------------------------------

                elif event.type == "response.output_item.added":

                    if event.item.type == "function_call":

                        tool_calls[
                            event.output_index
                        ] = event.item


                # --------------------------------
                # Tool call finished
                # --------------------------------

                elif event.type == "response.output_item.done":

                    model_output_items.append(
                        event.item
                    )


                    if event.item.type == "function_call":

                        tool_calls[
                            event.output_index
                        ] = event.item


            # --------------------------------
            # NO TOOL WAS USED
            # --------------------------------

            if not tool_calls:

                if final_answer.strip():

                    save_message(
                        conversation_id,
                        "assistant",
                        final_answer
                    )


                print("\nNo tool required.")

                print("\nFinal Response:")
                print(final_answer)

                return


            # --------------------------------
            # TOOL WAS REQUESTED
            # --------------------------------

            print("\nTool requested by LLM.")


            # The model output items must be
            # included when sending tool results
            # back to the model.

            tool_context = list(
                conversation_history
            )


            tool_context.extend(
                model_output_items
            )


            # --------------------------------
            # Execute requested tools
            # --------------------------------

            for tool_call in tool_calls.values():


                if tool_call.name == "calculator":


                    arguments = json.loads(
                        tool_call.arguments
                    )


                    operation = arguments[
                        "operation"
                    ]

                    a = arguments["a"]

                    b = arguments["b"]


                    print(
                        "\nCalculator requested:"
                    )

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
                    # REAL Python execution
                    # --------------------------------

                    result = calculate(
                        operation,
                        a,
                        b
                    )


                    print(
                        "Calculator result:",
                        result
                    )


                    # --------------------------------
                    # Send tool result back
                    # --------------------------------

                    tool_context.append({

                        "type":
                            "function_call_output",

                        "call_id":
                            tool_call.call_id,

                        "output":
                            json.dumps({
                                "result": result
                            })

                    })


            # --------------------------------
            # SECOND LLM CALL
            #
            # LLM now sees the tool result
            # and produces final answer.
            # --------------------------------

            final_stream = (
                client.responses.create(

                    model="gpt-6-luna",

                    instructions=SYSTEM_PROMPT,

                    input=tool_context,

                    tools=TOOLS,

                    tool_choice="none",

                    stream=True
                )
            )


            # Reset because any text from
            # the first response should not
            # become the stored final answer.

            final_answer = ""


            # --------------------------------
            # Stream final answer
            # --------------------------------

            for event in final_stream:


                if (
                    event.type
                    == "response.output_text.delta"
                ):

                    chunk = event.delta

                    final_answer += chunk

                    yield chunk


            # --------------------------------
            # Save complete AI response
            # --------------------------------

            if final_answer.strip():

                save_message(
                    conversation_id,
                    "assistant",
                    final_answer
                )


            print("\nFinal Response:")
            print(final_answer)


        except Exception as error:

            print("\nChat Error:")
            print(error)

            yield (
                "\nSorry, something went wrong "
                "while generating the response."
            )


    # --------------------------------
    # Send streaming response
    # --------------------------------

    return StreamingResponse(
        generate_response(),
        media_type="text/plain"
    )


# --------------------------------
# 14. Structured Chat Endpoint
# --------------------------------

@app.post("/chat/structured")
def structured_chat(
    request: ChatRequest
):

    conversation_id = (
        request.conversation_id
    )


    # --------------------------------
    # Get previous history
    # --------------------------------

    conversation_history = (
        get_conversation_history(
            conversation_id
        )
    )


    # --------------------------------
    # Add current message
    # --------------------------------

    conversation_history.append({

        "role": "user",

        "content": request.message

    })


    # --------------------------------
    # Structured AI call
    # --------------------------------

    response = client.responses.parse(

        model="gpt-6-luna",

        instructions="""
        You are a helpful AI tutor.

        Analyze the user's question
        and answer it clearly.

        Determine whether the topic
        is appropriate for a beginner,
        intermediate, or advanced learner.

        Provide the main answer and
        a few important key points.

        Keep the answer beginner-friendly
        whenever possible.
        """,

        input=conversation_history,

        text_format=StructuredChatResponse
    )


    structured_answer = (
        response.output_parsed
    )


    # --------------------------------
    # Defensive check
    # --------------------------------

    if structured_answer is None:

        raise HTTPException(

            status_code=500,

            detail=(
                "The AI did not return "
                "structured output."
            )
        )


    # --------------------------------
    # Save user message
    # --------------------------------

    save_message(
        conversation_id,
        "user",
        request.message
    )


    # --------------------------------
    # Create readable history text
    # --------------------------------

    assistant_text = (

        structured_answer.answer

        + "\n\nKey points:\n"

        + "\n".join(
            structured_answer.key_points
        )
    )


    # --------------------------------
    # Save assistant response
    # --------------------------------

    save_message(
        conversation_id,
        "assistant",
        assistant_text
    )


    # --------------------------------
    # Return structured JSON
    # --------------------------------

    return (
        structured_answer.model_dump()
    )


# --------------------------------
# 15. Clear Conversation
# --------------------------------

@app.post("/clear")
def clear_chat(
    request: ClearRequest
):

    connection = sqlite3.connect(
        DATABASE_NAME
    )

    cursor = connection.cursor()


    cursor.execute(

        """
        DELETE FROM messages
        WHERE conversation_id = ?
        """,

        (
            request.conversation_id,
        )
    )


    connection.commit()

    connection.close()


    print(
        f"\nConversation "
        f"{request.conversation_id} "
        f"cleared!"
    )


    return {

        "message":
            "Conversation cleared successfully"

    }