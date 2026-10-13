from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from typing import Literal

import chromadb
import json
import os
import sqlite3


# --------------------------------
# 1. Environment + Clients
# --------------------------------

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

app = FastAPI()


# --------------------------------
# 2. Configuration
# --------------------------------

DATABASE_NAME = "chatbot.db"

EMBEDDING_MODEL = "text-embedding-3-small"

CHROMA_PATH = "./chroma_db"

CHROMA_COLLECTION_NAME = "ai_journey_documents"

MAX_AGENT_STEPS = 5


# --------------------------------
# 3. Chroma Vector Database
# --------------------------------

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

vector_collection = (
    chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=None,
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        }
    )
)


# --------------------------------
# 4. System Prompts
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


RAG_SYSTEM_PROMPT = """
You are a helpful AI assistant.

Answer the user's question using only the provided context.

Use the retrieved context as the source of information for your answer.

Do not invent information that is not supported by the provided context.

If the retrieved context does not contain enough information to answer
the question, say that the available context does not contain enough
information.

Keep the answer clear and beginner-friendly.
"""


AGENT_SYSTEM_PROMPT = """
You are a tool-using AI agent.

Your job is to solve the user's request by deciding which available
tools are needed.

AVAILABLE TOOLS:

1. calculator
Use this tool whenever arithmetic is required.

2. search_knowledge
Use this tool when information may exist in the local knowledge base.

RULES:

You may call one tool, multiple tools, or no tools.

You may use tools across multiple steps.

When a tool returns information, examine the result and decide whether
another tool is needed.

For arithmetic, always use the calculator tool.

For questions about the local knowledge base, use search_knowledge
instead of relying only on your own memory.

When you have enough information, answer the user clearly and concisely.

Do not invent tool results.
"""


# --------------------------------
# 5. Initialize SQLite Database
# --------------------------------

def initialize_database():

    connection = sqlite3.connect(
        DATABASE_NAME
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            conversation_id TEXT NOT NULL,

            role TEXT NOT NULL,

            content TEXT NOT NULL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.commit()

    connection.close()


initialize_database()


# --------------------------------
# 6. Request / Response Models
# --------------------------------

class ChatRequest(BaseModel):

    conversation_id: str

    message: str


class EmbeddingSearchRequest(BaseModel):

    query: str


class RAGRequest(BaseModel):

    query: str


class AgentRequest(BaseModel):

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
# 7. Save Message
# --------------------------------

def save_message(

    conversation_id: str,

    role: str,

    content: str

):

    connection = sqlite3.connect(
        DATABASE_NAME
    )

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
# 8. Get Conversation History
# --------------------------------

def get_conversation_history(

    conversation_id: str

):

    connection = sqlite3.connect(
        DATABASE_NAME
    )

    cursor = connection.cursor()

    cursor.execute(

        """
        SELECT role, content

        FROM messages

        WHERE conversation_id = ?

        ORDER BY id
        """,

        (
            conversation_id,
        )
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
# 9. Calculator Python Function
# --------------------------------

def calculate(

    operation: str,

    a: float,

    b: float

):

    if operation == "add":

        return a + b


    if operation == "subtract":

        return a - b


    if operation == "multiply":

        return a * b


    if operation == "divide":

        if b == 0:

            return "Cannot divide by zero."

        return a / b


    return "Unsupported operation."


# --------------------------------
# 10. Calculator Tool Definition
# --------------------------------

CALCULATOR_TOOL = {

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


TOOLS = [

    CALCULATOR_TOOL

]


# --------------------------------
# 11. Sample Documents
# --------------------------------

SAMPLE_DOCUMENTS = [

    (
        "Python inheritance allows a child class "

        "to reuse properties and methods from "

        "a parent class."
    ),

    (
        "FastAPI is a Python web framework "

        "used to build APIs and backend services."
    ),

    (
        "SQLite is a lightweight relational "

        "database that stores data in a local file."
    ),

    (
        "Embeddings convert text into numerical "

        "vectors that represent semantic meaning."
    ),

    (
        "Retrieval Augmented Generation retrieves "

        "relevant information before asking an LLM "

        "to generate an answer."
    )

]


SAMPLE_DOCUMENT_IDS = [

    "doc_1",

    "doc_2",

    "doc_3",

    "doc_4",

    "doc_5"

]


SAMPLE_DOCUMENT_METADATA = [

    {

        "topic": "python",

        "category": "programming"

    },

    {

        "topic": "fastapi",

        "category": "backend"

    },

    {

        "topic": "sqlite",

        "category": "database"

    },

    {

        "topic": "embeddings",

        "category": "ai"

    },

    {

        "topic": "rag",

        "category": "ai"

    }

]


# --------------------------------
# 12. Create Embedding
# --------------------------------

def create_embedding(

    text: str

):

    embedding_response = (
        client.embeddings.create(

            model=EMBEDDING_MODEL,

            input=text

        )
    )

    return (
        embedding_response
        .data[0]
        .embedding
    )


# --------------------------------
# 13. Initialize Vector Database
# --------------------------------

def initialize_vector_database():

    existing_documents = (
        vector_collection.count()
    )

    if existing_documents > 0:

        print(

            f"\nVector database already initialized "

            f"with {existing_documents} documents."

        )

        return


    print(
        "\nCreating document embeddings..."
    )


    embedding_response = (
        client.embeddings.create(

            model=EMBEDDING_MODEL,

            input=SAMPLE_DOCUMENTS

        )
    )


    document_embeddings = []


    for item in embedding_response.data:

        document_embeddings.append(

            item.embedding

        )


    vector_collection.add(

        ids=SAMPLE_DOCUMENT_IDS,

        documents=SAMPLE_DOCUMENTS,

        embeddings=document_embeddings,

        metadatas=SAMPLE_DOCUMENT_METADATA

    )


    print(
        "Documents stored in Chroma."
    )


initialize_vector_database()


# --------------------------------
# 14. Retrieve Documents From Chroma
# --------------------------------

def retrieve_documents(

    query_text: str,

    n_results: int = 3

):

    total_documents = (
        vector_collection.count()
    )


    if total_documents == 0:

        return []


    actual_n_results = min(

        n_results,

        total_documents

    )


    query_embedding = (
        create_embedding(

            query_text

        )
    )


    search_results = (
        vector_collection.query(

            query_embeddings=[

                query_embedding

            ],

            n_results=actual_n_results,

            include=[

                "documents",

                "metadatas",

                "distances"

            ]

        )
    )


    document_ids = (
        search_results[
            "ids"
        ][0]
    )


    documents = (
        search_results[
            "documents"
        ][0]
    )


    metadatas = (
        search_results[
            "metadatas"
        ][0]
    )


    distances = (
        search_results[
            "distances"
        ][0]
    )


    retrieved_documents = []


    for index in range(

        len(documents)

    ):

        distance = (
            distances[index]
        )


        similarity = (
            1 - distance
        )


        retrieved_document = {

            "id":
                document_ids[index],

            "document":
                documents[index],

            "metadata":
                metadatas[index],

            "similarity":
                round(

                    similarity,

                    4

                )

        }


        retrieved_documents.append(

            retrieved_document

        )


    return retrieved_documents


# --------------------------------
# 15. Knowledge Search Python Tool
# --------------------------------

def search_knowledge(

    query: str

):

    matches = (
        retrieve_documents(

            query,

            n_results=3

        )
    )


    return {

        "query":
            query,

        "match_count":
            len(matches),

        "matches":
            matches

    }


# --------------------------------
# 16. Knowledge Search Tool Definition
# --------------------------------

KNOWLEDGE_SEARCH_TOOL = {

    "type": "function",

    "name": "search_knowledge",

    "description": (

        "Search the local knowledge base for "

        "documents relevant to the user's question."

    ),

    "parameters": {

        "type": "object",

        "properties": {

            "query": {

                "type": "string",

                "description": (

                    "The semantic search query."

                )

            }

        },

        "required": [

            "query"

        ],

        "additionalProperties": False

    },

    "strict": True

}


AGENT_TOOLS = [

    CALCULATOR_TOOL,

    KNOWLEDGE_SEARCH_TOOL

]


# --------------------------------
# 17. Agent Tool Router
# --------------------------------

def execute_agent_tool(

    tool_call

):

    try:

        arguments = json.loads(

            tool_call.arguments

        )


        if tool_call.name == "calculator":

            result = calculate(

                arguments["operation"],

                arguments["a"],

                arguments["b"]

            )


            return {

                "result":
                    result

            }


        if tool_call.name == "search_knowledge":

            return search_knowledge(

                arguments["query"]

            )


        return {

            "error":
                f"Unknown tool: {tool_call.name}"

        }


    except (

        json.JSONDecodeError,

        KeyError,

        TypeError,

        ValueError

    ) as error:

        return {

            "error":
                f"Tool execution failed: {error}"

        }


# --------------------------------
# 18. Serve Frontend
# --------------------------------

@app.get("/")
def home():

    return FileResponse(
        "index.html"
    )


# --------------------------------
# 19. Main Chat Endpoint
#
# Normal Chat + Streaming
# + Automatic Calculator Tool Calling
# --------------------------------

@app.post("/chat")
def chat(

    request: ChatRequest

):

    conversation_id = (
        request.conversation_id
    )


    conversation_history = (
        get_conversation_history(

            conversation_id

        )
    )


    conversation_history.append({

        "role":
            "user",

        "content":
            request.message

    })


    save_message(

        conversation_id,

        "user",

        request.message

    )


    def generate_response():

        final_answer = ""

        tool_calls = {}

        model_output_items = []


        try:

            stream = (
                client.responses.create(

                    model="gpt-6-luna",

                    instructions=SYSTEM_PROMPT,

                    input=conversation_history,

                    tools=TOOLS,

                    tool_choice="auto",

                    stream=True

                )
            )


            for event in stream:


                if (

                    event.type

                    == "response.output_text.delta"

                ):

                    chunk = event.delta

                    final_answer += chunk

                    yield chunk


                elif (

                    event.type

                    == "response.output_item.added"

                ):


                    if (

                        event.item.type

                        == "function_call"

                    ):

                        tool_calls[

                            event.output_index

                        ] = event.item


                elif (

                    event.type

                    == "response.output_item.done"

                ):


                    model_output_items.append(

                        event.item

                    )


                    if (

                        event.item.type

                        == "function_call"

                    ):

                        tool_calls[

                            event.output_index

                        ] = event.item


            # --------------------------------
            # No tool required
            # --------------------------------

            if not tool_calls:


                if final_answer.strip():

                    save_message(

                        conversation_id,

                        "assistant",

                        final_answer

                    )


                return


            # --------------------------------
            # Calculator requested
            # --------------------------------

            tool_context = list(

                conversation_history

            )


            tool_context.extend(

                model_output_items

            )


            for tool_call in (

                tool_calls.values()

            ):


                if (

                    tool_call.name

                    == "calculator"

                ):


                    arguments = json.loads(

                        tool_call.arguments

                    )


                    result = calculate(

                        arguments["operation"],

                        arguments["a"],

                        arguments["b"]

                    )


                    tool_context.append({

                        "type":
                            "function_call_output",

                        "call_id":
                            tool_call.call_id,

                        "output":
                            json.dumps({

                                "result":
                                    result

                            })

                    })


            # --------------------------------
            # Second LLM call
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


            final_answer = ""


            for event in final_stream:


                if (

                    event.type

                    == "response.output_text.delta"

                ):

                    chunk = event.delta

                    final_answer += chunk

                    yield chunk


            if final_answer.strip():

                save_message(

                    conversation_id,

                    "assistant",

                    final_answer

                )


        except Exception as error:

            print(

                "\nChat Error:",

                error

            )


            yield (

                "\nSorry, something went wrong "

                "while generating the response."

            )


    return StreamingResponse(

        generate_response(),

        media_type="text/plain"

    )


# --------------------------------
# 20. Structured Chat Endpoint
# --------------------------------

@app.post("/chat/structured")
def structured_chat(

    request: ChatRequest

):

    conversation_id = (
        request.conversation_id
    )


    conversation_history = (
        get_conversation_history(

            conversation_id

        )
    )


    conversation_history.append({

        "role":
            "user",

        "content":
            request.message

    })


    structured_response = (
        client.responses.parse(

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

            text_format=
                StructuredChatResponse

        )
    )


    structured_answer = (
        structured_response
        .output_parsed
    )


    if structured_answer is None:

        raise HTTPException(

            status_code=500,

            detail=(

                "The AI did not return "

                "structured output."

            )

        )


    save_message(

        conversation_id,

        "user",

        request.message

    )


    assistant_text = (

        structured_answer.answer

        + "\n\nKey points:\n"

        + "\n".join(

            structured_answer
            .key_points

        )

    )


    save_message(

        conversation_id,

        "assistant",

        assistant_text

    )


    return (
        structured_answer
        .model_dump()
    )


# --------------------------------
# 21. Vector Database Semantic Search
# --------------------------------

@app.post("/embeddings/search")
def embedding_search(

    request:
        EmbeddingSearchRequest

):

    query_text = (
        request.query
    )


    matches = (
        retrieve_documents(

            query_text,

            n_results=3

        )
    )


    if not matches:

        raise HTTPException(

            status_code=500,

            detail=(
                "The vector database is empty."
            )

        )


    return {

        "query":
            query_text,

        "total_documents":
            vector_collection.count(),

        "best_match":
            matches[0],

        "top_matches":
            matches

    }


# --------------------------------
# 22. RAG Endpoint
# --------------------------------

@app.post("/rag")
def rag_chat(

    request: RAGRequest

):

    query_text = (
        request.query
    )


    retrieved_documents = (
        retrieve_documents(

            query_text,

            n_results=3

        )
    )


    if not retrieved_documents:

        raise HTTPException(

            status_code=500,

            detail=(

                "No documents were available "

                "for retrieval."

            )

        )


    context_parts = []


    for index, item in enumerate(

        retrieved_documents,

        start=1

    ):

        context_part = (

            f"Document {index}:\n"

            f"{item['document']}"

        )


        context_parts.append(

            context_part

        )


    context = (
        "\n\n".join(

            context_parts

        )
    )


    rag_input = (

        "CONTEXT:\n"

        f"{context}\n\n"

        "USER QUESTION:\n"

        f"{query_text}"

    )


    llm_response = (
        client.responses.create(

            model="gpt-6-luna",

            instructions=
                RAG_SYSTEM_PROMPT,

            input=
                rag_input

        )
    )


    answer = (
        llm_response.output_text
    )


    return {

        "query":
            query_text,

        "answer":
            answer,

        "retrieved_documents":
            retrieved_documents

    }


# --------------------------------
# 23. Agent Endpoint
# --------------------------------

@app.post("/agent")
def agent_chat(

    request: AgentRequest

):

    conversation_id = (
        request.conversation_id
    )


    # --------------------------------
    # Load conversation history
    # --------------------------------

    conversation_history = (
        get_conversation_history(

            conversation_id

        )
    )


    # --------------------------------
    # Prepare agent context
    # --------------------------------

    agent_input = list(

        conversation_history

    )


    agent_input.append({

        "role":
            "user",

        "content":
            request.message

    })


    save_message(

        conversation_id,

        "user",

        request.message

    )


    tool_trace = []


    # --------------------------------
    # Agent loop
    # --------------------------------

    for step in range(

        1,

        MAX_AGENT_STEPS + 1

    ):


        print(

            f"\nAgent Step: {step}"

        )


        agent_response = (
            client.responses.create(

                model="gpt-6-luna",

                instructions=
                    AGENT_SYSTEM_PROMPT,

                input=
                    agent_input,

                tools=
                    AGENT_TOOLS,

                tool_choice="auto"

            )
        )


        # --------------------------------
        # Preserve model output
        # --------------------------------

        agent_input.extend(

            agent_response.output

        )


        # --------------------------------
        # Find function calls
        # --------------------------------

        tool_calls = [

            item

            for item in agent_response.output

            if item.type == "function_call"

        ]


        # --------------------------------
        # No tool call = final answer
        # --------------------------------

        if not tool_calls:


            final_answer = (
                agent_response.output_text
            )


            if final_answer.strip():

                save_message(

                    conversation_id,

                    "assistant",

                    final_answer

                )


            return {

                "answer":
                    final_answer,

                "steps_used":
                    step,

                "tool_trace":
                    tool_trace

            }


        # --------------------------------
        # Execute requested tools
        # --------------------------------

        for tool_call in tool_calls:


            print(

                "Agent requested tool:",

                tool_call.name

            )


            tool_result = (
                execute_agent_tool(

                    tool_call

                )
            )


            print(

                "Tool result:",

                tool_result

            )


            tool_trace.append({

                "step":
                    step,

                "tool":
                    tool_call.name,

                "result":
                    tool_result

            })


            # --------------------------------
            # Send tool result back to LLM
            # --------------------------------

            agent_input.append({

                "type":
                    "function_call_output",

                "call_id":
                    tool_call.call_id,

                "output":
                    json.dumps(

                        tool_result

                    )

            })


    # --------------------------------
    # Too many steps
    # --------------------------------

    raise HTTPException(

        status_code=500,

        detail=(

            "The agent reached the maximum "

            "number of execution steps."

        )

    )


# --------------------------------
# 24. Clear Conversation
# --------------------------------

@app.post("/clear")
def clear_chat(

    request: ClearRequest

):

    connection = (
        sqlite3.connect(

            DATABASE_NAME

        )
    )


    cursor = (
        connection.cursor()
    )


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