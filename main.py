from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from typing import Literal

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from langgraph.graph import (
    StateGraph,
    START,
    END,
    MessagesState
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition
)

import chromadb
import json
import os
import sqlite3


# --------------------------------
# 1. Setup
# --------------------------------

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

app = FastAPI()

DATABASE_NAME = "chatbot.db"

CHAT_MODEL = "gpt-6-luna"

EMBEDDING_MODEL = "text-embedding-3-small"

CHROMA_PATH = "./chroma_db"

CHROMA_COLLECTION_NAME = "ai_journey_documents"

MAX_AGENT_STEPS = 5


# --------------------------------
# 2. ChromaDB
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
# 3. Prompts
# --------------------------------

SYSTEM_PROMPT = """
ROLE:
You are a helpful AI assistant.

GOAL:
Help the user understand topics clearly and accurately.

STYLE:
Use simple, clean plain text.
Do not use Markdown formatting.
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


LANGCHAIN_AGENT_PROMPT = """
You are a helpful AI agent.

Use the available tools whenever they are needed.

Use the calculator tool for arithmetic.

Use the search_knowledge tool when the user asks about information
that may exist in the local knowledge base.

You may use multiple tools across multiple steps when necessary.

After receiving a tool result, decide whether another tool is needed.

When you have enough information, provide a clear final answer.

Do not invent tool results.
"""


LANGGRAPH_AGENT_PROMPT = """
You are a helpful AI agent running inside a LangGraph workflow.

Use the available tools whenever they are needed.

Use the calculator tool for arithmetic.

Use the search_knowledge tool when the user asks about information
that may exist in the local knowledge base.

You may use multiple tools across multiple graph cycles when necessary.

After receiving a tool result, decide whether another tool is needed.

When you have enough information, provide a clear final answer.

Do not invent tool results.
"""


# --------------------------------
# 4. SQLite Database
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
# 5. Save Message
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
# 6. Get Conversation History
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

    for role, content in rows:

        conversation_history.append({
            "role": role,
            "content": content
        })

    return conversation_history


# --------------------------------
# 7. Request / Response Models
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
# 8. Calculator Function
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
# 9. Manual Calculator Tool
# --------------------------------

CALCULATOR_TOOL = {

    "type":
        "function",

    "name":
        "calculator",

    "description":
        "Perform basic arithmetic calculations.",

    "parameters": {

        "type":
            "object",

        "properties": {

            "operation": {

                "type":
                    "string",

                "enum": [
                    "add",
                    "subtract",
                    "multiply",
                    "divide"
                ]

            },

            "a": {

                "type":
                    "number"

            },

            "b": {

                "type":
                    "number"

            }

        },

        "required": [
            "operation",
            "a",
            "b"
        ],

        "additionalProperties":
            False

    },

    "strict":
        True

}


TOOLS = [
    CALCULATOR_TOOL
]


# --------------------------------
# 10. Sample Knowledge
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
        "topic":
            "python",

        "category":
            "programming"
    },

    {
        "topic":
            "fastapi",

        "category":
            "backend"
    },

    {
        "topic":
            "sqlite",

        "category":
            "database"
    },

    {
        "topic":
            "embeddings",

        "category":
            "ai"
    },

    {
        "topic":
            "rag",

        "category":
            "ai"
    }

]


# --------------------------------
# 11. Create Embedding
# --------------------------------

def create_embedding(
    text: str
):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return (
        response
        .data[0]
        .embedding
    )


# --------------------------------
# 12. Initialize Vector Database
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


    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=SAMPLE_DOCUMENTS
    )


    document_embeddings = []


    for item in response.data:

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
# 13. Retrieve Documents
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

            n_results=
                actual_n_results,

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

        similarity = (
            1
            -
            distances[index]
        )


        retrieved_documents.append({

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

        })


    return retrieved_documents


# --------------------------------
# 14. Knowledge Search Function
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
# 15. Manual Knowledge Search Tool
# --------------------------------

KNOWLEDGE_SEARCH_TOOL = {

    "type":
        "function",

    "name":
        "search_knowledge",

    "description":
        "Search the local knowledge base for relevant documents.",

    "parameters": {

        "type":
            "object",

        "properties": {

            "query": {

                "type":
                    "string",

                "description":
                    "The semantic search query."

            }

        },

        "required": [
            "query"
        ],

        "additionalProperties":
            False

    },

    "strict":
        True

}


AGENT_TOOLS = [

    CALCULATOR_TOOL,

    KNOWLEDGE_SEARCH_TOOL

]


# --------------------------------
# 16. Manual Agent Tool Router
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
# 17. LangChain Model
# --------------------------------

langchain_model = ChatOpenAI(

    model=
        CHAT_MODEL,

    use_responses_api=
        True

)


# --------------------------------
# 18. LangChain Calculator Tool
# --------------------------------

@tool("calculator")
def langchain_calculator(
    operation: str,
    a: float,
    b: float
) -> dict:

    """
    Perform basic arithmetic calculations.

    Supported operations are:
    add, subtract, multiply, and divide.
    """

    result = calculate(
        operation,
        a,
        b
    )


    return {

        "result":
            result

    }


# --------------------------------
# 19. LangChain Knowledge Tool
# --------------------------------

@tool("search_knowledge")
def langchain_search_knowledge(
    query: str
) -> dict:

    """
    Search the local ChromaDB knowledge base
    for information relevant to the user's question.
    """

    return search_knowledge(
        query
    )


# --------------------------------
# 20. Create LangChain Agent
# --------------------------------

langchain_agent = create_agent(

    model=
        langchain_model,

    tools=[
        langchain_calculator,
        langchain_search_knowledge
    ],

    system_prompt=
        LANGCHAIN_AGENT_PROMPT

)


# --------------------------------
# 21. Extract AI Message Text
# --------------------------------

def extract_ai_text(
    message
):

    text_value = getattr(
        message,
        "text",
        ""
    )


    if (
        isinstance(
            text_value,
            str
        )
        and
        text_value.strip()
    ):

        return text_value


    content = getattr(
        message,
        "content",
        ""
    )


    if isinstance(
        content,
        str
    ):

        return content


    if isinstance(
        content,
        list
    ):

        text_parts = []


        for block in content:


            if not isinstance(
                block,
                dict
            ):

                continue


            block_text = (

                block.get(
                    "text"
                )

                or

                block.get(
                    "content"
                )

            )


            if isinstance(
                block_text,
                str
            ):

                text_parts.append(
                    block_text
                )


        return "\n".join(
            text_parts
        )


    return str(
        content
    )


# ============================================
# MILESTONE 17 — LANGGRAPH
# ============================================


# --------------------------------
# 22. LangGraph Tools
# --------------------------------

LANGGRAPH_TOOLS = [

    langchain_calculator,

    langchain_search_knowledge

]


# --------------------------------
# 23. Bind Tools To LangGraph Model
# --------------------------------

langgraph_model = (
    langchain_model.bind_tools(
        LANGGRAPH_TOOLS
    )
)


# --------------------------------
# 24. LangGraph Agent Node
# --------------------------------

def langgraph_agent_node(
    state: MessagesState
):

    model_messages = [

        SystemMessage(
            content=
                LANGGRAPH_AGENT_PROMPT
        ),

        *state[
            "messages"
        ]

    ]


    response = (
        langgraph_model.invoke(
            model_messages
        )
    )


    return {

        "messages": [
            response
        ]

    }


# --------------------------------
# 25. LangGraph Tool Node
# --------------------------------

langgraph_tool_node = ToolNode(

    LANGGRAPH_TOOLS

)


# --------------------------------
# 26. Build LangGraph Workflow
# --------------------------------

langgraph_builder = StateGraph(

    MessagesState

)


# Agent Node

langgraph_builder.add_node(

    "agent",

    langgraph_agent_node

)


# Tool Node

langgraph_builder.add_node(

    "tools",

    langgraph_tool_node

)


# START -> agent

langgraph_builder.add_edge(

    START,

    "agent"

)


# agent -> tools OR END

langgraph_builder.add_conditional_edges(

    "agent",

    tools_condition,

    {

        "tools":
            "tools",

        "__end__":
            END

    }

)


# tools -> agent

langgraph_builder.add_edge(

    "tools",

    "agent"

)


# Compile Graph

langgraph_agent = (
    langgraph_builder.compile()
)


# ============================================
# API ENDPOINTS
# ============================================


# --------------------------------
# 27. Serve Frontend
# --------------------------------

@app.get("/")
def home():

    return FileResponse(
        "index.html"
    )


# --------------------------------
# 28. Main Chat Endpoint
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

                    model=
                        CHAT_MODEL,

                    instructions=
                        SYSTEM_PROMPT,

                    input=
                        conversation_history,

                    tools=
                        TOOLS,

                    tool_choice=
                        "auto",

                    stream=
                        True

                )
            )


            for event in stream:


                if (

                    event.type

                    ==

                    "response.output_text.delta"

                ):


                    chunk = (
                        event.delta
                    )


                    final_answer += (
                        chunk
                    )


                    yield chunk


                elif (

                    event.type

                    ==

                    "response.output_item.added"

                ):


                    if (

                        event.item.type

                        ==

                        "function_call"

                    ):


                        tool_calls[

                            event.output_index

                        ] = event.item


                elif (

                    event.type

                    ==

                    "response.output_item.done"

                ):


                    model_output_items.append(

                        event.item

                    )


                    if (

                        event.item.type

                        ==

                        "function_call"

                    ):


                        tool_calls[

                            event.output_index

                        ] = event.item


            if not tool_calls:


                if final_answer.strip():


                    save_message(
                        conversation_id,
                        "assistant",
                        final_answer
                    )


                return


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

                    !=

                    "calculator"

                ):

                    continue


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


            final_stream = (
                client.responses.create(

                    model=
                        CHAT_MODEL,

                    instructions=
                        SYSTEM_PROMPT,

                    input=
                        tool_context,

                    tools=
                        TOOLS,

                    tool_choice=
                        "none",

                    stream=
                        True

                )
            )


            final_answer = ""


            for event in final_stream:


                if (

                    event.type

                    ==

                    "response.output_text.delta"

                ):


                    chunk = (
                        event.delta
                    )


                    final_answer += (
                        chunk
                    )


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

        media_type=
            "text/plain"

    )


# --------------------------------
# 29. Structured Chat Endpoint
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


    response = (
        client.responses.parse(

            model=
                CHAT_MODEL,

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

            input=
                conversation_history,

            text_format=
                StructuredChatResponse

        )
    )


    structured_answer = (
        response.output_parsed
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
            structured_answer.key_points
        )

    )


    save_message(
        conversation_id,
        "assistant",
        assistant_text
    )


    return (
        structured_answer.model_dump()
    )


# --------------------------------
# 30. Semantic Search Endpoint
# --------------------------------

@app.post("/embeddings/search")
def embedding_search(
    request: EmbeddingSearchRequest
):

    matches = (
        retrieve_documents(
            request.query,
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
            request.query,

        "total_documents":
            vector_collection.count(),

        "best_match":
            matches[0],

        "top_matches":
            matches

    }


# --------------------------------
# 31. RAG Endpoint
# --------------------------------

@app.post("/rag")
def rag_chat(
    request: RAGRequest
):

    retrieved_documents = (
        retrieve_documents(
            request.query,
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


        context_parts.append(

            f"Document {index}:\n"

            f"{item['document']}"

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

        f"{request.query}"

    )


    response = (
        client.responses.create(

            model=
                CHAT_MODEL,

            instructions=
                RAG_SYSTEM_PROMPT,

            input=
                rag_input

        )
    )


    return {

        "query":
            request.query,

        "answer":
            response.output_text,

        "retrieved_documents":
            retrieved_documents

    }


# --------------------------------
# 32. Manual Agent Endpoint
# --------------------------------

@app.post("/agent")
def agent_chat(
    request: AgentRequest
):

    conversation_id = (
        request.conversation_id
    )


    conversation_history = (
        get_conversation_history(
            conversation_id
        )
    )


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


    for step in range(

        1,

        MAX_AGENT_STEPS + 1

    ):


        print(
            f"\nAgent Step: {step}"
        )


        response = (
            client.responses.create(

                model=
                    CHAT_MODEL,

                instructions=
                    AGENT_SYSTEM_PROMPT,

                input=
                    agent_input,

                tools=
                    AGENT_TOOLS,

                tool_choice=
                    "auto"

            )
        )


        agent_input.extend(
            response.output
        )


        tool_calls = [

            item

            for item in response.output

            if item.type == "function_call"

        ]


        if not tool_calls:


            final_answer = (
                response.output_text
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


        for tool_call in tool_calls:


            tool_result = (
                execute_agent_tool(
                    tool_call
                )
            )


            print(
                "Agent requested tool:",
                tool_call.name
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


    raise HTTPException(

        status_code=500,

        detail=(

            "The agent reached the maximum "

            "number of execution steps."

        )

    )


# --------------------------------
# 33. LangChain Agent Endpoint
# --------------------------------

@app.post("/langchain/agent")
def langchain_agent_chat(
    request: AgentRequest
):

    conversation_id = (
        request.conversation_id
    )


    conversation_history = (
        get_conversation_history(
            conversation_id
        )
    )


    messages = list(
        conversation_history
    )


    messages.append({

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


    try:

        result = (
            langchain_agent.invoke({

                "messages":
                    messages

            })
        )


        final_message = (
            result[
                "messages"
            ][-1]
        )


        final_answer = (
            extract_ai_text(
                final_message
            )
        )


        if not final_answer.strip():


            raise ValueError(

                "LangChain returned "

                "an empty final answer."

            )


        save_message(
            conversation_id,
            "assistant",
            final_answer
        )


        return {

            "answer":
                final_answer,

            "message_count":
                len(
                    result[
                        "messages"
                    ]
                )

        }


    except Exception as error:


        print(
            "\nLangChain Agent Error:",
            error
        )


        raise HTTPException(

            status_code=500,

            detail=(

                "The LangChain agent failed "

                "to complete the request."

            )

        ) from error


# --------------------------------
# 34. LangGraph Agent Endpoint
# --------------------------------

@app.post("/langgraph/agent")
def langgraph_agent_chat(
    request: AgentRequest
):

    conversation_id = (
        request.conversation_id
    )


    conversation_history = (
        get_conversation_history(
            conversation_id
        )
    )


    messages = list(
        conversation_history
    )


    messages.append({

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


    try:

        result = (
            langgraph_agent.invoke(

                {

                    "messages":
                        messages

                },

                config={

                    "recursion_limit":
                        12

                }

            )
        )


        final_message = (
            result[
                "messages"
            ][-1]
        )


        final_answer = (
            extract_ai_text(
                final_message
            )
        )


        if not final_answer.strip():


            raise ValueError(

                "LangGraph returned "

                "an empty final answer."

            )


        save_message(
            conversation_id,
            "assistant",
            final_answer
        )


        return {

            "answer":
                final_answer,

            "message_count":
                len(
                    result[
                        "messages"
                    ]
                )

        }


    except Exception as error:


        print(
            "\nLangGraph Agent Error:",
            error
        )


        raise HTTPException(

            status_code=500,

            detail=(

                "The LangGraph agent failed "

                "to complete the request."

            )

        ) from error


# --------------------------------
# 35. Clear Conversation
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