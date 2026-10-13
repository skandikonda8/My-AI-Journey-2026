# My AI Journey 2026

This repository documents my step-by-step journey into Artificial Intelligence and Generative AI through practical projects.

# Project 1 — AI Chatbot

The goal of this project is to progressively build a production-style AI chatbot while learning the concepts behind modern LLM applications.

---

## Milestone 1 — Basic AI Chatbot API

Built the first working connection between a Python backend and an LLM.

### Learned

- FastAPI basics
- API endpoints
- GET requests
- Query parameters
- OpenAI API integration
- LLM request/response flow
- Environment variables
- API key handling

### Architecture

```text
Browser
   ↓
FastAPI
   ↓
OpenAI API
   ↓
LLM
   ↓
JSON Response
```

---

## Milestone 2 — Chatbot Web Interface

Added a browser-based chatbot interface using HTML, CSS, and JavaScript.

### Learned

- Frontend vs backend
- HTML/CSS basics
- JavaScript
- `fetch()`
- `async` / `await`
- Displaying AI responses dynamically

### Architecture

```text
User
↓
HTML / JavaScript
↓
FastAPI
↓
LLM
↓
FastAPI
↓
JavaScript
↓
Chat UI
```

---

## Milestone 3 — POST API and JSON Requests

Changed the chatbot from GET requests to POST requests.

### Learned

- GET vs POST
- JSON request bodies
- JSON responses
- HTTP headers
- `Content-Type`
- Pydantic models
- Request validation

### Example Request

```json
{
  "message": "What is artificial intelligence?"
}
```

---

## Milestone 4 — Conversation Memory

Added temporary conversation memory so the chatbot can remember earlier messages in the same conversation.

### Learned

- Conversation history
- User and assistant roles
- Context sent to an LLM
- Short-term chatbot memory
- Why LLMs do not automatically remember previous requests
- Token growth with conversation history

### Example

```text
User: My name is Subhash.

AI: Nice to meet you, Subhash.

User: What is my name?

AI: Your name is Subhash.
```

---

## Milestone 5 — New Chat / Clear Memory

Added the ability to start a new conversation and clear existing chatbot memory.

### Learned

- Backend state
- Clearing frontend state vs backend state
- Creating API endpoints for application actions

### Endpoint

```text
POST /clear
```

---

## Milestone 6 — Conversation IDs

Added unique conversation IDs so multiple conversations can have separate histories.

### Learned

- UUIDs
- Conversation/session identification
- State management
- Separating multiple conversations

### Example

```text
conversation_1
   ↓
History A

conversation_2
   ↓
History B
```

---

## Milestone 7 — Persistent Memory with SQLite

Moved conversation history from temporary Python memory into a SQLite database.

### Learned

- Persistent memory
- SQLite
- Database tables
- INSERT
- SELECT
- DELETE
- Parameterized SQL queries
- Storing conversation history
- Loading conversation history before calling the LLM
- Why in-memory storage disappears when a server restarts

### Database Structure

```text
messages

id
conversation_id
role
content
created_at
```

### Example

```text
1 | abc123 | user      | My name is Subhash
2 | abc123 | assistant | Nice to meet you, Subhash!
3 | abc123 | user      | What is my name?
4 | abc123 | assistant | Your name is Subhash.
```

### Architecture

```text
User
 ↓
Frontend
 ↓
Conversation ID
 ↓
FastAPI
 ↓
SQLite Database
 ↓
Load Conversation History
 ↓
OpenAI API
 ↓
LLM
 ↓
Save New Messages
 ↓
SQLite Database
 ↓
Frontend
```

The local SQLite database is excluded from GitHub because it may contain conversation data.

---

## Milestone 8 — Prompt Engineering

Improved control over the chatbot's behavior using structured system prompts.

### Learned

- System prompts
- User prompts
- Role prompting
- Prompt structure
- Zero-shot prompting
- Few-shot prompting
- Prompt constraints
- Context
- Hallucination reduction
- Prompt engineering vs model training
- Introduction to prompt injection

---

## Milestone 9 — Streaming AI Responses

Added real-time streaming so AI responses appear progressively while the model is generating them.

### Learned

- OpenAI response streaming
- `stream=True`
- Streaming events
- `response.output_text.delta`
- Python generators
- `yield`
- FastAPI `StreamingResponse`
- Browser `ReadableStream`
- `getReader()`
- `TextDecoder`
- Processing response chunks
- Building and storing the complete streamed response
- Streaming responses while maintaining SQLite conversation memory

### Architecture

```text
User
↓
FastAPI
↓
SQLite Conversation History
↓
OpenAI Streaming API
↓
Text Chunks
↓
FastAPI StreamingResponse
↓
JavaScript Stream Reader
↓
Live Chat Interface
```

---

## Milestone 10 — Structured Outputs

Added structured AI responses using Pydantic schemas and the OpenAI Responses API.

### Learned

- Free-form text vs structured data
- Structured Outputs
- JSON schemas
- Pydantic output models
- `client.responses.parse()`
- `text_format`
- `response.output_parsed`
- Typed LLM responses
- Enum-like constraints using `Literal`
- Machine-readable vs human-readable AI responses
- Structured Outputs vs JSON mode
- Using LLM output inside application logic

### Example Structured Response

```json
{
  "topic": "Python decorators",
  "difficulty": "intermediate",
  "answer": "Decorators extend the behavior of functions.",
  "key_points": [
    "Decorators wrap functions",
    "They commonly use the @ syntax",
    "They support reusable behavior"
  ]
}
```

---

## Milestone 11 — Tool Calling

Added tool calling directly into the main chatbot workflow.

The frontend now sends all normal user messages to one `/chat` endpoint. The LLM decides whether it can answer directly or needs to call an available Python tool.

### First Tool

Calculator

Supported operations:

- Addition
- Subtraction
- Multiplication
- Division

### Learned

- Function calling
- Tool definitions
- JSON Schema parameters
- Automatic tool selection
- `tool_choice`
- `function_call`
- Function arguments
- Python-side tool execution
- `function_call_output`
- `call_id`
- LLM → Tool → LLM workflow
- Combining tool calling with streaming
- Keeping tool-routing logic hidden from the frontend

### Architecture

```text
User
↓
Frontend
↓
POST /chat
↓
FastAPI
↓
LLM
↓
Does the question require a tool?

No
↓
Generate normal answer
↓
Stream response to browser

Yes
↓
Function Call Request
↓
Python Tool Execution
↓
Tool Result
↓
Result Sent Back to LLM
↓
LLM Generates Final Response
↓
Stream Response to Browser
```

---

## Milestone 12 — Embeddings and Semantic Search

Added text embeddings and built a basic semantic search system.

The application can now convert text into numerical vectors and compare the semantic meaning of a user query against sample documents.

### Embedding Model

`text-embedding-3-small`

### What Was Added

- Sample knowledge documents
- Query embeddings
- Document embeddings
- Cosine similarity calculation
- Semantic document search
- Ranking documents based on similarity
- Embedding caching

### Example

Text:

```text
Python inheritance allows a child class to reuse another class.
```

Becomes conceptually:

```text
[0.021, -0.047, 0.083, ...]
```

The numbers represent the semantic meaning of the text.

### Learned

- Embeddings
- Vector representations
- Embedding models
- `text-embedding-3-small`
- Semantic similarity
- Cosine similarity
- Query embeddings
- Document embeddings
- Semantic search
- Embedding caching
- Keyword search vs semantic search
- LLMs vs embedding models
- Ranking documents by similarity
- Foundation of vector databases
- Foundation of RAG

### Semantic Search Flow

```text
User Query
↓
Embedding Model
↓
Query Vector
↓
Compare with Document Vectors
↓
Cosine Similarity
↓
Calculate Similarity Scores
↓
Rank Documents
↓
Return Most Relevant Documents
```

### Example Search

User asks:

```text
How can one Python class reuse another class?
```

The wording does not need to exactly match the stored document.

The embedding system compares semantic meaning and can identify the document about Python inheritance as the most relevant result.

---

## Milestone 13 — Vector Database

Added ChromaDB as a persistent vector database for storing and searching document embeddings.

Instead of keeping vectors only inside Python memory, embeddings can now be stored persistently and searched using a vector database.

### Vector Database

`ChromaDB`

### What Changed

Before:

```text
Documents
↓
OpenAI Embedding Model
↓
Document Embeddings
↓
Python Memory
↓
Manual Cosine Similarity
↓
Sort Results
↓
Best Matching Document
```

Now:

```text
Documents
↓
OpenAI Embedding Model
↓
Document Embeddings
↓
ChromaDB
↓
Vector Index
↓
Similarity Search
↓
Top Matching Documents
```

### What Was Added

- ChromaDB
- Persistent vector storage
- Chroma `PersistentClient`
- Chroma collection
- Document IDs
- Document metadata
- Vector indexing
- Semantic vector search
- Persistent embeddings

### Learned

- Vector databases
- ChromaDB
- Persistent vector storage
- `PersistentClient`
- Collections
- Document IDs
- Metadata
- Vector indexing
- HNSW
- Cosine distance
- Query embeddings
- Nearest-neighbor search
- Persistent embeddings
- Semantic retrieval
- SQLite vs vector databases
- Foundation for RAG

### Vector Database Structure

```text
Chroma Collection

Document 1
├── ID
├── Original Text
├── Embedding Vector
└── Metadata

Document 2
├── ID
├── Original Text
├── Embedding Vector
└── Metadata
```

### Architecture

```text
Documents
↓
OpenAI Embedding Model
↓
Document Embeddings
↓
Chroma Vector Database
↓
Vector Index
```

Semantic search:

```text
User Query
↓
Embedding Model
↓
Query Embedding
↓
ChromaDB
↓
Vector Similarity Search
↓
Nearest Documents
↓
Return Top Matches
```

### Database Responsibilities

```text
SQLite
↓
Conversation Memory

ChromaDB
↓
Semantic Knowledge / Document Vectors
```

---


## Milestone 14 — Retrieval Augmented Generation (RAG)

Connected the ChromaDB retrieval system to the LLM and built the first Retrieval Augmented Generation workflow.

The application can now retrieve relevant documents from the vector database and provide those documents to the LLM as context before generating the final answer.

This allows the model to answer questions using retrieved knowledge instead of relying only on its general training.

### What Changed

Before RAG:

```text
User Question
↓
Embedding Model
↓
Query Vector
↓
ChromaDB
↓
Relevant Documents
↓
Return Documents
```

This was semantic search.

Now:

```text
User Question
↓
Embedding Model
↓
Query Vector
↓
ChromaDB
↓
Retrieve Relevant Documents
↓
Build Context
↓
Context + User Question
↓
LLM
↓
Grounded Answer
```

This is Retrieval Augmented Generation.

### RAG Meaning

RAG stands for:

```text
R = Retrieval
    Retrieve relevant information

A = Augmented
    Add retrieved information to the prompt

G = Generation
    Generate the final answer using the LLM
```

### What Was Added

- `RAGRequest`
- Dedicated `/rag` endpoint
- Reusable document retrieval function
- Retrieved document context
- Context building
- RAG-specific system prompt
- Context + question prompt construction
- LLM answer generation from retrieved knowledge
- Grounded responses
- Retrieval transparency by returning retrieved documents

### New Request Model

The RAG endpoint uses:

```python
class RAGRequest(BaseModel):
    query: str
```

Example request:

```json
{
  "query": "How does inheritance work in Python?"
}
```

### Document Retrieval Function

Created a reusable function:

```text
retrieve_documents()
```

This function performs:

```text
User Query
↓
Create Query Embedding
↓
Search ChromaDB
↓
Retrieve Similar Documents
↓
Calculate Similarity Information
↓
Return Retrieved Documents
```

Both semantic search and RAG can now reuse the same retrieval logic.

### Why Retrieval Was Moved Into a Function

Before:

```text
/embeddings/search
↓
Create embedding
↓
Search Chroma
↓
Extract documents
↓
Format results
```

If the same logic was copied into `/rag`, the code would be duplicated.

Now:

```text
                    retrieve_documents()
                    ↑                ↑
                    │                │
      /embeddings/search           /rag
```

This keeps the retrieval logic reusable and easier to maintain.

### Context Building

After the relevant documents are retrieved, they are combined into one context string.

Example:

```text
Document 1:
Python inheritance allows a child class to reuse
properties and methods from a parent class.

Document 2:
FastAPI is a Python web framework used to build APIs
and backend services.

Document 3:
SQLite is a lightweight relational database.
```

The combined information is stored in:

```text
context
```

### RAG Input

The final prompt sent to the LLM contains both the retrieved context and the original user question.

Example:

```text
CONTEXT:

Document 1:
Python inheritance allows a child class to reuse
properties and methods from a parent class.

Document 2:
FastAPI is a Python web framework used to build APIs.

USER QUESTION:

How does inheritance work in Python?
```

This combined input is stored in:

```text
rag_input
```

### RAG System Prompt

A separate system prompt was added for RAG.

The model is instructed to:

- Use the provided context
- Answer using retrieved information
- Avoid inventing unsupported information
- Say when the context does not contain enough information
- Keep answers beginner-friendly

### RAG Endpoint

Added:

```text
POST /rag
```

Example request:

```json
{
  "query": "How does inheritance work in Python?"
}
```

### RAG Execution Flow

```text
POST /rag
↓
RAGRequest
↓
request.query
↓
query_text
↓
retrieve_documents()
↓
create_embedding()
↓
OpenAI Embedding Model
↓
query_embedding
↓
ChromaDB
↓
search_results
↓
retrieved_documents
↓
Build Context
↓
context
↓
Build RAG Input
↓
rag_input
↓
OpenAI Responses API
↓
LLM
↓
llm_response
↓
output_text
↓
answer
↓
Return JSON Response
```

### Example Response

```json
{
  "query": "How does inheritance work in Python?",
  "answer": "Python inheritance allows a child class to reuse properties and methods from a parent class.",
  "retrieved_documents": [
    {
      "id": "doc_1",
      "document": "Python inheritance allows a child class to reuse properties and methods from a parent class.",
      "metadata": {
        "topic": "python",
        "category": "programming"
      },
      "similarity": 0.89
    }
  ]
}
```

### Semantic Search vs RAG

Semantic Search:

```text
Question
↓
Embedding
↓
Vector Database
↓
Relevant Documents
```

Output:

```text
Documents
```

RAG:

```text
Question
↓
Embedding
↓
Vector Database
↓
Relevant Documents
↓
LLM
↓
Generated Answer
```

Output:

```text
Grounded Natural-Language Answer
```

### Main Variables

`query_text`

Stores the user's original question.

`query_embedding`

Stores the numerical vector representing the question.

`search_results`

Stores the raw ChromaDB search result.

`retrieved_documents`

Stores the cleaned list of relevant documents returned from ChromaDB.

`context_parts`

Stores individual formatted document strings.

`context`

Stores all retrieved documents combined into one string.

`rag_input`

Stores the retrieved context together with the user's question.

`llm_response`

Stores the response object returned by the LLM.

`answer`

Stores the final generated RAG answer.

### Learned

- Retrieval Augmented Generation
- RAG architecture
- Retrieval
- Augmentation
- Generation
- Context retrieval
- Context construction
- Prompt augmentation
- Grounded generation
- ChromaDB + LLM integration
- Reusable retrieval functions
- Avoiding duplicated backend logic
- Semantic search vs RAG
- Using retrieved knowledge as LLM context
- RAG-specific system prompts
- Reducing unsupported answers
- Returning sources alongside generated answers
- Foundation for document question answering

### Architecture

```text
                    User Question
                          ↓
                       FastAPI
                          ↓
                  Create Query Embedding
                          ↓
               OpenAI Embedding Model
                          ↓
                    Query Vector
                          ↓
                       ChromaDB
                          ↓
                 Similarity Search
                          ↓
                Relevant Documents
                          ↓
                    Build Context
                          ↓
             Context + User Question
                          ↓
                    OpenAI LLM
                          ↓
                 Grounded Answer
```

### Current AI Application Architecture

```text
                         User
                          ↓
                     Web Interface
                          ↓
                        FastAPI
                          ↓
        ┌─────────────────┼──────────────────┐
        ↓                 ↓                  ↓
     SQLite           Tool Calling       RAG Pipeline
        ↓                 ↓                  ↓
 Conversation        Calculator         Embeddings
   Memory                                    ↓
                                          ChromaDB
                                            ↓
                                     Relevant Documents
                                            ↓
                                           LLM
                                            ↓
                                     Grounded Answer
```

---



## Milestone 15 — Basic AI Agent

Added a multi-step AI agent that can decide which tools are needed, execute those tools, inspect their results, and continue until the task is complete.

This milestone extends simple tool calling into an iterative agent workflow.

### What Is an Agent?

An AI agent can do more than answer a question once.

It can:

```text
Decide
↓
Take Action
↓
Observe the Result
↓
Decide Again
↓
Repeat
↓
Final Answer
```

### Agent Formula

```text
Agent
=
LLM
+
Tools
+
Decision Making
+
Execution Loop
+
State / Context
```

### Available Agent Tools

The agent currently has two tools.

#### 1. Calculator

Used for:

- Addition
- Subtraction
- Multiplication
- Division

#### 2. Knowledge Search

Used for:

- Searching the local ChromaDB knowledge base
- Retrieving semantically relevant documents
- Returning retrieved information to the agent

### What Was Added

- `AgentRequest`
- `AGENT_SYSTEM_PROMPT`
- `MAX_AGENT_STEPS`
- `search_knowledge()`
- `KNOWLEDGE_SEARCH_TOOL`
- `AGENT_TOOLS`
- `execute_agent_tool()`
- `/agent` endpoint
- Multi-step agent loop
- Dynamic tool selection
- Multiple tool calls
- Tool result observation
- Agent execution trace
- Maximum-step protection

### Agent Request Model

```python
class AgentRequest(BaseModel):
    conversation_id: str
    message: str
```

Example request:

```json
{
  "conversation_id": "agent-test-1",
  "message": "Search the knowledge base for embeddings and multiply the number of returned documents by 10."
}
```

### Tool Router

The model does not directly execute Python functions.

It requests a tool, and the backend routes that request.

```text
LLM Tool Request
↓
execute_agent_tool()
↓
Check Tool Name
↓
┌────────────────────┬─────────────────────────┐
↓                    ↓
calculator       search_knowledge
↓                    ↓
calculate()      retrieve_documents()
↓                    ↓
Result           ChromaDB Results
```

### Agent Loop

The main difference between normal tool calling and an agent is the loop.

```text
LLM
↓
Decide
↓
Use Tool
↓
Observe Tool Result
↓
Decide Again
↓
Use Another Tool If Needed
↓
Repeat
↓
Final Answer
```

The agent continues until:

```text
No more tools are needed
```

or:

```text
MAX_AGENT_STEPS is reached
```

### Example Multi-Step Task

User request:

```text
Search the knowledge base for embeddings
and multiply the number of returned documents by 10.
```

Possible execution:

```text
STEP 1

LLM
↓
search_knowledge
↓
ChromaDB
↓
3 documents returned


STEP 2

LLM sees:
match_count = 3
↓
calculator
↓
3 × 10
↓
30


STEP 3

LLM
↓
No more tools needed
↓
Final Answer
```

### Function Call Flow

```text
agent_response.output
↓
Find function_call items
↓
tool_call
↓
execute_agent_tool()
↓
tool_result
↓
function_call_output
↓
Add result to agent_input
↓
Call LLM Again
```

### Agent State

The variable:

```text
agent_input
```

stores the information needed for the next agent step.

It may contain:

```text
Previous Conversation
+
Current User Message
+
Model Tool Requests
+
Tool Results
```

This allows the agent to continue from what already happened.

### Tool Trace

The endpoint also returns:

```text
tool_trace
```

This makes the tool execution easier to inspect.

Example:

```json
{
  "tool_trace": [
    {
      "step": 1,
      "tool": "search_knowledge"
    },
    {
      "step": 2,
      "tool": "calculator"
    }
  ]
}
```

### Tool Calling vs Agent

Milestone 11:

```text
User
↓
LLM
↓
Tool
↓
LLM
↓
Answer
```

Milestone 15:

```text
User
↓
LLM
↓
Tool
↓
LLM
↓
Another Tool
↓
LLM
↓
Another Action If Needed
↓
Final Answer
```

The important difference is the repeated decision loop.

### Full Agent Architecture

```text
                    User
                     ↓
                POST /agent
                     ↓
                AgentRequest
                     ↓
                 agent_input
                     ↓
              ┌──────────────┐
              │  AGENT LOOP  │
              └──────────────┘
                     ↓
                    LLM
                     ↓
              Tool Required?
                /         \
              No           Yes
              ↓             ↓
        Final Answer    function_call
                            ↓
                   execute_agent_tool()
                     /             \
                    /               \
             calculator       search_knowledge
                 ↓                  ↓
            calculate()      ChromaDB Retrieval
                 ↓                  ↓
              result             result
                    \             /
                     \           /
                  function_call_output
                           ↓
                      agent_input
                           ↓
                      NEXT STEP
```

### Learned

- AI agents
- Agentic AI basics
- Tool-using agents
- Multi-step execution
- Agent loops
- Dynamic decision making
- Tool routing
- Multiple tools
- Function calling
- `function_call`
- `function_call_output`
- `call_id`
- Maintaining agent context
- Tool result observation
- Maximum execution limits
- Reusing RAG retrieval as an agent tool
- Difference between tool calling and agents
- Decide → Act → Observe → Repeat

--- 

---

## Milestone 16 — LangChain Agent

Added LangChain to the project and rebuilt the existing agent workflow using LangChain's agent and tool abstractions.

The manual agent from Milestone 15 was kept in the project so the difference between building an agent manually and using an agent framework can be clearly understood.

### Why LangChain?

In Milestone 15, the complete agent loop was written manually.

The application had to handle:

```text
LLM Request
↓
Detect Tool Calls
↓
Read Tool Name
↓
Parse Tool Arguments
↓
Route Tool
↓
Execute Python Function
↓
Create Tool Result
↓
Send Tool Result Back To LLM
↓
Call LLM Again
↓
Repeat
↓
Final Answer
```

LangChain provides abstractions that manage much of this orchestration automatically.

### Packages Added

```bash
pip install -U langchain langchain-openai
```

### New Imports

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
```

### LangChain Model

Added:

```python
langchain_model = ChatOpenAI(
    model=CHAT_MODEL,
    use_responses_api=True
)
```

This creates a LangChain model wrapper around the OpenAI model.

### Manual OpenAI Client vs LangChain Model

The project now contains both:

```text
OpenAI SDK

client = OpenAI(...)
```

used by:

```text
/chat
/rag
/agent
/embeddings/search
```

and:

```text
LangChain

langchain_model = ChatOpenAI(...)
```

used by:

```text
/langchain/agent
```

### LangChain Tools

The existing Python calculator and knowledge retrieval functions were reused as LangChain tools.

#### Calculator Tool

```python
@tool("calculator")
def langchain_calculator(
    operation: str,
    a: float,
    b: float
) -> dict:
```

The tool internally reuses:

```text
calculate()
```

### Knowledge Search Tool

```python
@tool("search_knowledge")
def langchain_search_knowledge(
    query: str
) -> dict:
```

The tool internally reuses:

```text
search_knowledge()
↓
retrieve_documents()
↓
ChromaDB
```

### What `@tool` Does

Before LangChain, tool schemas were manually written using JSON.

Example:

```text
Tool Name
+
Description
+
Parameters
+
Required Fields
+
JSON Schema
```

LangChain can create the tool representation using:

```text
Python Function
+
Type Hints
+
Docstring
```

Example:

```python
@tool("calculator")
def langchain_calculator(
    operation: str,
    a: float,
    b: float
) -> dict:
```

### LangChain Agent

Created the agent using:

```python
langchain_agent = create_agent(
    model=langchain_model,
    tools=[
        langchain_calculator,
        langchain_search_knowledge
    ],
    system_prompt=LANGCHAIN_AGENT_PROMPT
)
```

The agent receives:

```text
Model
+
Tools
+
System Prompt
```

LangChain then manages the repeated model and tool execution cycle.

### New Endpoint

Added:

```text
POST /langchain/agent
```

Example request:

```json
{
  "conversation_id": "lc-test-1",
  "message": "Search the knowledge base for embeddings and multiply the number of returned documents by 10."
}
```

### LangChain Agent Flow

```text
User
↓
POST /langchain/agent
↓
AgentRequest
↓
Load SQLite Conversation History
↓
messages
↓
langchain_agent.invoke()
↓
LangChain Agent
↓
LLM
↓
Tool Required?
↓
┌────────────────────┬─────────────────────┐
↓                    ↓
calculator       search_knowledge
↓                    ↓
calculate()      ChromaDB Retrieval
↓                    ↓
Tool Result       Tool Result
└──────────────┬─────┘
               ↓
       LangChain Agent Loop
               ↓
      Another Tool Needed?
               ↓
          Final Answer
               ↓
          Save SQLite
               ↓
         API Response
```

### Example Multi-Step Execution

User:

```text
Search the knowledge base for embeddings
and multiply the number of returned documents by 10.
```

Possible execution:

```text
Step 1

LangChain Agent
↓
search_knowledge
↓
ChromaDB
↓
3 documents returned


Step 2

LangChain Agent
↓
calculator
↓
3 × 10
↓
30


Step 3

LangChain Agent
↓
Final Answer
```

### Manual Agent vs LangChain Agent

Manual Agent:

```text
POST /agent
↓
OpenAI Responses API
↓
Our Python Agent Loop
↓
Detect Tool Calls
↓
execute_agent_tool()
↓
function_call_output
↓
Call LLM Again
```

LangChain Agent:

```text
POST /langchain/agent
↓
langchain_agent.invoke()
↓
LangChain
↓
Model
↓
Tool
↓
Tool Result
↓
Model
↓
Final Answer
```

### What Changed?

Before:

```text
We manually managed:

Tool Schemas
+
Function Calls
+
Tool Routing
+
Tool Results
+
Agent Loop
+
State Passing
```

Now:

```text
We provide:

Model
+
Tools
+
Prompt

↓

LangChain manages the agent loop
```

### Conversation Memory

The LangChain agent still uses the existing SQLite conversation history.

```text
conversation_id
↓
SQLite
↓
get_conversation_history()
↓
messages
↓
LangChain Agent
```

This allows existing project memory to work with the new LangChain agent.

### Important Variables

`langchain_model`

Stores the OpenAI model wrapped by LangChain.

`langchain_calculator`

LangChain calculator tool.

`langchain_search_knowledge`

LangChain knowledge retrieval tool.

`langchain_agent`

Agent created using `create_agent()`.

`messages`

Stores the conversation history and current user message.

`result`

Stores the complete result returned by the LangChain agent.

`result["messages"]`

Stores the complete message sequence generated during agent execution.

`final_message`

Stores the last message generated by the agent.

`final_answer`

Stores the final text returned to the user.

### Current Agent Architecture

```text
                         FastAPI
                            ↓
             ┌──────────────┴──────────────┐
             ↓                             ↓
          /agent                    /langchain/agent
             ↓                             ↓
       Manual Agent                  LangChain Agent
             ↓                             ↓
     Our Python Loop                  create_agent()
             ↓                             ↓
     ┌───────┴────────┐             ┌──────┴─────────┐
     ↓                ↓             ↓                ↓
calculator     search_knowledge  calculator   search_knowledge
                      ↓                              ↓
                   ChromaDB                       ChromaDB
```

### Why Keep Both Agents?

The manual agent teaches:

```text
How agent loops actually work
```

The LangChain agent teaches:

```text
How frameworks simplify agent development
```

Keeping both makes it easier to understand exactly what LangChain is doing for the application.

### Learned

- LangChain
- `ChatOpenAI`
- `create_agent()`
- `@tool`
- LangChain tools
- Tool schemas from Python functions
- Tool descriptions using docstrings
- LangChain model wrappers
- Agent invocation
- `agent.invoke()`
- Agent message state
- Framework-managed agent loops
- Reusing existing Python functions
- Reusing ChromaDB retrieval
- Manual agents vs LangChain agents
- Agent orchestration
- Model + Tools + Prompt architecture

---

## Milestone 17 — LangGraph

Next: build the agent workflow using LangGraph and learn:

- State
- Nodes
- Edges
- Conditional routing
- Tool nodes
- Graph execution
- Stateful workflows

---

## Milestone 17 — LangGraph

Build stateful agent workflows and multi-step execution graphs.

---

## Milestone 18 — MCP

Learn how AI applications connect to tools and external systems using the Model Context Protocol.

---

## Milestone 19 — Deployment

Deploy the chatbot so it can be used outside localhost.

---



# Goal

The goal of this project is not only to build a chatbot, but to understand how modern AI applications are designed from the ground up.

Each milestone introduces a new AI or backend engineering concept while improving the same application.