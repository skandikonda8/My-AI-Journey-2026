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

# Upcoming Milestones

## Milestone 14 — RAG

Build document and PDF question answering using Retrieval Augmented Generation.

The application will retrieve relevant documents from the vector database and provide that context to the LLM before generating an answer.

### Planned Flow

```text
User Question
↓
Create Query Embedding
↓
Search ChromaDB
↓
Retrieve Relevant Documents
↓
Combine Documents + User Question
↓
Send Context to LLM
↓
Generate Grounded Answer
```

---

## Milestone 15 — Agents

Allow the AI to decide which tools to use and perform multi-step tasks.

---

## Milestone 16 — LangChain

Understand how LangChain simplifies parts of the architecture that were first built manually.

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

# Project Architecture So Far

```text
                        User
                         ↓
                    Chat Interface
                         ↓
                       FastAPI
                         ↓
            ┌────────────┼────────────┐
            ↓            ↓            ↓
         SQLite       OpenAI LLM    Embeddings
            ↓            ↓            ↓
      Conversation    Responses      Vectors
         Memory          ↓            ↓
                         ↓         ChromaDB
                         ↓            ↓
                    Tool Calling   Semantic Search
                         ↓            ↓
                    Calculator    Relevant Documents
```

---

# Goal

The goal of this project is not only to build a chatbot, but to understand how modern AI applications are designed from the ground up.

Each milestone introduces a new AI or backend engineering concept while improving the same application.