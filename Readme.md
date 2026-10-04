# My AI Journey 2026

This repository documents my step-by-step journey into Artificial Intelligence and Generative AI through practical projects.

# Project 1 — AI Chatbot

The goal of this project is to progressively build a production-style AI chatbot while learning the concepts behind LLM applications.

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

Example request:

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

Example:

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

Endpoint:

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

Example:

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

Example:

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

# Upcoming Milestones

## Milestone 8 — Prompt Engineering

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

## Milestone 9 — Streaming Responses

Make AI responses appear gradually instead of waiting for the entire answer.

## Milestone 10 — Structured Outputs

Make the LLM return predictable structured data.

## Milestone 11 — Tool Calling

Allow the LLM to call Python functions and external APIs.

## Milestone 12 — Embeddings

Learn how text is converted into vectors.

## Milestone 13 — Vector Database

Store and retrieve embeddings.

## Milestone 14 — RAG

Build document/PDF question answering using retrieval augmented generation.

## Milestone 15 — Agents

Allow the AI to decide which tools to use and perform multi-step tasks.

## Milestone 16 — LangChain

Understand how LangChain simplifies parts of our existing architecture.

## Milestone 17 — LangGraph

Build stateful agent workflows and multi-step execution graphs.

## Milestone 18 — MCP

Learn how AI applications connect to tools and external systems using the Model Context Protocol.

## Milestone 19 — Deployment

Deploy the chatbot so it can be used outside localhost.

# Goal

The goal of this project is not only to build a chatbot, but to understand how modern AI applications are designed from the ground up.

Each milestone introduces a new AI or backend engineering concept while improving the same application.