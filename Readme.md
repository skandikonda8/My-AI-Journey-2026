# My AI Journey 2026

This repository documents my step-by-step journey into Artificial Intelligence and Generative AI.

I am building small projects to understand AI concepts practically instead of only learning theory.

## Project 1 — AI Chatbot

My first project is a simple AI chatbot built using Python and FastAPI.

### Current Features

- Python backend
- FastAPI
- OpenAI API integration
- GPT-6 Luna
- HTML/CSS/JavaScript frontend
- POST API requests
- JSON request and response handling
- Pydantic validation
- Conversation memory
- New Chat / Clear Memory functionality

## Current Architecture

```text
User
  ↓
HTML / CSS / JavaScript
  ↓
FastAPI
  ↓
Conversation History
  ↓
OpenAI API
  ↓
GPT-6 Luna
  ↓
FastAPI
  ↓
Frontend
  ↓
User
```

## Run the Project

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```text
OPENAI_API_KEY=your_api_key_here
```

Run the FastAPI server:

```bash
uvicorn main:app --reload
```

Open the application in the browser:

```text
http://127.0.0.1:8000
```

## AI Learning Roadmap

This repository will gradually cover:

- LLM fundamentals
- Prompt engineering
- Conversation memory
- Conversation IDs
- Persistent memory
- Databases
- Streaming responses
- Structured outputs
- Function calling
- Tool calling
- Embeddings
- Vector databases
- RAG
- PDF/document question answering
- Agents
- LangChain
- LangGraph
- MCP
- Multi-agent workflows
- AI application deployment

## Goal

The goal of this repository is to build a strong practical understanding of AI engineering by developing projects step by step and understanding how every component connects together.