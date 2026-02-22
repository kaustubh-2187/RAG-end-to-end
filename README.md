# Multi-Document Chat

**Live Demo:** https://multi-doc-chat-655964848309.us-central1.run.app

A RAG application that enables conversations with multiple documents simultaneously. Built with FastAPI, LangChain, and FAISS, deployed on Google Cloud Run.

---

## Overview

This project lets you upload documents and chat with them using retrieval-augmented generation. Each document maintains its own isolated conversation history, and you can switch between documents seamlessly. The system chunks and embeds documents into a FAISS vector store, retrieves relevant context at query time, and passes it to an LLM to generate grounded responses.

---

## Architecture

```
User Query
    ↓
[Document Upload] → Text Splitting → Embeddings → FAISS Index
                                                        ↓
[Chat Request] → Query Embedding → Similarity Search → Retrieved Chunks
                                                        ↓
                                              LLM (with context) → Response
```

**Key Components:**

- **Document Ingestion** (`multi_doc_chat/src/document_ingestion/`): Handles file parsing (PDF, DOCX, TXT), text splitting, and embedding generation. Each document gets its own FAISS index stored separately.

- **Document Chat** (`multi_doc_chat/src/document_chat/`): Manages retrieval and response generation. Uses LangChain's conversational retrieval chain with per-document memory buffers.

- **Config** (`multi_doc_chat/config/`): Centralizes LLM provider selection, embedding model, chunking parameters, and retrieval settings.

- **Frontend** (`static/`, `templates/`): Vanilla JS with a 20/80 split layout — document sidebar on the left, chat on the right.

---

## Project Structure

```
RAG/
├── main.py                      # FastAPI entry point
├── requirements.txt
├── Dockerfile
├── Jenkinsfile                  # CI/CD pipeline
│
├── multi_doc_chat/
│   ├── config/                  # Configuration
│   ├── model/                   # Pydantic schemas
│   ├── prompts/                 # Prompt templates
│   ├── src/
│   │   ├── document_ingestion/  # Parsing, chunking, indexing
│   │   └── document_chat/       # Retrieval and chat logic
│   └── utils/
│
├── static/                      # CSS, JS
├── templates/index.html         # Frontend
├── notebook/                    # Experimentation notebooks
└── evaluation/                  # Evaluation scripts
```

---

## Setup

**Prerequisites:**
- Python 3.11+
- At least one API key: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, or `GROQ_API_KEY`

**Local Development:**

```bash
git clone <repo-url>
cd RAG

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Set environment variables
export GOOGLE_API_KEY="your-key"
export GROQ_API_KEY="your-key"

python main.py
```

Visit `http://localhost:8000`

---

## Deployment

**Docker:**

```bash
docker build -t multi-doc-chat .
docker run -p 8000:8000 \
  -e GOOGLE_API_KEY=$GOOGLE_API_KEY \
  -e GROQ_API_KEY=$GROQ_API_KEY \
  multi-doc-chat
```

**Cloud Run:**

The project includes a `Jenkinsfile` pipeline that builds the Docker image, pushes to Google Container Registry, and deploys to Cloud Run.

---

## API Endpoints

```
GET  /health        # Health check
POST /upload        # Upload and index a document
POST /chat          # Send a message to a document
```

Full docs available at `http://localhost:8000/docs` when running locally.
