from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from multi_doc_chat.model.models import UploadResponse, ChatResponse, ChatRequest, ChatAnswer
from multi_doc_chat.utils.document_ops import FastAPIFileAdapter
from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from langchain_core.messages import HumanMessage, AIMessage
from multi_doc_chat.exception.custom_exception import DocumentPortalException

app = FastAPI(title="multi-doc=chat", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Static and Templates
BASE_DIR = Path(__file__).resolve().parent
static_dir = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# Storage directories (use env vars for GCS mount)
DATA_DIR = os.getenv("DATA_DIR", "data")
FAISS_DIR = os.getenv("FAISS_DIR", "faiss_index")

SESSIONS: Dict[str, List[dict]] = {}

@app.get('/health')
def health() -> Dict[str, str]:
    return {"status" : "ok"}

@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request":request})

@app.post('/upload', response_model=UploadResponse)
async def upload(files: List[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    try:
        # Wrap FastAPI files to preserve filename/text and provide a read buffer
        wrapped_files = [FastAPIFileAdapter(f) for f in files]

        ingestor = ChatIngestor(
            use_session_dirs=True,
            temp_base=DATA_DIR,
            faiss_base=FAISS_DIR
        )
        session_id = ingestor.session_id

        # Save, load, split, embed and write FAISS index
        ingestor.built_retriver(uploaded_files=wrapped_files)

        # Initialize empty history for this session
        SESSIONS[session_id] = []

        return UploadResponse(session_id=session_id, indexed=True, message="Indexing Complete")
    except DocumentPortalException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")
    
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id
    message = req.message.strip()
    if not session_id or session_id not in SESSIONS:
        raise HTTPException(status_code=400, detail="Invalid or expired session_id. Re-upload documents")
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Build RAG and load retriever from persisted FAISS
        rag = ConversationalRAG(session_id=session_id)
        index_path = f"{FAISS_DIR}/{session_id}"
        rag.load_retriever_from_faiss(index_path=index_path)

        # Use simple in-memory history and convert to BaseMessage List
        simple = SESSIONS.get(session_id, [])
        lc_history = []
        for m in simple:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                lc_history.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_history.append(AIMessage(content=content))
        
        answer = rag.invoke(message, chat_history=lc_history)

        # Update History
        simple.append({"role":"user", "content":message})
        simple.append({"role":"assistant", "content" : answer})
        SESSIONS[session_id] = simple

        return ChatResponse(answer=answer)
    except DocumentPortalException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed : {e}")

if __name__=="__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))  # Convert string to int
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
