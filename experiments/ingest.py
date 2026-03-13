from __future__ import annotations

import sys
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

load_dotenv()

# Add experiments dir to path so local imports work
_EXPERIMENTS_DIR = str(Path(__file__).resolve().parent)
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)

from utils.model_loader import ModelLoader


def _generate_session_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"session_{timestamp}_{unique_id}"


def build_index(doc_path: str, config: dict, faiss_dir: str, data_dir: str) -> str:
    """
    Independently builds a FAISS index for the given document and config.
    Completely self-contained — no dependency on multi_doc_chat.
    Returns the session_id of the newly built index.
    """
    session_id = _generate_session_id()
    session_faiss_path = Path(faiss_dir) / session_id

    # Wipe and recreate session dir for a clean run
    if session_faiss_path.exists():
        shutil.rmtree(session_faiss_path)
    session_faiss_path.mkdir(parents=True, exist_ok=True)

    # Load document
    loader = TextLoader(doc_path, encoding="utf-8")
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )
    chunks = splitter.split_documents(docs)

    # Add chunk_id metadata
    for i, chunk in enumerate(chunks):
        source_stem = Path(chunk.metadata.get("source", "unknown")).stem
        chunk.metadata["chunk_id"] = f"{source_stem}_chunk{i}"

    print(f"    Chunks created: {len(chunks)}")

    # Load local embeddings (no internet required)
    embeddings = ModelLoader().load_embeddings()

    # Build and save FAISS index
    texts = [c.page_content for c in chunks]
    metas = [c.metadata for c in chunks]
    vs = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metas)
    vs.save_local(str(session_faiss_path))

    print(f"    FAISS index saved: {session_faiss_path}")

    return session_id
