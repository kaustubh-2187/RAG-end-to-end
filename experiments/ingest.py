from __future__ import annotations

import sys
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Add the project root (one level above experiments/) to sys.path so that
# the multi_doc_chat package is importable without installing it.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor

load_dotenv()


# CHANGE 1: replaced raw open() + f.name assignment (which raises AttributeError
# on built-in file objects) with a proper wrapper class.
# save_uploaded_files() in file_io.py resolves the filename via getattr(uf, "name", ...)
# and reads content via uf.read() — this wrapper satisfies both requirements.
class _FileWrapper:
    """
    Wraps a binary file path so ChatIngestor.built_retriver() can consume it.
    Matches the interface expected by save_uploaded_files() in file_io.py:
      - .name  → used to detect file extension and pick the right loader
      - .read() → used to write file bytes to the temp directory
    """
    def __init__(self, path: Path):
        self.name = path.name
        self._path = path

    def read(self) -> bytes:
        return self._path.read_bytes()


def build_index(doc_path: str, config: dict, faiss_dir: str, data_dir: str) -> str:
    """
    Build a fresh FAISS index for the given document and config.

    Wipes any existing index at the session path so each
    experiment starts clean — no stale data from a previous run.

    Returns the session_id of the newly built index.
    """
    ingestor = ChatIngestor(
        temp_base=data_dir,
        faiss_base=faiss_dir,
        use_session_dirs=True,
    )
    session_id = ingestor.session_id

    # Wipe the FAISS dir for this session if it already exists
    session_faiss_path = Path(faiss_dir) / session_id
    if session_faiss_path.exists():
        shutil.rmtree(session_faiss_path)
        session_faiss_path.mkdir(parents=True, exist_ok=True)

    # CHANGE 2: use _FileWrapper instead of open() + f.name assignment
    wrapper = _FileWrapper(Path(doc_path))

    ingestor.built_retriver(
        uploaded_files=[wrapper],
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        k=config["k"],
        search_type=config["search_type"],
        fetch_k=config.get("fetch_k", 20),
        lambda_mult=config.get("lambda_mult", 0.5),
    )

    return session_id
