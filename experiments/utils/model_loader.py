import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException
from configs import EMBEDDING_MODEL, LLM_SETTINGS, JUDGE_EMBEDDING_MODEL

# ── Retrieval embedding singleton ─────────────────────────────────────────────
# MiniLM — used exclusively for FAISS indexing and retrieval.
# Loaded once per process, served from local disk cache after first download.
# Never used for RAGAS evaluation metrics.
_EMBEDDING_INSTANCE: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _EMBEDDING_INSTANCE
    if _EMBEDDING_INSTANCE is None:
        try:
            _EMBEDDING_INSTANCE = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"local_files_only": True},
            )
            log.info("Retrieval embedding model loaded from local cache", model=EMBEDDING_MODEL)
        except Exception:
            log.info("Retrieval embedding model not in cache, downloading", model=EMBEDDING_MODEL)
            _EMBEDDING_INSTANCE = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            log.info("Retrieval embedding model downloaded and cached", model=EMBEDDING_MODEL)
    return _EMBEDDING_INSTANCE


# ── Judge embedding singleton ─────────────────────────────────────────────────
# OpenAI text-embedding-3-small — used exclusively for RAGAS evaluation metrics.
# Higher quality than MiniLM for semantic judgment tasks.
# Never used for FAISS indexing or retrieval.
_JUDGE_EMBEDDING_INSTANCE: OpenAIEmbeddings | None = None


def _get_judge_embeddings(api_key: str) -> OpenAIEmbeddings:
    global _JUDGE_EMBEDDING_INSTANCE
    if _JUDGE_EMBEDDING_INSTANCE is None:
        _JUDGE_EMBEDDING_INSTANCE = OpenAIEmbeddings(
            model=JUDGE_EMBEDDING_MODEL,
            openai_api_key=api_key,
        )
        log.info("Judge embedding model initialised", model=JUDGE_EMBEDDING_MODEL)
    return _JUDGE_EMBEDDING_INSTANCE


class ApiKeyManager:
    REQUIRED_KEYS = ["GROQ_API_KEY", "GOOGLE_API_KEY", "HF_TOKEN", "OPENAI_API_KEY"]

    def __init__(self):
        self.api_keys = {}
        for key in self.REQUIRED_KEYS:
            env_val = os.getenv(key)
            if env_val:
                self.api_keys[key] = env_val

        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            log.error("Missing required API keys", missing_keys=missing)
            raise DocumentPortalException("Missing API keys", sys)

    def get(self, key: str) -> str:
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for {key} is missing")
        return val


class ModelLoader:
    def __init__(self):
        load_dotenv()
        self.api_key_mgr = ApiKeyManager()

    def load_embeddings(self) -> HuggingFaceEmbeddings:
        """Return the MiniLM retrieval embedding model (FAISS / retrieval only)."""
        try:
            return _get_embeddings()
        except Exception as e:
            log.error("Error loading retrieval embedding model", error=str(e))
            raise DocumentPortalException("Failed to load retrieval embedding model", sys)

    def load_judge_embeddings(self) -> OpenAIEmbeddings:
        """Return the OpenAI embedding model used exclusively by RAGAS evaluation."""
        try:
            return _get_judge_embeddings(self.api_key_mgr.get("OPENAI_API_KEY"))
        except Exception as e:
            log.error("Error loading judge embedding model", error=str(e))
            raise DocumentPortalException("Failed to load judge embedding model", sys)

    def load_llm(self, provider_override: str = None):
        provider = provider_override or "google"
        cfg      = LLM_SETTINGS[provider]

        if provider == "google":
            return ChatGoogleGenerativeAI(
                model=cfg["model_name"],
                google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
                temperature=cfg["temperature"],
                max_output_tokens=cfg["max_output_tokens"],
            )
        elif provider == "groq":
            return ChatGroq(
                model_name=cfg["model_name"],
                groq_api_key=self.api_key_mgr.get("GROQ_API_KEY"),
                temperature=cfg["temperature"],
            )
        elif provider in ("openai", "openai_judge"):
            # Both "openai" (gpt-4o-mini, RAG pipeline) and
            # "openai_judge" (gpt-4o, RAGAS judge) use ChatOpenAI.
            return ChatOpenAI(
                model=cfg["model_name"],
                openai_api_key=self.api_key_mgr.get("OPENAI_API_KEY"),
                temperature=cfg["temperature"],
                max_tokens=cfg["max_output_tokens"],
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
