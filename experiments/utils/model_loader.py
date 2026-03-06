import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException
from configs import EMBEDDING_MODEL, LLM_SETTINGS


class ApiKeyManager:
    REQUIRED_KEYS = ["GROQ_API_KEY", "GOOGLE_API_KEY", "HF_TOKEN"]

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

    def load_embeddings(self):
        try:
            return HuggingFaceEndpointEmbeddings(
                huggingfacehub_api_token=self.api_key_mgr.get("HF_TOKEN"),
                model=EMBEDDING_MODEL,
            )
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise DocumentPortalException("Failed to load embedding model", sys)

    def load_llm(self, provider_override: str = None):
        provider = provider_override or "google"
        cfg = LLM_SETTINGS[provider]

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
        else:
            raise ValueError(f"Unsupported provider: {provider}")
