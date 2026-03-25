import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from pydantic import ValidationError

from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger import GLOBAL_LOGGER as log
from prompts.prompt_library import PROMPT_REGISTRY
from model.models import PromptType, ChatAnswer


class EvalConversationalRAG:

    def __init__(self, session_id: Optional[str], retriever=None, provider_override: str = None):
        try:
            self.session_id       = session_id
            self.provider_override = provider_override

            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXT_QA.value
            ]

            self.retriever           = retriever
            self.chain               = None
            self.last_retrieved_docs: list = []

            if self.retriever is not None:
                self._build_lcel_chain()

            log.info("EvalConversationalRAG initialized", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to initialize EvalConversationalRAG", error=str(e))
            raise DocumentPortalException("Initialization error in EvalConversationalRAG", sys)

    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: int = 5,
        index_name: str = "index",
        search_type: str = "similarity",
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

            embeddings  = ModelLoader().load_embeddings()
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,
            )

            if search_kwargs is None:
                search_kwargs = {"k": k}
                if search_type == "mmr":
                    search_kwargs["fetch_k"]     = fetch_k
                    search_kwargs["lambda_mult"] = lambda_mult

            self.retriever = vectorstore.as_retriever(
                search_type=search_type, search_kwargs=search_kwargs
            )
            self._build_lcel_chain()

            log.info(
                "FAISS retriever loaded",
                index_path=index_path,
                search_type=search_type,
                k=k,
                session_id=self.session_id,
            )
            return self.retriever

        except Exception as e:
            log.error("Failed to load retriever from FAISS", error=str(e))
            raise DocumentPortalException("Loading error in EvalConversationalRAG", sys)

    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "Chain not initialized. Call load_retriever_from_faiss() first.", sys
                )
            chat_history = chat_history or []
            answer = self.chain.invoke({"input": user_input, "chat_history": chat_history})
            if not answer:
                return "no answer generated."
            try:
                validated = ChatAnswer(answer=str(answer))
                answer    = validated.answer
            except ValidationError as ve:
                log.error("Invalid chat answer", error=str(ve))
                raise DocumentPortalException("Invalid chat answer", sys)
            log.info("Chain invoked", session_id=self.session_id, preview=str(answer)[:150])
            return answer
        except Exception as e:
            log.error("Failed to invoke EvalConversationalRAG", error=str(e))
            raise DocumentPortalException("Invocation error in EvalConversationalRAG", sys)

    def invoke_with_context(
        self,
        user_input: str,
        chat_history: Optional[List[BaseMessage]] = None,
    ) -> dict:
        """Run the full RAG pipeline. Returns {"answer": str, "contexts": List[str]}."""
        try:
            self.last_retrieved_docs = []
            answer   = self.invoke(user_input, chat_history=chat_history)
            contexts = [
                getattr(doc, "page_content", str(doc))
                for doc in self.last_retrieved_docs
            ]
            log.info(
                "invoke_with_context complete",
                session_id=self.session_id,
                chunks_retrieved=len(contexts),
            )
            return {"answer": answer, "contexts": contexts}
        except Exception as e:
            log.error("Failed in invoke_with_context", error=str(e))
            raise DocumentPortalException("invoke_with_context error", sys)

    # ---------- Internals ----------

    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm(provider_override=self.provider_override)
            if not llm:
                raise ValueError("LLM could not be loaded")
            return llm
        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("LLM loading error in EvalConversationalRAG", sys)

    def _format_docs(self, docs) -> str:
        self.last_retrieved_docs = docs
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise DocumentPortalException("No retriever set before building chain", sys)

            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            retrieve_docs = question_rewriter | self.retriever | self._format_docs

            self.chain = (
                {
                    "context":      retrieve_docs,
                    "input":        itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            log.info("LCEL chain built", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to build LCEL chain", error=str(e))
            raise DocumentPortalException("Failed to build LCEL chain", sys)
