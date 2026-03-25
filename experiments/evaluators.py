from __future__ import annotations

import time
import pandas as pd
from dotenv import load_dotenv

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig
from ragas.metrics import (
    LLMContextRecall,
    LLMContextPrecisionWithReference,
    Faithfulness,
    AnswerRelevancy,
    FactualCorrectness,
    SemanticSimilarity,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from utils.model_loader import ModelLoader
from rag_pipeline import EvalConversationalRAG
from configs import JUDGE_PROVIDER, JUDGE_EMBEDDING_MODEL

load_dotenv()

# Refusal phrases the model is expected to produce for no_answer questions.
REFUSAL_PHRASES = {"i dont know", "i don't know"}

# Columns that are identifiers, not metrics — excluded from numeric averaging.
NON_METRIC_COLS = {"document_index"}

# RAGAS RunConfig — max_workers=1 forces sequential scoring calls.
# This prevents the parallel burst of API requests that causes 429 rate limit
# errors on tier 1 OpenAI accounts. Slower but reliable.
RAGAS_RUN_CONFIG = RunConfig(
    max_workers=1,
    timeout=180,
    max_retries=5,
    max_wait=60,
)


def _is_refusal(answer: str) -> bool:
    """Return True if the model's answer is a refusal / abstention."""
    return answer.strip().lower().rstrip(".") in REFUSAL_PHRASES


def _compute_mrr(samples_meta: list[dict]) -> list[float]:
    """
    MRR per sample = 1 / rank_of_first_hit  (0.0 if no hit in top-k).
    A chunk is a "hit" if it contains at least 2 keywords (len > 4) from the reference.
    """
    mrr_scores = []
    for meta in samples_meta:
        ref_words = {w.lower() for w in meta["reference"].split() if len(w) > 4}
        hit_rank  = None
        for rank, chunk in enumerate(meta["retrieved_contexts"], start=1):
            if sum(1 for w in ref_words if w in chunk.lower()) >= 2:
                hit_rank = rank
                break
        mrr_scores.append(1.0 / hit_rank if hit_rank else 0.0)
    return mrr_scores


def run_ragas_evaluation(
    session_id: str,
    faiss_dir: str,
    provider: str,
    dataset: list,
    config: dict,
) -> tuple[dict, pd.DataFrame]:
    """
    Track A — Answerable QA evaluation (single_passage + multi_passage only).

    One RAG object is created per call and reused across all questions.

    Two separate embedding models are in use within this function:
      - MiniLM (via rag.load_retriever_from_faiss): used for FAISS retrieval only.
      - OpenAI text-embedding-3-small (via judge_loader.load_judge_embeddings):
        used exclusively by RAGAS to score embedding-dependent metrics
        (SemanticSimilarity, AnswerRelevancy). Never touches FAISS.

    RAGAS judge LLM: configured by JUDGE_PROVIDER in configs.py.
    RAGAS judge embeddings: configured by JUDGE_EMBEDDING_MODEL in configs.py.
    Returns (scores_dict, results_dataframe).
    """
    rag = EvalConversationalRAG(session_id=session_id, provider_override=provider)
    rag.load_retriever_from_faiss(
        index_path=f"{faiss_dir}/{session_id}",
        k=config.get("k", 5),
        search_type=config.get("search_type", "similarity"),
        fetch_k=config.get("fetch_k", 20),
        lambda_mult=config.get("lambda_mult", 0.5),
    )

    samples      = []
    samples_meta = []

    for i, example in enumerate(dataset):
        print(f"      [{i+1}/{len(dataset)}] [{example['question_type']}] {example['question'][:60]}...")

        result = rag.invoke_with_context(user_input=example["question"], chat_history=[])
        time.sleep(2)

        samples.append(SingleTurnSample(
            user_input=example["question"],
            response=result["answer"],
            retrieved_contexts=result["contexts"],
            reference=example["answer"],
        ))
        samples_meta.append({
            "question_type":      example["question_type"],
            "reference":          example["answer"],
            "retrieved_contexts": result["contexts"],
        })

    judge_loader = ModelLoader()
    print(f"      RAGAS judge LLM       : {JUDGE_PROVIDER}")
    print(f"      RAGAS judge embeddings: {JUDGE_EMBEDDING_MODEL}")
    ragas_result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[
            LLMContextRecall(),
            LLMContextPrecisionWithReference(),
            Faithfulness(),
            AnswerRelevancy(),
            FactualCorrectness(),
            SemanticSimilarity(),
        ],
        llm=LangchainLLMWrapper(judge_loader.load_llm(provider_override=JUDGE_PROVIDER)),
        embeddings=LangchainEmbeddingsWrapper(judge_loader.load_judge_embeddings()),
        run_config=RAGAS_RUN_CONFIG,
    )

    results_df = ragas_result.to_pandas()

    meta_df    = pd.DataFrame(samples_meta).reset_index(drop=True)
    results_df = pd.concat([results_df.reset_index(drop=True), meta_df[["question_type"]]], axis=1)
    results_df["mrr"] = _compute_mrr(samples_meta)

    metric_cols = [c for c in results_df.select_dtypes(include="number").columns if c not in NON_METRIC_COLS]
    scores = results_df[metric_cols].mean().to_dict()
    # Replace invalid characters in metric names for MLflow compatibility
    cleaned_scores = {}
    for k, v in scores.items():
        clean_key = k.replace("(", "_").replace(")", "_").replace("=", "_").replace("-", "_")
        cleaned_scores[clean_key] = v
    print(f"      ragas scores: {cleaned_scores}")
    return cleaned_scores, results_df


def run_abstention_evaluation(
    session_id: str,
    faiss_dir: str,
    provider: str,
    dataset: list,
    config: dict,
) -> tuple[dict, pd.DataFrame]:
    """
    Track B — No-answer / hallucination evaluation (no_answer questions only).

    One RAG object is created per call and reused across all questions.

    Returned dataframe uses the same column names as Track A so both
    tracks merge cleanly into one CSV:
        user_input, retrieved_contexts, response, reference,
        question_type, correctly_abstained
    RAGAS metric columns will be NaN for these rows.

    Metrics:
        abstention_accuracy  = correct_refusals / total_no_answer_questions
        hallucination_rate   = 1 - abstention_accuracy
    Returns (scores_dict, results_dataframe).
    """
    rag = EvalConversationalRAG(session_id=session_id, provider_override=provider)
    rag.load_retriever_from_faiss(
        index_path=f"{faiss_dir}/{session_id}",
        k=config.get("k", 5),
        search_type=config.get("search_type", "similarity"),
        fetch_k=config.get("fetch_k", 20),
        lambda_mult=config.get("lambda_mult", 0.5),
    )

    rows = []

    for i, example in enumerate(dataset):
        print(f"      [{i+1}/{len(dataset)}] [no_answer] {example['question'][:60]}...")

        result  = rag.invoke_with_context(user_input=example["question"], chat_history=[])
        refused = _is_refusal(result["answer"])
        time.sleep(2)

        rows.append({
            "user_input":          example["question"],
            "retrieved_contexts":  str(result["contexts"]),
            "response":            result["answer"],
            "reference":           example["answer"],
            "question_type":       "no_answer",
            "correctly_abstained": int(refused),
        })

    results_df = pd.DataFrame(rows)

    total               = len(results_df)
    correct_refusals    = results_df["correctly_abstained"].sum()
    abstention_accuracy = correct_refusals / total if total > 0 else 0.0
    hallucination_rate  = 1.0 - abstention_accuracy

    scores = {
        "abstention_accuracy": round(abstention_accuracy, 4),
        "hallucination_rate":  round(hallucination_rate, 4),
    }
    print(f"      abstention scores: {scores}")
    return scores, results_df
