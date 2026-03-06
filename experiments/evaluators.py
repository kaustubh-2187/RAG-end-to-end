from __future__ import annotations

import time
import pandas as pd
from dotenv import load_dotenv

from ragas import evaluate, EvaluationDataset, SingleTurnSample
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

load_dotenv()


def get_ragas_llm():
    llm = ModelLoader().load_llm(provider_override="google")
    return LangchainLLMWrapper(llm)


def get_ragas_embeddings():
    embeddings = ModelLoader().load_embeddings()
    return LangchainEmbeddingsWrapper(embeddings)


def compute_mrr(samples_meta: list[dict]) -> list[float]:
    """
    Compute MRR for each sample.

    For each question we have a reference answer and an ordered list of
    retrieved chunks.  A chunk is considered a "hit" if any meaningful
    keyword from the reference answer appears in it (case-insensitive).
    For no-answer questions (reference == "I don't know.") MRR is set to
    NaN because there is no positive context to rank.

    MRR per sample = 1 / rank_of_first_hit  (0.0 if no hit in top-k)
    """
    NO_ANSWER = "i don't know."
    mrr_scores = []

    for meta in samples_meta:
        reference = meta["reference"].strip()

        # No-answer questions: skip (NaN)
        if reference.lower() == NO_ANSWER:
            mrr_scores.append(float("nan"))
            continue

        # Build a small set of distinctive keywords from the reference
        # (words longer than 4 chars to avoid stop-word noise)
        ref_words = {w.lower() for w in reference.split() if len(w) > 4}

        hit_rank = None
        for rank, chunk in enumerate(meta["retrieved_contexts"], start=1):
            chunk_lower = chunk.lower()
            # A chunk is relevant if it contains at least 2 reference keywords
            matches = sum(1 for w in ref_words if w in chunk_lower)
            if matches >= 2:
                hit_rank = rank
                break

        mrr_scores.append(1.0 / hit_rank if hit_rank else 0.0)

    return mrr_scores


def run_ragas_evaluation(
    session_id: str,
    faiss_dir: str,
    provider: str,
    dataset: list,
    config_name: str,
) -> tuple:
    """
    Runs EvalConversationalRAG on every question in the dataset,
    builds RAGAS SingleTurnSamples, scores with 6 metrics + MRR.
    Returns (scores_dict, results_dataframe).
    """
    print(f"  Building RAGAS samples for: {config_name}")
    samples = []
    samples_meta = []   # keep raw data for MRR computation

    for i, example in enumerate(dataset):
        print(f"    [{i+1}/{len(dataset)}] {example['question'][:70]}...")

        rag = EvalConversationalRAG(
            session_id=session_id,
            provider_override=provider,
        )
        rag.load_retriever_from_faiss(
            index_path=f"{faiss_dir}/{session_id}",
        )

        result = rag.invoke_with_context(
            user_input=example["question"],
            chat_history=[],
        )

        time.sleep(2)  # avoid Groq rate limit (429)

        samples.append(
            SingleTurnSample(
                user_input=example["question"],
                response=result["answer"],
                retrieved_contexts=result["contexts"],
                reference=example["answer"],
            )
        )

        samples_meta.append({
            "reference": example["answer"],
            "retrieved_contexts": result["contexts"],
        })

    ragas_dataset = EvaluationDataset(samples=samples)
    judge_llm = get_ragas_llm()
    judge_embeddings = get_ragas_embeddings()

    ragas_result = evaluate(
        dataset=ragas_dataset,
        metrics=[
            LLMContextRecall(),
            LLMContextPrecisionWithReference(),
            Faithfulness(),
            AnswerRelevancy(),
            FactualCorrectness(),
            SemanticSimilarity(),
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    results_df = ragas_result.to_pandas()

    # Append MRR column — computed independently, not via RAGAS
    results_df["mrr"] = compute_mrr(samples_meta)

    scores = results_df.mean(numeric_only=True).to_dict()
    print(f"  Scores: {scores}")

    return scores, results_df
