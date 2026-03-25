from __future__ import annotations

import sys
import json
import csv
import pandas as pd
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()

from configs import (
    CONFIGS,
    DATASET_PATH,
    DATA_DIR,
    FAISS_DIR,
    OUTPUT_DIR,
    DOCUMENT_INDEX_MAP,
    EVALUATION_METHOD,
    METHOD2_DOCUMENT_INDEX,
    ACTIVE_DOCUMENTS,
)
from ingest import build_index, teardown_index
from evaluators import run_ragas_evaluation, run_abstention_evaluation, NON_METRIC_COLS

# MLflow imports
import mlflow
import mlflow.sklearn

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"


def _load_summary() -> dict[str, dict]:
    if not SUMMARY_PATH.exists():
        return {}
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["experiment"]: row for row in reader}


def _append_summary(row: dict, fieldnames: list[str]):
    write_header = not SUMMARY_PATH.exists()
    with open(SUMMARY_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _group_by_document(dataset: list[dict]) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for entry in dataset:
        groups[entry["document_index"]].append(entry)
    return dict(groups)


def run_all_experiments():
    # Set up MLflow experiment
    experiment_name = f"RAG_{EVALUATION_METHOD}"
    mlflow.set_experiment(experiment_name)
    
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    doc_groups = _group_by_document(dataset)

    # Determine which documents to evaluate:
    # method2  → always the single document from METHOD2_DOCUMENT_INDEX
    # method1 + ACTIVE_DOCUMENTS set → only the listed indices
    # method1 + ACTIVE_DOCUMENTS = None → all documents in the dataset
    if EVALUATION_METHOD == "method2":
        doc_indices = [METHOD2_DOCUMENT_INDEX]
    elif ACTIVE_DOCUMENTS is not None:
        doc_indices = sorted(ACTIVE_DOCUMENTS)
    else:
        doc_indices = sorted(doc_groups.keys())

    print(f"Evaluation method : {EVALUATION_METHOD}")
    print(f"Dataset           : {DATASET_PATH}")
    print(f"Output directory  : {OUTPUT_DIR}")
    print(f"Dataset loaded    : {len(dataset)} total questions across {len(doc_groups)} documents")
    print(f"Documents to run  : {doc_indices}")
    for idx in doc_indices:
        entries = doc_groups.get(idx, [])
        qtypes  = defaultdict(int)
        for e in entries:
            qtypes[e["question_type"]] += 1
        print(f"  doc {idx:>2}: {len(entries):>3} questions  {dict(qtypes)}")

    existing           = _load_summary()
    summary_fieldnames: list[str] = []

    for config in CONFIGS:
        config_name = config["name"]
        csv_path    = OUTPUT_DIR / f"{config_name}.csv"

        print(f"\n{'='*60}")
        print(f"Experiment : {config_name}")

        if config_name in existing and csv_path.exists():
            print(f"  [SKIP] Already complete → {csv_path}")
            continue

        # Start MLflow run for this configuration
        with mlflow.start_run(run_name=config_name):
            print(f"Config     : {config}")
            print(f"{'='*60}")

            ragas_results      = []
            abstention_results = []

            for doc_idx in doc_indices:
                doc_questions = doc_groups.get(doc_idx, [])
                if not doc_questions:
                    print(f"  [WARN] No questions found for doc_index={doc_idx}, skipping.")
                    continue

                txt_filename = DOCUMENT_INDEX_MAP.get(doc_idx)
                if not txt_filename:
                    print(f"  [WARN] No txt file mapped for doc_index={doc_idx}, skipping.")
                    continue

                doc_path = DATA_DIR / txt_filename
                if not doc_path.exists():
                    print(f"  [WARN] File not found: {doc_path}, skipping doc {doc_idx}.")
                    continue

                answerable = [q for q in doc_questions if q["question_type"] != "no_answer"]
                no_answer  = [q for q in doc_questions if q["question_type"] == "no_answer"]

                print(f"\n  ── Document {doc_idx:>2}: {txt_filename} ──")

                # ── Step 1: Build index ───────────────────────────────
                print(f"    [1/3] Indexing...")
                session_id = build_index(
                    doc_path=str(doc_path),
                    config=config,
                    faiss_dir=str(FAISS_DIR),
                )

                # ── Step 2A: Track A — answerable questions ───────────
                if answerable:
                    print(f"    [2/3] Track A — {len(answerable)} answerable questions...")
                    _, ragas_df = run_ragas_evaluation(
                        session_id=session_id,
                        faiss_dir=str(FAISS_DIR),
                        provider=config["provider"],
                        dataset=answerable,
                        config=config,
                    )
                    ragas_df["document_index"] = doc_idx
                    ragas_results.append(ragas_df)

                # ── Step 2B: Track B — no_answer questions ────────────
                if no_answer:
                    print(f"    [2/3] Track B — {len(no_answer)} no_answer questions...")
                    _, abstention_df = run_abstention_evaluation(
                        session_id=session_id,
                        faiss_dir=str(FAISS_DIR),
                        provider=config["provider"],
                        dataset=no_answer,
                        config=config,
                    )
                    abstention_df["document_index"] = doc_idx
                    abstention_results.append(abstention_df)

                # ── Step 3: Tear down index ───────────────────────────
                print(f"    [3/3] Tearing down index...")
                teardown_index(session_id, str(FAISS_DIR))

            if not ragas_results and not abstention_results:
                print(f"  [WARN] No results collected for {config_name}. Skipping save.")
                continue

            # ── Merge both tracks into one CSV ────────────────────────
            # Track A (answerable) rows: RAGAS metric cols populated, correctly_abstained = NaN
            # Track B (no_answer) rows:  RAGAS metric cols = NaN, correctly_abstained populated
            ragas_combined = pd.concat(ragas_results, ignore_index=True) if ragas_results else pd.DataFrame()
            abst_combined  = pd.concat(abstention_results, ignore_index=True) if abstention_results else pd.DataFrame()
            combined_df    = pd.concat([ragas_combined, abst_combined], ignore_index=True)

            combined_df.to_csv(csv_path, index=False)
            print(f"\n  Saved: {config_name}.csv ({len(combined_df)} rows)")

            # ── Compute scores — exclude non-metric identifier columns ─
            ragas_scores    = {}
            category_scores = {}

            if not ragas_combined.empty:
                metric_cols  = [
                    c for c in ragas_combined.select_dtypes(include="number").columns
                    if c not in NON_METRIC_COLS
                ]
                ragas_scores = ragas_combined[metric_cols].mean().to_dict()

                for qtype in ["single_passage", "multi_passage"]:
                    if "question_type" in ragas_combined.columns:
                        subset = ragas_combined[ragas_combined["question_type"] == qtype]
                        if not subset.empty:
                            for col in metric_cols:
                                category_scores[f"{qtype}__{col}"] = subset[col].mean()

            abstention_scores = {}
            if not abst_combined.empty:
                total               = len(abst_combined)
                correct             = abst_combined["correctly_abstained"].sum()
                abstention_accuracy = round(correct / total, 4)
                abstention_scores   = {
                    "abstention_accuracy": abstention_accuracy,
                    "hallucination_rate":  round(1.0 - abstention_accuracy, 4),
                }

            # ── Print summary ─────────────────────────────────────────
            print(f"\n  Track A — RAGAS scores (answerable):")
            for k, v in ragas_scores.items():
                print(f"    {k}: {v:.4f}")

            print(f"\n  Track A — Category breakdown:")
            for k, v in category_scores.items():
                print(f"    {k}: {v:.4f}")

            print(f"\n  Track B — Abstention scores (no_answer):")
            for k, v in abstention_scores.items():
                print(f"    {k}: {v:.4f}")

            # ── Log parameters to MLflow ──────────────────────────────
            # Configuration parameters
            mlflow.log_param("config_name", config_name)
            mlflow.log_param("chunk_size", config.get("chunk_size", 1000))
            mlflow.log_param("chunk_overlap", config.get("chunk_overlap", 200))
            mlflow.log_param("k", config.get("k", 5))
            mlflow.log_param("search_type", config.get("search_type", "similarity"))
            mlflow.log_param("fetch_k", config.get("fetch_k", 20))
            mlflow.log_param("lambda_mult", config.get("lambda_mult", 0.5))
            mlflow.log_param("provider", config.get("provider", "openai"))
            
            # Evaluation metadata
            mlflow.log_param("evaluation_method", EVALUATION_METHOD)
            mlflow.log_param("judge_provider", "openai")  # From configs.py

            # ── Log metrics to MLflow ─────────────────────────────────
            # Clean metric names for MLflow compatibility (replace invalid characters)
            def clean_metric_name(name):
                return name.replace("(", "_").replace(")", "_").replace("=", "_").replace("-", "_")
            
            # RAGAS scores
            for k, v in ragas_scores.items():
                clean_k = clean_metric_name(k)
                mlflow.log_metric(clean_k, v)
            
            # Category breakdown scores
            for k, v in category_scores.items():
                clean_k = clean_metric_name(k)
                mlflow.log_metric(clean_k, v)
            
            # Abstention scores
            for k, v in abstention_scores.items():
                clean_k = clean_metric_name(k)
                mlflow.log_metric(clean_k, v)

            # ── Log artifacts to MLflow ───────────────────────────────
            mlflow.log_artifact(str(csv_path))
            
            # Log summary.csv if it exists
            if SUMMARY_PATH.exists():
                mlflow.log_artifact(str(SUMMARY_PATH))

            # ── Append to summary.csv ─────────────────────────────────
            row = {
                "experiment":        config_name,
                "evaluation_method": EVALUATION_METHOD,
                **config,
                **ragas_scores,
                **category_scores,
                **abstention_scores,
            }
            if not summary_fieldnames:
                summary_fieldnames = list(row.keys())
            _append_summary(row, summary_fieldnames)
            print(f"  Summary updated → {SUMMARY_PATH}")

    print(f"\n{'='*60}")
    print(f"All experiments complete.")
    print(f"Summary   : {SUMMARY_PATH}")
    print(f"Per-config: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_all_experiments()
