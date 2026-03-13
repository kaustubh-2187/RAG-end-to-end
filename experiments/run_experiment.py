from __future__ import annotations

import sys
import json
import csv
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv()

from configs import (
    CONFIGS,
    DATASET_PATH,
    DOCUMENT_PATH,
    FAISS_DIR,
    DATA_DIR,
    OUTPUT_DIR,
)
from ingest import build_index
from evaluators import run_ragas_evaluation

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUTPUT_DIR / "summary.csv"


def _load_summary() -> dict[str, dict]:
    """Load existing summary rows keyed by experiment name."""
    if not SUMMARY_PATH.exists():
        return {}
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["experiment"]: row for row in reader}


def _append_summary(row: dict, fieldnames: list[str]):
    """Append a single result row to the summary CSV (create header if needed)."""
    write_header = not SUMMARY_PATH.exists()
    with open(SUMMARY_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_all_experiments():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    existing = _load_summary()
    summary_fieldnames: list[str] = []

    for config in CONFIGS:
        config_name = config["name"]
        csv_path = OUTPUT_DIR / f"{config_name}.csv"

        print(f"\n{'='*60}")
        print(f"Experiment : {config_name}")

        # ── Skip if already done ──────────────────────────────────
        if config_name in existing and csv_path.exists():
            print(f"  [SKIP] Results already exist → {csv_path}")
            print(f"{'='*60}")
            continue

        print(f"Config     : {config}")
        print(f"{'='*60}")

        # ── Step 1: Index ─────────────────────────────────────────
        print(f"  [1/3] Indexing — chunk_size={config['chunk_size']} overlap={config['chunk_overlap']}...")
        session_id = build_index(
            doc_path=DOCUMENT_PATH,
            config=config,
            faiss_dir=FAISS_DIR,
            data_dir=DATA_DIR,
        )
        print(f"  [1/3] Done. session_id={session_id}")

        # ── Step 2: Evaluate ──────────────────────────────────────
        print(f"  [2/3] Running RAGAS evaluation...")
        scores, results_df = run_ragas_evaluation(
            session_id=session_id,
            faiss_dir=FAISS_DIR,
            provider=config["provider"],
            dataset=dataset,
            config_name=config_name,
            config=config,
        )
        print(f"  [2/3] Done.")

        # ── Step 3: Save per-config CSV immediately ───────────────
        print(f"  [3/3] Saving results...")
        results_df.to_csv(csv_path, index=False)
        print(f"  [3/3] Per-config CSV saved → {csv_path}")

        # ── Append to rolling summary immediately ─────────────────
        row = {"experiment": config_name, **config, **scores}
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
