"""
scripts/evaluate.py

Evaluates Project Aegis end-to-end using the RAGAS framework.

Metrics:
  faithfulness      — answers are grounded in the retrieved context
  answer_relevancy  — the answer actually addresses the question
  context_precision — retrieved chunks are relevant to the question
  context_recall    — all relevant information was retrieved

Input:  JSON file of {question, ground_truth} pairs (see --qa_file)
Output: CSV of per-question scores + printed summary statistics

Usage:
    python scripts/evaluate.py --qa_file data/eval_qa.json

Sample eval_qa.json:
    [
      {"question": "What is the per diem for USA travel?",
       "ground_truth": "$120 per day"},
      {"question": "How many weeks of maternity leave are employees entitled to?",
       "ground_truth": "26 weeks"}
    ]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset
from loguru import logger
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app.generator import generate_answer
from retrieval.pipeline import retrieve


# ── Dataset builder ──────────────────────────────────────────────────────────

def build_eval_dataset(qa_pairs: list[dict]) -> Dataset:
    """
    Run the full RAG pipeline on each QA pair and collect:
      question, answer, contexts, ground_truth
    """
    rows: list[dict] = []

    for item in qa_pairs:
        question     = item["question"]
        ground_truth = item.get("ground_truth", "")

        chunks  = retrieve(question)
        result  = generate_answer(query=question, chunks=chunks)

        rows.append({
            "question":     question,
            "answer":       result["answer"],
            "contexts":     [c["chunk_text"] for c in chunks],
            "ground_truth": ground_truth,
        })
        logger.info(f"Evaluated: {question[:60]}")

    return Dataset.from_list(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Project Aegis with RAGAS"
    )
    parser.add_argument(
        "--qa_file", required=True,
        help="Path to JSON file with [{question, ground_truth}, ...]"
    )
    parser.add_argument(
        "--output", default="evaluation_results.csv",
        help="Output CSV path (default: evaluation_results.csv)"
    )
    args = parser.parse_args()

    qa_path = Path(args.qa_file)
    if not qa_path.exists():
        raise SystemExit(f"QA file not found: {qa_path}")

    qa_pairs = json.loads(qa_path.read_text())
    logger.info(f"Running evaluation on {len(qa_pairs)} question(s)...")

    dataset = build_eval_dataset(qa_pairs)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    df: pd.DataFrame = result.to_pandas()
    df.to_csv(args.output, index=False)
    logger.success(f"Results saved to {args.output}")

    print("\n── Evaluation Summary ────────────────────────────────────────────")
    summary_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    available = [c for c in summary_cols if c in df.columns]
    print(df[available].describe().round(3).to_string())
    print("──────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
