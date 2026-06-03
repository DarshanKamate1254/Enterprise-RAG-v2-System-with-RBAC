"""
RAGAS Evaluation Script for the LlamaIndex RAG v2 Pipeline.

Compatible with ragas>=0.4.3 (modern API).

Evaluates:
  - faithfulness       — answer grounded in retrieved context
  - answer_relevancy   — answer addresses the question
  - context_precision  — retrieved context is relevant
  - context_recall     — retrieved context covers the ground truth

Usage:
    python eval/evaluate.py

Requires: OPENAI_API_KEY set in .env
Output:   eval/ragas_results.json  and  eval/ragas_results.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from app.rag.pipeline import run_chat
from app.utils.config import get_settings

# ── Evaluation dataset ────────────────────────────────────────────────────────
EVAL_DATASET = [
    {
        "question": "What is the leave policy for employees?",
        "namespace": "hr",
        "user": {"role": "hr", "default_namespace": "hr"},
        "ground_truth": "Employees are entitled to leave as defined by the HR policy.",
    },
    {
        "question": "How many sick days do employees get per year?",
        "namespace": "hr",
        "user": {"role": "hr", "default_namespace": "hr"},
        "ground_truth": "The HR policy specifies the number of sick days allocated annually.",
    },
    {
        "question": "What was the quarterly revenue in Q3?",
        "namespace": "finance",
        "user": {"role": "finance", "default_namespace": "finance"},
        "ground_truth": "The quarterly financial report contains Q3 revenue figures.",
    },
    {
        "question": "Summarise the key financial highlights.",
        "namespace": "finance",
        "user": {"role": "finance", "default_namespace": "finance"},
        "ground_truth": "The financial summary document outlines the key financial highlights.",
    },
    {
        "question": "What were the main marketing campaigns in 2024?",
        "namespace": "marketing",
        "user": {"role": "marketing", "default_namespace": "marketing"},
        "ground_truth": "The 2024 marketing report describes the main campaigns.",
    },
    {
        "question": "What technologies does the engineering team use?",
        "namespace": "engineering",
        "user": {"role": "engineering", "default_namespace": "engineering"},
        "ground_truth": "The engineering master document lists the technologies in use.",
    },
    {
        "question": "What are the company's core values?",
        "namespace": "general",
        "user": {"role": "employee", "default_namespace": "general"},
        "ground_truth": "The employee handbook describes the company's core values.",
    },
]


def run_evaluation():
    get_settings()  # validates .env is loaded
    print("=" * 60)
    print("  RAGAS Evaluation — LlamaIndex RAG v2 Pipeline")
    print("=" * 60)

    # ── Step 1: Collect answers from the RAG pipeline ─────────────────────────
    samples = []
    for i, item in enumerate(EVAL_DATASET, 1):
        print(f"\n[{i}/{len(EVAL_DATASET)}] {item['question'][:60]}")
        try:
            result = run_chat(
                question=item["question"],
                namespace=item["namespace"],
                user=item["user"],
            )
            samples.append({
                "user_input":         item["question"],
                "response":           result["answer"],
                "retrieved_contexts": result["contexts"] if result["contexts"] else [""],
                "reference":          item["ground_truth"],
            })
            print(f"  ✓ {result['answer'][:80]}")
        except Exception as exc:
            print(f"  ✗ Error: {exc}")
            samples.append({
                "user_input":         item["question"],
                "response":           f"Error: {exc}",
                "retrieved_contexts": [""],
                "reference":          item["ground_truth"],
            })

    # ── Step 2: Run RAGAS ─────────────────────────────────────────────────────
    print("\n\nRunning RAGAS metrics ...")

    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    # ── Step 3: Save results ──────────────────────────────────────────────────
    out_dir = Path("eval")
    out_dir.mkdir(exist_ok=True)

    df = result.to_pandas()
    csv_path  = out_dir / "ragas_results.csv"
    json_path = out_dir / "ragas_results.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print("\n" + "=" * 60)
    print("  RAGAS Results Summary")
    print("=" * 60)
    for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if col in df.columns:
            print(f"  {col:<25} {df[col].mean():.4f}")

    print(f"\n  Results saved to:")
    print(f"    {json_path}")
    print(f"    {csv_path}")
    return result


if __name__ == "__main__":
    run_evaluation()