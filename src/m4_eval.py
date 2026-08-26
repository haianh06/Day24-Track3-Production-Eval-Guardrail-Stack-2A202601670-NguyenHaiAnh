from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


import types
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _m = types.ModuleType("langchain_community.chat_models.vertexai")
    _m.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _m


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    from config import GOOGLE_API_KEY, OPENAI_API_KEY, USE_GEMINI, LLM_BASE_URL, LLM_MODEL
    key = GOOGLE_API_KEY or OPENAI_API_KEY
    if not key:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from langchain_community.embeddings import HuggingFaceEmbeddings

        if USE_GEMINI or (GOOGLE_API_KEY and not OPENAI_API_KEY):
            llm = ChatOpenAI(
                base_url=LLM_BASE_URL or "https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=GOOGLE_API_KEY,
                model=LLM_MODEL or "gemini-flash-latest",
                max_retries=5
            )
            answer_relevancy.strictness = 1
        else:
            llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", max_retries=5)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings
        )
        df = result.to_pandas()
        per_question = []
        for _, row in df.iterrows():
            per_question.append(EvalResult(
                question=str(row.get("question", "")),
                answer=str(row.get("answer", "")),
                contexts=list(row.get("contexts", [])),
                ground_truth=str(row.get("ground_truth", "")),
                faithfulness=float(row.get("faithfulness", 0.0) if str(row.get("faithfulness")) != "nan" and row.get("faithfulness") is not None else 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) if str(row.get("answer_relevancy")) != "nan" and row.get("answer_relevancy") is not None else 0.0),
                context_precision=float(row.get("context_precision", 0.0) if str(row.get("context_precision")) != "nan" and row.get("context_precision") is not None else 0.0),
                context_recall=float(row.get("context_recall", 0.0) if str(row.get("context_recall")) != "nan" and row.get("context_recall") is not None else 0.0),
            ))

        def _safe_mean(vals):
            valid = [v for v in vals if not (v is None or str(v) == "nan")]
            return round(sum(valid) / len(valid), 4) if valid else 0.0

        f_val = result.get("faithfulness")
        ar_val = result.get("answer_relevancy")
        cp_val = result.get("context_precision")
        cr_val = result.get("context_recall")

        faith_score = float(f_val) if f_val is not None and str(f_val) != "nan" else _safe_mean([r.faithfulness for r in per_question])
        ar_score = float(ar_val) if ar_val is not None and str(ar_val) != "nan" else _safe_mean([r.answer_relevancy for r in per_question])
        cp_score = float(cp_val) if cp_val is not None and str(cp_val) != "nan" else _safe_mean([r.context_precision for r in per_question])
        cr_score = float(cr_val) if cr_val is not None and str(cr_val) != "nan" else _safe_mean([r.context_recall for r in per_question])

        return {
            "faithfulness": faith_score,
            "answer_relevancy": ar_score,
            "context_precision": cp_score,
            "context_recall": cr_score,
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating — generated answer not fully supported by context", "Tighten prompt, lower temperature, add strict grounding instructions"),
        "context_recall": ("Missing relevant chunks — retriever failed to retrieve necessary context", "Improve chunking strategy or add BM25/keyword indexing"),
        "context_precision": ("Too many irrelevant chunks — top retrieved chunks contain noise", "Add cross-encoder reranking or metadata pre-filtering"),
        "answer_relevancy": ("Answer doesn't match question — response wandered or missed the question intent", "Improve system prompt template and few-shot formatting"),
    }

    scored_items = []
    for item in eval_results:
        metrics = {
            "faithfulness": item.faithfulness,
            "answer_relevancy": item.answer_relevancy,
            "context_precision": item.context_precision,
            "context_recall": item.context_recall,
        }
        avg_score = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        worst_score = metrics[worst_metric]
        diagnosis, suggested_fix = diagnostic_tree.get(
            worst_metric, ("General retrieval/generation error", "Tune pipeline hyperparameters")
        )
        scored_items.append({
            "question": item.question,
            "worst_metric": worst_metric,
            "score": round(worst_score, 4),
            "avg_score": round(avg_score, 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    scored_items.sort(key=lambda x: x["avg_score"])
    return scored_items[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
