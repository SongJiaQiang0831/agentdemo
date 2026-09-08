"""Lesson 17: benchmark retrieval with similar documents and ranked metrics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag import retrieve_with_scores  # noqa: E402

HYBRID_PATH = ROOT / "lessons/16_hybrid_retrieval/hybrid_rag.py"
spec = importlib.util.spec_from_file_location("hybrid_rag", HYBRID_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"无法加载混合检索模块：{HYBRID_PATH}")
hybrid_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hybrid_module)
hybrid_retrieve = hybrid_module.hybrid_retrieve

DATASET = [
    ("RAG 的基本流程是什么？", "rag_intro.md"),
    ("Recall@K 用来评估什么？", "rag_evaluation.md"),
    ("混合检索为什么还需要重排序？", "vector_search.md"),
    ("Agent 如何处理临时网络错误？", "agent_workflow.md"),
    ("Python 如何定义函数？", "python_basics.md"),
]


def evaluate(name: str, retriever, k: int) -> float:
    hits = 0
    print(f"\n{name} Recall@{k}")
    for question, expected in DATASET:
        results = retriever(question, k=k)
        sources = [Path(doc.metadata.get("source", "")).name for doc, _ in results]
        rank = sources.index(expected) + 1 if expected in sources else None
        hits += rank is not None
        print(f"- {'命中' if rank else '未命中'} | 排名：{rank or '-'} | {question} | {sources}")
    score = hits / len(DATASET)
    print(f"总分：{hits}/{len(DATASET)} = {score:.1%}")
    return score


def main() -> None:
    for k in (1, 2):
        evaluate("向量检索", retrieve_with_scores, k)
        evaluate("混合检索 + 重排序", hybrid_retrieve, k)


if __name__ == "__main__":
    main()
