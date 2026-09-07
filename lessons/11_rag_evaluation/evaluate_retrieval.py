"""Lesson 11: evaluate whether retrieval returns the expected source files."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag import retrieve_with_scores  # noqa: E402

DATASET = [
    {
        "question": "RAG 的基本流程是什么？",
        "expected_source": "rag_intro.md",
    },
    {
        "question": "Python 如何定义函数？",
        "expected_source": "python_basics.md",
    },
    {
        "question": "Agent 为什么需要工具？",
        "expected_source": "agent_intro.md",
    },
]


def evaluate_at_k(k: int) -> None:
    hits = 0
    print(f"Recall@{k} 检索评估：")
    for item in DATASET:
        results = retrieve_with_scores(item["question"], k=k)
        sources = [Path(doc.metadata.get("source", "")).name for doc, _ in results]
        hit = item["expected_source"] in sources
        hits += int(hit)
        print(f"- {'通过' if hit else '失败'} | {item['question']}")
        print(f"  期望：{item['expected_source']} | 实际：{sources}")
    score = hits / len(DATASET)
    print(f"\nRecall@{k} = {hits}/{len(DATASET)} = {score:.1%}")


if __name__ == "__main__":
    evaluate_at_k(1)
    print()
    evaluate_at_k(2)
