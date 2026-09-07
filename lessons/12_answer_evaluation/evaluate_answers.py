"""Lesson 12: evaluate generated answers with expected facts and latency."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag import answer_from_knowledge  # noqa: E402

DATASET = [
    {
        "question": "RAG 的基本流程是什么？",
        "expected_facts": ["加载文档", "切分文本", "生成向量", "相似度检索"],
    },
    {
        "question": "Python 如何定义函数？",
        "expected_facts": ["def", "函数"],
    },
    {
        "question": "Agent 为什么需要工具？",
        "expected_facts": ["选择下一步行动", "调用外部工具"],
    },
]


def evaluate_answer(question: str, expected_facts: list[str]) -> dict:
    started = time.perf_counter()
    answer = answer_from_knowledge(question)
    latency = time.perf_counter() - started
    matched = [fact for fact in expected_facts if fact.lower() in answer.lower()]
    return {
        "answer": answer,
        "matched": matched,
        "score": len(matched) / len(expected_facts),
        "latency": latency,
    }


def main() -> None:
    total_score = 0.0
    total_latency = 0.0
    for item in DATASET:
        result = evaluate_answer(item["question"], item["expected_facts"])
        total_score += result["score"]
        total_latency += result["latency"]
        print(f"问题：{item['question']}")
        print(f"命中事实：{result['matched']}")
        print(f"关键词得分：{result['score']:.1%}")
        print(f"回答耗时：{result['latency']:.2f} 秒")
        print(f"回答：{result['answer']}\n")
    print(f"平均关键词得分：{total_score / len(DATASET):.1%}")
    print(f"平均回答耗时：{total_latency / len(DATASET):.2f} 秒")


if __name__ == "__main__":
    main()
