"""Lesson 19: connect retrieval quality with answer quality and latency."""

from __future__ import annotations

import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag import answer, retrieve_relevant, retrieve_with_scores  # noqa: E402


DATASET = [
    {
        "question": "怎样让模型先查外部资料，再结合资料生成答案？",
        "expected_source": "rag_intro.md",
        "expected_facts": ["检索", "文本片段", "生成答案"],
    },
    {
        "question": "我想检查搜索模块有没有把正确资料放在靠前位置，该看哪些指标？",
        "expected_source": "rag_evaluation.md",
        "expected_facts": ["Recall@K", "MRR", "检索评估"],
    },
    {
        "question": "任务步骤会根据工具返回内容变化，应该使用固定链还是可变工作流？",
        "expected_source": "agent_workflow.md",
        "expected_facts": ["Chain", "Agent", "LangGraph"],
    },
]


def evaluate_item(item: dict[str, object]) -> dict[str, object]:
    question = str(item["question"])
    expected_source = str(item["expected_source"])
    expected_facts = [str(fact) for fact in item["expected_facts"]]
    started = time.perf_counter()
    scored = retrieve_with_scores(question, k=4)
    documents = retrieve_relevant(question, k=4)
    retrieval_sources = [Path(doc.metadata.get("source", "")).name for doc, _ in scored]
    generated = answer(question, documents)
    elapsed = time.perf_counter() - started
    matched = [fact for fact in expected_facts if fact.lower() in generated.lower()]
    return {
        "question": question,
        "retrieval_rank": retrieval_sources.index(expected_source) + 1
        if expected_source in retrieval_sources
        else None,
        "answer_score": len(matched) / len(expected_facts),
        "matched": matched,
        "latency": elapsed,
        "answer": generated,
    }


def main() -> None:
    results = [evaluate_item(item) for item in DATASET]
    print("RAG 端到端质量报告")
    for result in results:
        print(f"\n问题：{result['question']}")
        print(f"正确来源排名：{result['retrieval_rank'] or '-'}")
        print(f"事实覆盖率：{result['answer_score']:.1%} | 命中：{result['matched']}")
        print(f"延迟：{result['latency']:.2f} 秒")
        print(f"回答：{result['answer']}")
    average = sum(float(result["answer_score"]) for result in results) / len(results)
    latency = sum(float(result["latency"]) for result in results) / len(results)
    print(f"\n平均事实覆盖率：{average:.1%}")
    print(f"平均端到端延迟：{latency:.2f} 秒")


if __name__ == "__main__":
    main()
