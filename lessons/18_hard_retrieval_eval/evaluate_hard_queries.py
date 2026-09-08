"""Lesson 18: evaluate rewritten and ambiguous queries with Recall and MRR."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

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

Retriever = Callable[[str, int], list[tuple[Document, float]]]

DATASET = [
    {
        "type": "同义改写",
        "question": "怎样让模型先查外部资料，再结合资料生成答案？",
        "expected": "rag_intro.md",
    },
    {
        "type": "隐含表达",
        "question": "我想检查搜索模块有没有把正确资料放在靠前位置，该看哪些指标？",
        "expected": "rag_evaluation.md",
    },
    {
        "type": "同义改写",
        "question": "用户换了一种说法，但含义相同，系统该怎样找到相关片段？",
        "expected": "vector_search.md",
    },
    {
        "type": "相邻主题",
        "question": "任务步骤会根据工具返回内容变化，应该使用固定链还是可变工作流？",
        "expected": "agent_workflow.md",
    },
    {
        "type": "事实定位",
        "question": "哪种编程语言用缩进划分代码块，并且用 def 声明函数？",
        "expected": "python_basics.md",
    },
    {
        "type": "概念辨析",
        "question": "什么组件能够自己决定下一步行动，并选择外部能力完成任务？",
        "expected": "agent_intro.md",
    },
]


def source_names(results: list[tuple[Document, float]]) -> list[str]:
    return [Path(doc.metadata.get("source", "")).name for doc, _ in results]


def evaluate(name: str, retriever: Retriever) -> dict[str, float]:
    recall_1_hits = 0
    recall_2_hits = 0
    reciprocal_rank_sum = 0.0
    print(f"\n{name}")
    for item in DATASET:
        sources = source_names(retriever(item["question"], k=6))
        rank = sources.index(item["expected"]) + 1 if item["expected"] in sources else None
        recall_1_hits += rank == 1
        recall_2_hits += rank is not None and rank <= 2
        reciprocal_rank_sum += 1 / rank if rank else 0.0
        print(
            f"- [{item['type']}] 排名：{rank or '-'} | {item['question']}\n"
            f"  期望：{item['expected']} | 前三名：{sources[:3]}"
        )

    count = len(DATASET)
    metrics = {
        "recall_at_1": recall_1_hits / count,
        "recall_at_2": recall_2_hits / count,
        "mrr": reciprocal_rank_sum / count,
    }
    print(
        f"指标：Recall@1={metrics['recall_at_1']:.1%} | "
        f"Recall@2={metrics['recall_at_2']:.1%} | MRR={metrics['mrr']:.3f}"
    )
    return metrics


def main() -> None:
    baseline = evaluate("向量检索", retrieve_with_scores)
    hybrid = evaluate("混合检索 + 重排序", hybrid_retrieve)
    print("\n混合检索相对向量检索的变化")
    for metric in ("recall_at_1", "recall_at_2", "mrr"):
        print(f"- {metric}: {hybrid[metric] - baseline[metric]:+.3f}")


if __name__ == "__main__":
    main()
