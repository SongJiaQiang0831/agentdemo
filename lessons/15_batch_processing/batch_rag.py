"""Lesson 15: answer several knowledge-base questions with batch()."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag import create_rag_chain, get_vector_store  # noqa: E402

QUESTIONS = [
    "RAG 的基本流程是什么？",
    "Python 如何定义函数？",
    "Agent 为什么需要工具？",
]


def main() -> None:
    # Warm up Chroma once before worker threads start.
    get_vector_store()
    chain = create_rag_chain()
    inputs = [{"question": question, "chat_history": []} for question in QUESTIONS]
    started = time.perf_counter()
    answers = chain.batch(inputs, config={"max_concurrency": 3})
    elapsed = time.perf_counter() - started
    for question, answer in zip(QUESTIONS, answers):
        print(f"问题：{question}\n回答：{answer}\n")
    print(f"批量处理 {len(QUESTIONS)} 个问题耗时：{elapsed:.2f} 秒")
    print(f"吞吐量：{len(QUESTIONS) / elapsed:.2f} 个问题/秒")


if __name__ == "__main__":
    main()
