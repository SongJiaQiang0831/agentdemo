"""Lesson 14: stream the final answer from the LCEL RAG chain."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag import create_rag_chain, retrieve_with_scores  # noqa: E402


def main() -> None:
    question = input("问题：").strip() or "RAG 的基本流程是什么？"
    scored_documents = retrieve_with_scores(question)
    print("\n检索来源：")
    for document, score in scored_documents:
        print(f"- {document.metadata.get('source', '未知')} | 距离：{score:.4f}")

    print("\n助手：", end="", flush=True)
    chain = create_rag_chain()
    for chunk in chain.stream({"question": question, "chat_history": []}):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
