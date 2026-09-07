"""Lesson 13: stream Qwen output as it is generated."""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag import _settings  # noqa: E402


def create_model() -> ChatOpenAI:
    key, base_url, _ = _settings()
    return ChatOpenAI(
        model="qwen-plus",
        api_key=key,
        base_url=base_url,
        temperature=0.4,
    )


def main() -> None:
    model = create_model()
    question = "用三点解释为什么 RAG 需要向量检索。"
    messages = [
        SystemMessage(content="你是清晰、简洁的 RAG 学习助手。"),
        HumanMessage(content=question),
    ]
    print(f"问题：{question}\n\n助手：", end="", flush=True)
    for chunk in model.stream(messages):
        print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
