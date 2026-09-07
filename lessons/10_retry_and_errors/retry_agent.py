"""Lesson 10: retry transient tool failures and report permanent failures."""

from __future__ import annotations

import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
from langchain.tools import tool
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag import _settings  # noqa: E402

attempt_count = 0


@tool
def unstable_learning_service(topic: str) -> str:
    """Query a learning service that may temporarily fail."""
    global attempt_count
    attempt_count += 1
    print(f"[工具执行] 第 {attempt_count} 次尝试，主题：{topic}")
    if attempt_count < 3:
        raise ConnectionError("模拟的临时网络错误")
    return f"服务恢复成功：{topic} 的学习顺序是基础概念、动手实验、质量评估。"


def build_agent():
    key, base_url, _ = _settings()
    model = ChatOpenAI(
        model="qwen-plus",
        api_key=key,
        base_url=base_url,
        temperature=0.0,
    )
    return create_agent(
        model=model,
        tools=[unstable_learning_service],
        system_prompt=(
            "你是学习助手。用户询问学习顺序时必须调用 "
            "unstable_learning_service，并根据工具结果简洁回答。"
        ),
        middleware=[
            ToolRetryMiddleware(
                max_retries=2,
                tools=["unstable_learning_service"],
                retry_on=(ConnectionError, TimeoutError),
                initial_delay=0.1,
                backoff_factor=2.0,
                jitter=False,
                on_failure="continue",
            ),
            ModelRetryMiddleware(
                max_retries=2,
                retry_on=(ConnectionError, TimeoutError),
                initial_delay=0.5,
                backoff_factor=2.0,
                jitter=False,
                on_failure="continue",
            ),
        ],
    )


def main() -> None:
    result = build_agent().invoke({"messages": [("user", "RAG 应该按什么顺序学习？")]})
    print(f"\n助手：{result['messages'][-1].content}")


if __name__ == "__main__":
    main()
