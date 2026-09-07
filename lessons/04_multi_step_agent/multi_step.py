"""Lesson 04: an Agent that can call multiple tools for one task."""

from __future__ import annotations

import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lessons/03_agent"))

from agent_chat import calculator, search_knowledge_base  # noqa: E402
from rag import _settings  # noqa: E402


def main() -> None:
    key, base_url, _ = _settings()
    model = ChatOpenAI(model="qwen-plus", api_key=key, base_url=base_url, temperature=0.2)
    agent = create_agent(
        model=model,
        tools=[search_knowledge_base, calculator],
        system_prompt=(
            "你是学习计划助手。遇到学习笔记问题，必须先调用 search_knowledge_base；"
            "遇到数学计算，必须调用 calculator。若一个任务同时需要资料和计算，"
            "先搜索，再根据搜索结果计算，最后解释完整过程。"
        ),
    )
    question = "根据知识库中的 RAG 学习内容，如果我每天学习 2 小时，连续学习 7 天一共是多少小时？"
    result = agent.invoke({"messages": [("user", question)]})
    for message in result["messages"]:
        if getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                print(f"[工具调用] {call['name']}({call['args']})")
    print(f"\n问题：{question}\n\n助手：{result['messages'][-1].content}")


if __name__ == "__main__":
    main()
