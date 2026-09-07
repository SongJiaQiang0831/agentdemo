"""Lesson 09: an integrated persistent Agent with RAG and calculator tools."""

from __future__ import annotations

import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lessons/03_agent"))

from agent_chat import calculator, search_knowledge_base  # noqa: E402
from rag import _settings  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

DB_PATH = Path(__file__).with_name("assistant.sqlite")


def build_agent(checkpointer):
    key, base_url, _ = _settings()
    model = ChatOpenAI(model="qwen-plus", api_key=key, base_url=base_url, temperature=0.2)
    return create_agent(
        model=model,
        tools=[search_knowledge_base, calculator],
        system_prompt=(
            "你是个人学习助手。涉及学习笔记的问题必须调用 search_knowledge_base；"
            "涉及计算的问题必须调用 calculator；普通聊天可以直接回答。"
            "回答要简洁，并说明使用了哪些工具。若工具被用户拒绝，"
            "必须说明操作未执行，不得自行计算或猜测结果。"
        ),
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "calculator": {"allowed_decisions": ["approve", "reject"]},
                    "search_knowledge_base": False,
                },
                description_prefix="计算工具执行前需要用户确认",
            )
        ],
        checkpointer=checkpointer,
    )


def invoke_with_approval(agent, question: str, config: dict):
    result = agent.invoke({"messages": [("user", question)]}, config=config)
    while result.get("__interrupt__"):
        request = result["__interrupt__"][0].value
        decisions = []
        for action in request["action_requests"]:
            print(f"\n[等待审批] 工具：{action['name']}")
            print(f"参数：{action['args']}")
            approved = input("是否允许执行？(y/n)：").strip().lower() in {"y", "yes"}
            if approved:
                decisions.append({"type": "approve"})
            else:
                decisions.append({"type": "reject", "message": "用户拒绝执行该工具。"})
        result = agent.invoke(Command(resume={"decisions": decisions}), config=config)
    return result


def main() -> None:
    thread_id = input("会话编号（直接回车使用 default）：").strip() or "default"
    print("整合助手已启动，输入 quit 或 exit 退出。")
    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        agent = build_agent(checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        while True:
            question = input("\n你：").strip()
            if question.lower() in {"quit", "exit"}:
                print("再见！")
                return
            if not question:
                continue
            result = invoke_with_approval(agent, question, config)
            for message in result["messages"]:
                for call in getattr(message, "tool_calls", []):
                    print(f"[工具请求] {call['name']}({call['args']})")
            print(f"\n助手：{result['messages'][-1].content}")


if __name__ == "__main__":
    main()
