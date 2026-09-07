"""Lesson 08: pause a LangGraph workflow until a human approves an action."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lessons/03_agent"))

from agent_chat import calculator  # noqa: E402


class ApprovalState(TypedDict):
    expression: str
    approved: bool
    result: str


def approval_node(state: ApprovalState) -> dict:
    decision = interrupt({
        "message": "即将执行计算，请确认。",
        "expression": state["expression"],
    })
    return {"approved": decision is True}


def calculate_node(state: ApprovalState) -> dict:
    if not state["approved"]:
        return {"result": "用户拒绝了操作，未执行计算。"}
    return {"result": calculator.invoke({"expression": state["expression"]})}


def build_graph():
    graph = StateGraph(ApprovalState)
    graph.add_node("approval", approval_node)
    graph.add_node("calculate", calculate_node)
    graph.add_edge(START, "approval")
    graph.add_edge("approval", "calculate")
    graph.add_edge("calculate", END)
    return graph.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "approval-demo"}}
    first = graph.invoke(
        {"expression": "18 * 7 + 2", "approved": False, "result": ""},
        config=config,
    )
    request = first["__interrupt__"][0].value
    print(f"暂停：{request['message']}")
    print(f"待执行表达式：{request['expression']}")
    user_input = input("是否允许执行？(y/n)：").strip().lower()
    approved = user_input in {"y", "yes"}
    final = graph.invoke(Command(resume=approved), config=config)
    print(f"结果：{final['result']}")


if __name__ == "__main__":
    main()
