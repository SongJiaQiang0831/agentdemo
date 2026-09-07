"""Lesson 07: persist LangGraph conversation state in SQLite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag import _settings  # noqa: E402

DB_PATH = Path(__file__).with_name("conversation.sqlite")


def create_model() -> ChatOpenAI:
    key, base_url, _ = _settings()
    return ChatOpenAI(
        model="qwen-plus",
        api_key=key,
        base_url=base_url,
        temperature=0.2,
    )


def build_graph(checkpointer: SqliteSaver):
    model = create_model()

    def chat_node(state: MessagesState) -> dict:
        messages = [
            SystemMessage(content="你是学习助手。请利用对话历史简洁回答。"),
            *state["messages"],
        ]
        return {"messages": [model.invoke(messages)]}

    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile(checkpointer=checkpointer)


def ask(message: str, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        graph = build_graph(checkpointer)
        result = graph.invoke({"messages": [("user", message)]}, config=config)
    return str(result["messages"][-1].content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", default="demo-user")
    parser.add_argument("--message")
    args = parser.parse_args()

    if args.message:
        print(ask(args.message, args.thread))
        return

    print(f"持久化对话已启动，会话编号：{args.thread}")
    while True:
        message = input("\n你：").strip()
        if message.lower() in {"quit", "exit"}:
            print("再见！")
            return
        if message:
            print(f"\n助手：{ask(message, args.thread)}")


if __name__ == "__main__":
    main()
