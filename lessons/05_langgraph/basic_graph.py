"""Lesson 05: represent a RAG workflow with an explicit LangGraph state graph."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag import answer, retrieve_relevant  # noqa: E402


class RAGState(TypedDict):
    question: str
    documents: list
    answer: str


def retrieve_node(state: RAGState) -> dict:
    """Node 1: retrieve relevant documents and write them into state."""
    return {"documents": retrieve_relevant(state["question"])}


def answer_node(state: RAGState) -> dict:
    """Node 2: generate an answer from the question and retrieved documents."""
    return {"answer": answer(state["question"], state["documents"])}


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def main() -> None:
    question = "15+5等于多少"
    # question = "RAG 的基本流程是什么？"
    result = build_graph().invoke({"question": question, "documents": [], "answer": ""})
    print(f"问题：{question}")
    print(f"检索到的片段数：{len(result['documents'])}")
    print(f"\n助手：{result['answer']}")


if __name__ == "__main__":
    main()
