"""Lesson 06: add a conditional branch for unsupported questions."""

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
    route: str
    answer: str


def retrieve_node(state: RAGState) -> dict:
    documents = retrieve_relevant(state["question"])
    return {"documents": documents, "route": "answer" if documents else "fallback"}


def route_after_retrieve(state: RAGState) -> str:
    """Conditional edge: choose answer only when relevant documents exist."""
    return state["route"]


def answer_node(state: RAGState) -> dict:
    return {"answer": answer(state["question"], state["documents"])}


def fallback_node(state: RAGState) -> dict:
    return {"answer": "知识库中没有足够相关的信息，暂时无法回答这个问题。"}


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("fallback", fallback_node)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"answer": "answer", "fallback": "fallback"},
    )
    graph.add_edge("answer", END)
    graph.add_edge("fallback", END)
    return graph.compile()


def run(question: str) -> None:
    result = build_graph().invoke({"question": question, "documents": [], "route": "", "answer": ""})
    print(f"问题：{question}")
    print(f"分支：{result['route']}")
    print(f"回答：{result['answer']}\n")


if __name__ == "__main__":
    run("RAG 的基本流程是什么？")
    run("今天北京天气怎么样？")
