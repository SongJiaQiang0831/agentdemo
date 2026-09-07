"""Lesson 03: an Agent that decides when to search the knowledge base."""

from __future__ import annotations

import sys
import ast
import operator
from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import tool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag import retrieve_relevant  # noqa: E402
from rag import _settings  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402


@tool
def search_knowledge_base(query: str) -> str:
    """Search the local learning notes and return relevant passages."""
    documents = retrieve_relevant(query)
    if not documents:
        return "知识库中没有足够相关的信息。"
    return "\n\n".join(
        f"来源：{doc.metadata.get('source', '未知')}\n{doc.page_content}"
        for doc in documents
    )


def _evaluate(node: ast.AST) -> float:
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in operators:
        return operators[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    raise ValueError("只支持数字和 +、-、*、/、%、** 运算")


@tool
def calculator(expression: str) -> str:
    """Safely calculate a basic arithmetic expression, such as 800 / 5."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_evaluate(tree.body))
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        return f"无法计算：{exc}"


def main() -> None:
    key, base_url, _ = _settings()
    model = ChatOpenAI(
        model="qwen-plus",
        api_key=key,
        base_url=base_url,
        temperature=0.2,
    )
    agent = create_agent(
        model=model,
        tools=[search_knowledge_base, calculator],
        system_prompt=(
            "你是学习助手。涉及用户学习笔记的问题，必须调用搜索工具；"
            "涉及数学计算时，必须调用 calculator 工具；普通闲聊可以直接回答。"
            "只能根据工具返回的资料回答知识库问题。"
        ),
    )
    print("Agent 助手已启动，输入 quit 或 exit 退出。")
    while True:
        question = input("\n你：").strip()
        if question.lower() in {"quit", "exit"}:
            print("再见！")
            return
        if not question:
            continue
        result = agent.invoke({"messages": [("user", question)]})
        for message in result["messages"]:
            if getattr(message, "tool_calls", None):
                for call in message.tool_calls:
                    print(f"[工具调用] {call['name']}({call['args']})")
        print(f"\n助手：{result['messages'][-1].content}")


if __name__ == "__main__":
    main()
