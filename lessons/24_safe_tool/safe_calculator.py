"""Lesson 24: validate structured Agent tool arguments before execution."""

from __future__ import annotations

import ast
import operator
import re
from typing import Any

from langchain.tools import tool
from pydantic import BaseModel, Field, field_validator


class CalculatorInput(BaseModel):
    """DTO-like input object for the calculator tool."""

    expression: str = Field(min_length=1, max_length=100, description="基础算术表达式")

    @field_validator("expression")
    @classmethod
    def allow_only_arithmetic_characters(cls, value: str) -> str:
        """Reject names, calls, imports, and other non-arithmetic input early."""
        if not re.fullmatch(r"[0-9+\-*/%().\s]+", value):
            raise ValueError("只允许数字、空格和基础算术运算符")
        return value


def evaluate(node: ast.AST) -> float:
    """Evaluate a restricted AST; never use Python eval on model input."""
    operators: dict[type[ast.operator], Any] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = evaluate(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in operators:
        return operators[type(node.op)](evaluate(node.left), evaluate(node.right))
    raise ValueError("表达式包含不支持的语法")


@tool(args_schema=CalculatorInput)
def safe_calculator(expression: str) -> str:
    """Safely calculate a basic arithmetic expression after schema validation."""
    try:
        # Parsing creates an AST; the evaluator below accepts only approved nodes.
        tree = ast.parse(expression, mode="eval")
        result = evaluate(tree.body)
        return str(result)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
        return f"计算失败：{exc}"


def try_tool(raw_input: dict[str, object]) -> None:
    """Show validation and execution behavior for one untrusted tool payload."""
    try:
        result = safe_calculator.invoke(raw_input)
        print(f"输入：{raw_input} -> {result}")
    except Exception as exc:
        # Invalid schema input is rejected before the function body runs.
        print(f"输入：{raw_input} -> 参数校验失败：{exc}")


def main() -> None:
    try_tool({"expression": "18 * 7 + 2"})
    try_tool({"expression": "(10 + 5) / 3"})
    try_tool({"expression": "__import__('os').system('whoami')"})
    try_tool({"expression": "answer + 1"})
    try_tool({"expression": ""})


if __name__ == "__main__":
    main()
