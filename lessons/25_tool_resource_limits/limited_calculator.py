"""Lesson 25: protect a valid tool call from excessive resource consumption."""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, Field, field_validator


class CalculatorInput(BaseModel):
    """Validate the external request shape before parsing its expression."""

    expression: str = Field(min_length=1, max_length=100)

    @field_validator("expression")
    @classmethod
    def validate_characters(cls, value: str) -> str:
        # Step 1: reject every character outside the arithmetic allowlist.
        if not re.fullmatch(r"[0-9+\-*/%().\s]+", value):
            raise ValueError("只允许数字、空格和基础算术运算符")
        return value


@dataclass(frozen=True)
class EvaluationLimits:
    """Central configuration for computation limits, similar to a Java record."""

    max_nodes: int = 30
    max_depth: int = 8
    max_exponent: float = 10
    max_absolute_value: float = 1_000_000_000


OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def validate_tree(tree: ast.AST, limits: EvaluationLimits) -> None:
    """Reject an expression whose syntax tree is too large or too deeply nested."""
    # Step 2: count all AST nodes to cap total expression complexity.
    nodes = list(ast.walk(tree))
    if len(nodes) > limits.max_nodes:
        raise ValueError(f"表达式过于复杂，最多允许 {limits.max_nodes} 个语法节点")

    def check_depth(node: ast.AST, depth: int) -> None:
        # Step 3: cap recursive nesting before recursive evaluation begins.
        if depth > limits.max_depth:
            raise ValueError(f"表达式嵌套过深，最多允许 {limits.max_depth} 层")
        for child in ast.iter_child_nodes(node):
            check_depth(child, depth + 1)

    check_depth(tree, 1)


def ensure_result_in_range(value: float, limits: EvaluationLimits) -> float:
    """Reject non-finite and excessively large intermediate results."""
    # Step 4: validate every intermediate value, not only the final result.
    if not math.isfinite(value):
        raise ValueError("计算结果不是有限数字")
    if abs(value) > limits.max_absolute_value:
        raise ValueError(f"计算结果超过允许范围：±{limits.max_absolute_value:g}")
    return value


def evaluate(node: ast.AST, limits: EvaluationLimits) -> float:
    """Recursively evaluate only approved nodes and operators."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return ensure_result_in_range(float(node.value), limits)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = evaluate(node.operand, limits)
        result = value if isinstance(node.op, ast.UAdd) else -value
        return ensure_result_in_range(result, limits)
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left = evaluate(node.left, limits)
        right = evaluate(node.right, limits)
        # Exponentiation needs a tighter limit than ordinary arithmetic.
        if isinstance(node.op, ast.Pow) and abs(right) > limits.max_exponent:
            raise ValueError(f"指数绝对值不能超过 {limits.max_exponent:g}")
        result = OPERATORS[type(node.op)](left, right)
        return ensure_result_in_range(result, limits)
    raise ValueError("表达式包含不支持的语法")


def calculate(raw_input: dict[str, object], limits: EvaluationLimits | None = None) -> dict[str, object]:
    """Run the complete validation pipeline and return a stable result contract."""
    active_limits = limits or EvaluationLimits()
    try:
        # Step 1 is performed by Pydantic while constructing the DTO.
        request = CalculatorInput.model_validate(raw_input)
        tree = ast.parse(request.expression, mode="eval")
        validate_tree(tree, active_limits)
        value = evaluate(tree.body, active_limits)
        return {"success": True, "value": value, "error": None}
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
        # A stable error DTO is easier for an Agent or Java API client to handle.
        return {"success": False, "value": None, "error": str(exc)}


def main() -> None:
    cases = [
        {"expression": "18 * 7 + 2"},
        {"expression": "2 ** 10"},
        {"expression": "2 ** 1000"},
        {"expression": "999999999 * 999999999"},
        {"expression": "1 / 0"},
        {"expression": "1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1"},
    ]
    for raw_input in cases:
        print(f"输入：{raw_input} -> {calculate(raw_input)}")


if __name__ == "__main__":
    main()
