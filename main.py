"""A minimal command-line chat program for Alibaba Cloud Qwen."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def create_model() -> ChatOpenAI:
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未找到 API Key。请在 .env 中设置 DASHSCOPE_API_KEY=你的密钥")
    base_url = os.getenv("DASHSCOPE_BASE_URL") or os.getenv("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return ChatOpenAI(model=os.getenv("QWEN_MODEL", "qwen-plus"), api_key=api_key, base_url=base_url, temperature=0.7)


def ask(model: ChatOpenAI, question: str) -> str:
    response = model.invoke([
        SystemMessage(content="你是一个耐心、清晰的 Python 和 RAG 学习助手。"),
        HumanMessage(content=question),
    ])
    return str(response.content)


def main() -> None:
    try:
        model = create_model()
    except Exception as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print("Qwen 学习助手已启动，输入 quit 或 exit 退出。")
    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if question.lower() in {"quit", "exit"}:
            print("再见！")
            break
        if not question:
            continue
        try:
            print(f"\n助手：{ask(model, question)}")
        except Exception as exc:
            print(f"调用模型失败：{exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
