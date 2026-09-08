"""Lesson 22: require and validate source citations in RAG answers."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag import _settings, format_documents, retrieve_relevant  # noqa: E402


def create_citation_chain():
    """Build a chain that explicitly requires a machine-checkable source section."""
    key, base_url, _ = _settings()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是严格的知识库问答助手，只能依据资料回答。"
            "回答正文结束后必须输出一行‘参考来源：’，"
            "下一行开始逐行列出实际使用的文件名。"
            "只能引用资料中出现的文件，不要编造来源。\n\n资料：\n{context}",
        ),
        ("human", "问题：{question}"),
    ])
    model = ChatOpenAI(
        model="qwen-plus",
        api_key=key,
        base_url=base_url,
        temperature=0.0,
    )
    return prompt | model | StrOutputParser()


def extract_citations(answer_text: str) -> list[str]:
    """Extract and normalize filenames listed after the citation heading."""
    marker = "参考来源："
    if marker not in answer_text:
        return []
    citation_section = answer_text.split(marker, 1)[1]
    # Accept bullets, plain filenames, and absolute paths ending in '.md'.
    paths = re.findall(r"(?:^|\n)\s*[-*]?\s*([^\n]+?\.md)\b", citation_section)
    return [Path(path.strip(" `[]()" )).name for path in paths]


def validate_citations(answer_text: str, retrieved_sources: set[str]) -> dict[str, object]:
    """Check citation presence and reject files absent from retrieved context."""
    citations = extract_citations(answer_text)
    unknown = sorted(set(citations) - retrieved_sources)
    return {
        "has_citation_section": "参考来源：" in answer_text,
        "citations": citations,
        "unknown_citations": unknown,
        "valid": bool(citations) and not unknown,
    }


def run_question(question: str) -> None:
    """Retrieve context, generate an answer, then enforce the citation contract."""
    # Step 1: retrieve the same documents that will be passed to the model.
    documents = retrieve_relevant(question, k=4)
    retrieved_sources = {
        Path(document.metadata.get("source", "")).name for document in documents
    }
    # Step 2: generate with an explicit, parseable citation requirement.
    answer_text = create_citation_chain().invoke({
        "question": question,
        "context": format_documents(documents),
    })
    # Step 3: validate the model's citations against actual retrieved sources.
    validation = validate_citations(answer_text, retrieved_sources)
    print(f"\n问题：{question}")
    print(f"检索来源：{sorted(retrieved_sources)}")
    print(f"模型引用：{validation['citations'] or '无'}")
    print(f"未知引用：{validation['unknown_citations'] or '无'}")
    print(f"引用校验：{'通过' if validation['valid'] else '失败'}")
    print(f"回答：\n{answer_text}")


def main() -> None:
    run_question("怎样让模型先查外部资料，再结合资料生成答案？")
    run_question("任务步骤会根据工具返回内容变化，应该使用固定链还是可变工作流？")


if __name__ == "__main__":
    main()
