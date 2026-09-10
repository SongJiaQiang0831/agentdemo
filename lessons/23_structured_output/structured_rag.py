"""Lesson 23: use structured model output to validate RAG citations."""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag import _settings, format_documents, retrieve_relevant  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402


class RAGResponse(BaseModel):
    """DTO-like schema: the model must return answer text and source filenames."""

    answer: str = Field(description="基于资料生成的回答正文")
    citations: list[str] = Field(description="实际使用的 Markdown 文件名列表")


def create_structured_chain():
    """Create a chain whose output is parsed into RAGResponse, not raw text."""
    key, base_url, _ = _settings()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是知识库问答助手，只能依据资料回答。"
            "citations 必须填写实际使用的 Markdown 文件名，不要填写绝对路径，"
            "不要引用资料中没有出现的文件。\n\n资料：\n{context}",
        ),
        ("human", "问题：{question}"),
    ])
    model = ChatOpenAI(
        model="qwen-plus",
        api_key=key,
        base_url=base_url,
        temperature=0.0,
    )
    # with_structured_output converts the model response into RAGResponse.
    return prompt | model.with_structured_output(RAGResponse)


def validate_response(response: RAGResponse, retrieved_sources: set[str]) -> dict[str, object]:
    """Check that every cited file was actually retrieved for this question."""
    normalized = {Path(citation).name for citation in response.citations}
    unknown = sorted(normalized - retrieved_sources)
    return {
        "citations": sorted(normalized),
        "unknown_citations": unknown,
        "valid": bool(normalized) and not unknown,
    }


def run_question(question: str) -> None:
    """Execute retrieval, structured generation, and citation validation."""
    # Step 1: retrieve context and record the exact source set.
    documents = retrieve_relevant(question, k=4)
    retrieved_sources = {
        Path(document.metadata.get("source", "")).name for document in documents
    }
    # Step 2: pass only retrieved context to the structured-output chain.
    response = create_structured_chain().invoke({
        "question": question,
        "context": format_documents(documents),
    })
    # Step 3: compare model citations with actual sources.
    validation = validate_response(response, retrieved_sources)
    print(f"\n问题：{question}")
    print(f"检索来源：{sorted(retrieved_sources)}")
    print(f"模型引用：{validation['citations']}")
    print(f"未知引用：{validation['unknown_citations'] or '无'}")
    print(f"引用校验：{'通过' if validation['valid'] else '失败'}")
    print(f"回答：{response.answer}")


def main() -> None:
    run_question("怎样让模型先查外部资料，再结合资料生成答案？")
    run_question("任务步骤会根据工具返回内容变化，应该使用固定链还是可变工作流？")


if __name__ == "__main__":
    main()
