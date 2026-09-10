"""Application services for retrieval, generation, and citation validation."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.schemas import AskResponse
from rag import _settings, format_documents, retrieve_relevant


class ModelAnswer(BaseModel):
    """Internal structured result produced by the language model."""

    answer: str = Field(description="仅依据检索资料生成的回答")
    citations: list[str] = Field(description="实际使用的 Markdown 文件名")


class KnowledgeService:
    """Coordinate the complete RAG use case behind one public method."""

    def __init__(self) -> None:
        key, base_url, _ = _settings()
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "你是个人学习助手，只能依据资料回答。资料不足时明确说明。"
                "citations 只填写实际使用的 Markdown 文件名，不得编造来源。"
                "\n\n资料：\n{context}",
            ),
            ("human", "问题：{question}"),
        ])
        model = ChatOpenAI(
            model=os.getenv("QWEN_MODEL", "qwen-plus"),
            api_key=key,
            base_url=base_url,
            temperature=0.0,
        )
        # Key step: force a typed model result instead of parsing free-form text.
        self._chain = prompt | model.with_structured_output(ModelAnswer)

    def ask(self, question: str, top_k: int) -> AskResponse:
        """Retrieve evidence, generate an answer, and enforce citation validity."""
        # Step 1: only documents passing the relevance threshold enter the prompt.
        documents = retrieve_relevant(question, k=top_k)
        retrieved_sources = sorted({
            Path(document.metadata.get("source", "")).name for document in documents
        })
        if not documents:
            return AskResponse(
                answer="知识库中没有足够相关的信息。",
                citations=[],
                retrieved_sources=[],
                citation_valid=True,
            )

        # Step 2: generate from exactly the context recorded above.
        model_answer = self._chain.invoke({
            "question": question,
            "context": format_documents(documents),
        })

        # Step 3: normalize paths and reject citations absent from retrieval.
        citations = sorted({Path(citation).name for citation in model_answer.citations})
        unknown_citations = set(citations) - set(retrieved_sources)
        citation_valid = bool(citations) and not unknown_citations
        return AskResponse(
            answer=model_answer.answer,
            citations=citations,
            retrieved_sources=retrieved_sources,
            citation_valid=citation_valid,
        )


@lru_cache(maxsize=1)
def get_knowledge_service() -> KnowledgeService:
    """Create one reusable service instance, similar to a Spring singleton bean."""
    return KnowledgeService()
