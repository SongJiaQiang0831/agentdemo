"""Lesson 16: combine vector and lexical retrieval, then rerank candidates."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document  # noqa: E402
from rag import load_documents, retrieve_with_scores  # noqa: E402


def tokenize(text: str) -> list[str]:
    """Keep English words and overlapping two-character CJK terms."""
    terms: list[str] = []
    for part in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            terms.extend(part[index : index + 2] for index in range(len(part) - 1))
        else:
            terms.append(part)
    return terms


def lexical_score(question: str, document: Document) -> float:
    query_terms = set(tokenize(question))
    document_terms = tokenize(document.page_content)
    if not query_terms or not document_terms:
        return 0.0
    matches = sum(term in document_terms for term in query_terms)
    return matches / len(query_terms)


def hybrid_retrieve(question: str, k: int = 4, candidate_k: int = 12) -> list[tuple[Document, float]]:
    """Return documents ranked by a weighted vector/lexical relevance score."""
    vector_results = retrieve_with_scores(question, k=candidate_k)
    all_documents = load_documents()
    candidates: dict[str, Document] = {
        doc.metadata.get("source", "") + "\n" + doc.page_content: doc
        for doc, _ in vector_results
    }
    # Add lexical matches that vector search may miss, especially exact terms.
    for doc in all_documents:
        if lexical_score(question, doc) > 0:
            candidates.setdefault(doc.metadata.get("source", "") + "\n" + doc.page_content, doc)

    distances = [score for _, score in vector_results]
    max_distance = max(distances, default=1.0)
    min_distance = min(distances, default=0.0)
    span = max(max_distance - min_distance, 1e-9)
    vector_scores = {
        doc.metadata.get("source", "") + "\n" + doc.page_content: 1 - (score - min_distance) / span
        for doc, score in vector_results
    }
    ranked: list[tuple[Document, float]] = []
    for key, doc in candidates.items():
        vector_score = vector_scores.get(key, 0.0)
        lexical = lexical_score(question, doc)
        combined = 0.7 * vector_score + 0.3 * lexical
        ranked.append((doc, combined))
    return sorted(ranked, key=lambda item: item[1], reverse=True)[:k]


DATASET = [
    ("RAG 的基本流程是什么？", "rag_intro.md"),
    ("Python 如何定义函数？", "python_basics.md"),
    ("Agent 为什么需要工具？", "agent_intro.md"),
]


def recall_at_k(retriever, k: int) -> float:
    hits = 0
    for question, expected_source in DATASET:
        results = retriever(question, k=k)
        sources = [Path(doc.metadata.get("source", "")).name for doc, _ in results]
        hits += expected_source in sources
        print(f"- {question} | 期望：{expected_source} | 实际：{sources}")
    return hits / len(DATASET)


def main() -> None:
    print("向量检索 Recall@2")
    baseline = recall_at_k(retrieve_with_scores, k=2)
    print(f"结果：{baseline:.1%}\n")
    print("混合检索 + 重排序 Recall@2")
    hybrid = recall_at_k(hybrid_retrieve, k=2)
    print(f"结果：{hybrid:.1%}")


if __name__ == "__main__":
    main()
