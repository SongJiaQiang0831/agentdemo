"""Lesson 20: estimate whether generated answer sentences are grounded in context."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag import answer, format_documents, retrieve_relevant  # noqa: E402


DATASET = [
    "怎样让模型先查外部资料，再结合资料生成答案？",
    "我想检查搜索模块有没有把正确资料放在靠前位置，该看哪些指标？",
    "任务步骤会根据工具返回内容变化，应该使用固定链还是可变工作流？",
]


def tokens(text: str) -> set[str]:
    """Extract simple English words and Chinese bigrams for overlap comparison."""
    result: set[str] = set()
    for part in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            result.update(part[index : index + 2] for index in range(len(part) - 1))
        else:
            result.add(part)
    return result


def split_sentences(text: str) -> list[str]:
    """Split Chinese and English prose at common sentence boundaries."""
    return [part.strip() for part in re.split(r"[。！？!?\n]+", text) if part.strip()]


def grounding_score(answer_text: str, context: str, threshold: float = 0.15) -> tuple[float, list[str]]:
    context_terms = tokens(context)
    sentences = split_sentences(answer_text)
    unsupported: list[str] = []
    grounded = 0
    for sentence in sentences:
        sentence_terms = tokens(sentence)
        overlap = len(sentence_terms & context_terms) / max(len(sentence_terms), 1)
        if overlap >= threshold:
            grounded += 1
        else:
            unsupported.append(sentence)
    return grounded / max(len(sentences), 1), unsupported


def main() -> None:
    scores: list[float] = []
    for question in DATASET:
        documents = retrieve_relevant(question, k=4)
        context = format_documents(documents)
        generated = answer(question, documents)
        score, unsupported = grounding_score(generated, context)
        scores.append(score)
        print(f"\n问题：{question}")
        print(f"依据覆盖率：{score:.1%}")
        print(f"检索来源：{[Path(doc.metadata.get('source', '')).name for doc in documents]}")
        print(f"可能缺少依据的句子：{unsupported or '无'}")
        print(f"回答：{generated}")
    print(f"\n平均依据覆盖率：{sum(scores) / len(scores):.1%}")


if __name__ == "__main__":
    main()
