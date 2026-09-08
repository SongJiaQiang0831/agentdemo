"""Lesson 21: reduce groundedness false positives with document-level matching."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag import answer, retrieve_relevant  # noqa: E402

QUESTIONS = [
    "怎样让模型先查外部资料，再结合资料生成答案？",
    "我想检查搜索模块有没有把正确资料放在靠前位置，该看哪些指标？",
    "任务步骤会根据工具返回内容变化，应该使用固定链还是可变工作流？",
]


def normalize_markdown(text: str) -> str:
    """Remove formatting that should not affect support matching."""
    text = re.sub(r"[*_`>#\"“”‘’]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> set[str]:
    result: set[str] = set()
    for part in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", normalize_markdown(text).lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            result.update(part[index : index + 2] for index in range(len(part) - 1))
        else:
            result.add(part)
    return result


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"[。！？!?\n]+", normalize_markdown(text))
    return [part.strip(" -:：") for part in parts if len(tokens(part)) >= 2]


def support_score(sentence: str, documents: list) -> float:
    sentence_terms = tokens(sentence)
    if not sentence_terms:
        return 0.0
    document_scores = []
    for document in documents:
        document_terms = tokens(document.page_content)
        document_scores.append(len(sentence_terms & document_terms) / len(sentence_terms))
    return max(document_scores, default=0.0)


def evaluate(question: str, threshold: float = 0.2) -> tuple[float, list[str]]:
    documents = retrieve_relevant(question, k=4)
    generated = answer(question, documents)
    sentences = split_sentences(generated)
    unsupported = [sentence for sentence in sentences if support_score(sentence, documents) < threshold]
    score = 1 - len(unsupported) / max(len(sentences), 1)
    print(f"\n问题：{question}")
    print(f"来源：{[Path(doc.metadata.get('source', '')).name for doc in documents]}")
    print(f"依据覆盖率：{score:.1%}")
    print(f"可能缺少依据：{unsupported or '无'}")
    return score, unsupported


def main() -> None:
    scores = [evaluate(question)[0] for question in QUESTIONS]
    print(f"\n平均依据覆盖率：{sum(scores) / len(scores):.1%}")


if __name__ == "__main__":
    main()
