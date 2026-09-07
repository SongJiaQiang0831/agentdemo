"""Shared RAG components: document loading, indexing, retrieval and answering."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_ROOT / "data/documents"
PERSIST_DIR = PROJECT_ROOT / "data/chroma"


def _settings() -> tuple[str, str, str]:
    load_dotenv()
    key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("未找到 API Key，请在 .env 中设置 DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL") or os.getenv("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3")
    return key, base_url, embedding_model


def embeddings() -> OpenAIEmbeddings:
    key, base_url, embedding_model = _settings()
    # DashScope expects raw strings. Disabling tiktoken avoids sending token IDs.
    return OpenAIEmbeddings(
        model=embedding_model,
        api_key=key,
        base_url=base_url,
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )


def load_documents() -> list[Document]:
    if not DOCUMENTS_DIR.exists():
        raise RuntimeError(f"文档目录不存在：{DOCUMENTS_DIR}")
    loader = DirectoryLoader(
        str(DOCUMENTS_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    documents = loader.load()
    if not documents:
        raise RuntimeError(f"{DOCUMENTS_DIR} 中没有找到 Markdown 文档")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    return splitter.split_documents(documents)


def build_index() -> int:
    get_vector_store.cache_clear()
    chunks = load_documents()
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    if (PERSIST_DIR / "chroma.sqlite3").exists():
        old_store = Chroma(
            persist_directory=str(PERSIST_DIR),
            embedding_function=embeddings(),
            collection_name="knowledge",
        )
        old_store.delete_collection()
    Chroma.from_documents(chunks, embedding=embeddings(), persist_directory=str(PERSIST_DIR), collection_name="knowledge")
    get_vector_store.cache_clear()
    return len(chunks)


def retrieve(question: str, k: int = 4) -> list[Document]:
    return get_vector_store().similarity_search(question, k=k)


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Open the persisted Chroma vector store."""
    return Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings(),
        collection_name="knowledge",
    )


def create_retriever(k: int = 4):
    """Create a Retriever, a component that turns a question into documents."""
    return get_vector_store().as_retriever(search_kwargs={"k": k})


def retrieve_with_scores(question: str, k: int = 4) -> list[tuple[Document, float]]:
    """Return documents and Chroma distance scores (lower means more similar)."""
    return get_vector_store().similarity_search_with_score(question, k=k)


def retrieve_relevant(
    question: str, k: int = 4, max_distance: float | None = None
) -> list[Document]:
    """Retrieve only documents within the configured distance threshold."""
    if max_distance is None:
        max_distance = float(os.getenv("RAG_MAX_DISTANCE", "1.2"))
    return [
        document
        for document, score in retrieve_with_scores(question, k=k)
        if score <= max_distance
    ]


def format_documents(documents: list[Document]) -> str:
    return "\n\n".join(
        f"来源：{doc.metadata.get('source', '未知')}\n{doc.page_content}"
        for doc in documents
    )


def format_sources(documents: list[Document]) -> str:
    """Format unique source paths for a readable citation list."""
    sources = dict.fromkeys(doc.metadata.get("source", "未知") for doc in documents)
    return "\n".join(f"- {source}" for source in sources)


def create_answer_chain():
    key, base_url, _ = _settings()
    model = ChatOpenAI(model=os.getenv("QWEN_MODEL", "qwen-plus"), api_key=key, base_url=base_url, temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是知识库问答助手。只能依据下方资料回答；"
            "资料没有答案时，明确说‘知识库中没有相关信息’。\n\n"
            "资料：\n{context}",
        ),
        ("human", "问题：{question}"),
    ])
    return (
        {
            "context": RunnableLambda(lambda value: format_documents(value["documents"])),
            "question": RunnableLambda(lambda value: value["question"]),
        }
        | prompt
        | model
        | StrOutputParser()
    )


def create_rag_chain(k: int = 4):
    """Build an end-to-end LCEL chain: question -> retrieval -> answer."""
    retriever = RunnableLambda(lambda value: retrieve_relevant(value["question"], k=k))
    return (
        {
            "context": retriever | RunnableLambda(format_documents),
            "question": RunnableLambda(lambda value: value["question"]),
            "chat_history": RunnableLambda(lambda value: value.get("chat_history", [])),
        }
        | ChatPromptTemplate.from_messages([
            (
                "system",
                "你是知识库问答助手。只能依据下方资料回答；"
                "资料没有答案时，明确说‘知识库中没有相关信息’。"
                "回答末尾必须添加‘参考来源：’并列出实际使用的文件路径。\n\n"
                "资料：\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "问题：{question}"),
        ])
        | ChatOpenAI(
            model=os.getenv("QWEN_MODEL", "qwen-plus"),
            api_key=_settings()[0],
            base_url=_settings()[1],
            temperature=0.2,
        )
        | StrOutputParser()
    )


def answer_from_knowledge(question: str, k: int = 4) -> str:
    return create_rag_chain(k=k).invoke({"question": question})


def answer(question: str, documents: list[Document]) -> str:
    chain = create_answer_chain()
    return chain.invoke({"question": question, "documents": documents})
