import os

from langchain_core.messages import AIMessage, HumanMessage

from rag import PERSIST_DIR, create_rag_chain, format_sources, retrieve_relevant, retrieve_with_scores


def main() -> None:
    if not PERSIST_DIR.exists():
        raise SystemExit("尚未建立索引，请先运行：python build_index.py")
    print("RAG 助手已启动，输入 quit 或 exit 退出。")
    chat_chain = create_rag_chain()
    chat_history = []
    while True:
        question = input("\n你：").strip()
        if question.lower() in {"quit", "exit"}:
            print("再见！")
            return
        if not question:
            continue
        scored_documents = retrieve_with_scores(question)
        max_distance = float(os.getenv("RAG_MAX_DISTANCE", "1.2"))
        print(f"\n检索来源（距离分数越低越相似，阈值：{max_distance:.2f}）：")
        for document, score in scored_documents:
            status = "采用" if score <= max_distance else "过滤"
            print(f"- {document.metadata.get('source', '未知')} | 距离分数：{score:.4f} | {status}")
        relevant_documents = retrieve_relevant(question)
        answer = chat_chain.invoke({"question": question, "chat_history": chat_history})
        chat_history.extend([HumanMessage(content=question), AIMessage(content=answer)])
        print(f"\n助手：{answer}")
        print(f"\n实际采用的来源：\n{format_sources(relevant_documents) or '- 无（知识库中没有足够相关资料）'}")


if __name__ == "__main__":
    main()
