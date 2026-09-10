# Qwen LangChain 学习项目

当前项目包含两个版本：基础对话和本地知识库 RAG。

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

在 `.env` 中填写未注释的 `DASHSCOPE_API_KEY`。输入 `quit` 或 `exit` 退出。

## 第一个 RAG 版本

把自己的 Markdown 笔记放入 `data/documents/`，先建立向量索引：

```bash
python build_index.py
python rag_chat.py
```

提问时程序会先打印检索到的文件来源，再让 Qwen 依据这些片段回答。
当前版本使用 DashScope 的 `text-embedding-v3`，可通过 `QWEN_EMBEDDING_MODEL` 修改。
`RAG_MAX_DISTANCE` 控制检索距离阈值，默认 `1.2`；距离超过阈值的片段会被过滤。

## 个人 AI 学习助手 MVP

项目已进入真实应用阶段。FastAPI 后端位于 `app/`，运行方式与接口说明见 `app/README.md`。
