# 第 14 课：流式 RAG

本案例把 `stream()` 接到完整的 RAG Chain：先显示检索来源，再逐段显示 Qwen 的回答。

运行：

```bash
python lessons/14_streaming_rag/streaming_rag.py
```

输入问题后，答案会随着模型生成逐步出现在终端。
