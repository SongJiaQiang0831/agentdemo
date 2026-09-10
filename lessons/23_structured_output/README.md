# 第 23 课：结构化输出与引用校验

第 22 课让模型在自由文本末尾输出引用，再用正则解析。本课使用 Pydantic 定义固定返回结构：

```json
{
  "answer": "回答正文",
  "citations": ["rag_intro.md"]
}
```

流程：检索上下文 -> 模型按 Schema 输出 -> 校验引用是否来自检索结果 -> 打印结构化报告。

运行：

```bash
python lessons/23_structured_output/structured_rag.py
```

结构化输出能减少格式解析错误，但不能自动保证事实正确，仍需结合检索和忠实性评估。
