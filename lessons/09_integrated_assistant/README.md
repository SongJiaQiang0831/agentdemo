# 第 09 课：整合助手

本案例整合：

- RAG 知识库搜索
- `calculator` 计算工具
- Agent 自动选择工具
- SQLite 持久化消息历史
- `thread_id` 会话隔离
- 计算器执行前的人工审批

运行：

```bash
python lessons/09_integrated_assistant/app.py
```

可测试：

```text
RAG 的基本流程是什么？
请计算 18 * 7 + 2
我叫什么名字？
```

计算器被调用时，输入 `y` 批准执行，输入 `n` 拒绝执行。知识库搜索无需审批。
