# 第 07 课：持久化对话状态

本案例使用 SQLite Checkpointer 保存 LangGraph 的消息状态。相同 `thread_id`
会继续同一段对话，不同 `thread_id` 会得到相互隔离的对话。

交互运行：

```bash
python lessons/07_persistent_memory/persistent_chat.py --thread user-1
```

分两次运行，验证程序重启后的记忆：

```bash
python lessons/07_persistent_memory/persistent_chat.py --thread user-1 --message "我叫小明"
python lessons/07_persistent_memory/persistent_chat.py --thread user-1 --message "我叫什么名字？"
```

会话数据保存在当前目录的 `conversation.sqlite`。
