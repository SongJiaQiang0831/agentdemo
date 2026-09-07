# 第 13 课：流式输出

`Streaming`（流式输出）会在模型生成每个片段时立即显示，降低用户感知的等待时间。
`Chunk`（数据块）是一次流式返回的部分内容，多个 chunk 拼起来就是完整答案。

运行：

```bash
python lessons/13_streaming/stream_chat.py
```
