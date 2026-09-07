# 第 10 课：错误处理与重试

本案例演示两类重试：

- `ToolRetryMiddleware`：工具发生临时错误时重试
- `ModelRetryMiddleware`：模型接口发生临时错误时重试

只对 `ConnectionError` 和 `TimeoutError` 重试。参数错误等永久错误不应盲目重试。

退避时间为 `0.1`、`0.2` 秒，最多重试两次，也就是最多执行三次。

运行：

```bash
python lessons/10_retry_and_errors/retry_agent.py
```
