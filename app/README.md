# 个人 AI 学习助手 MVP

当前版本提供：

- `/`：可直接使用的学习问答页面
- `GET /health`：本地健康检查
- `POST /api/v1/knowledge/ask`：带结构化引用校验的 RAG 问答
- `/docs`：FastAPI 自动生成的 Swagger UI

安装依赖并启动：

```bash
python -m pip install -r requirements.txt
python run_app.py
```

开发时需要自动重载，也可以运行 `python -m uvicorn app.main:app --reload`。

请求示例：

```json
{
  "question": "RAG 的基本流程是什么？",
  "top_k": 4
}
```

Python 与 Java/Spring 对照：

- `app/main.py`：Controller 和应用启动入口
- `app/services.py`：Service 层
- `app/schemas.py`：请求与响应 DTO
- `Depends(...)`：轻量依赖注入
- `response_model`：接口响应类型约束
