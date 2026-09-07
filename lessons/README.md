# 课程案例

每个子目录对应一个学习阶段：

- `03_agent`：Agent 根据问题决定是否调用知识库搜索工具
- `04_multi_step_agent`：一个任务连续调用搜索和计算工具
- `05_langgraph`：用 State、Node、Edge 表达 RAG 工作流
- `06_conditional_graph`：根据检索结果选择回答或兜底分支
- `07_persistent_memory`：使用 SQLite 保存跨程序重启的对话状态
- `08_human_approval`：敏感操作前暂停并等待用户确认
- `09_integrated_assistant`：整合 RAG、Agent、工具和持久化会话
- `10_retry_and_errors`：处理工具和模型的临时错误并有限重试
- `11_rag_evaluation`：用测试集衡量检索是否找对文档
- `12_answer_evaluation`：检查回答关键事实并记录生成延迟
- `13_streaming`：让模型答案以流式方式实时显示
- `14_streaming_rag`：把流式输出接入完整 RAG Chain
- `15_batch_processing`：批量执行多个 RAG 问题并测量吞吐量

根目录的 `rag.py` 是后续课程共用的 RAG 组件。
