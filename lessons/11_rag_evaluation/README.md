# 第 11 课：RAG 检索评估

`Dataset` 是问题和期望来源组成的测试集。本案例计算 `Recall@K`：
期望来源是否出现在前 K 个检索结果中。

运行前确保已经建立索引：

```bash
python build_index.py
python lessons/11_rag_evaluation/evaluate_retrieval.py
```
