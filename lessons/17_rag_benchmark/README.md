# 第 17 课：构造 RAG 基准集

本案例使用主题相近的文档测试检索排序，比较向量检索与混合检索的 `Recall@1` 和 `Recall@2`。

先更新索引：

```bash
python build_index.py
```

再运行：

```bash
python lessons/17_rag_benchmark/benchmark.py
```

评估集中的 `expected_source` 是人工标注的正确来源。真实项目还应加入多个正确来源、难例和版本化数据集。
