# 第 21 课：改进依据评估器

第 20 课的词项重叠基线会把“具体做法是：”、引用符号和 Markdown 列表误判为缺少依据。本课增加预处理和逐文档匹配：

- 去掉 Markdown 标记、引用符号和纯标点句子
- 将答案句子与每个检索文档分别比较
- 使用最高文档支持度判断句子是否有依据
- 同时输出被标记的句子，方便人工复核

运行：

```bash
python lessons/21_groundedness_improved/evaluate_grounding.py
```

这仍是可解释的教学基线，不等同于完整的事实核查或 LLM-as-a-Judge。
