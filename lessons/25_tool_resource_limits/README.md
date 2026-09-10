# 第 25 课：工具资源限制

字符白名单和 AST 白名单只能阻止任意代码执行，仍需防止合法表达式消耗过多资源。本课增加：

- AST 节点数量限制
- AST 嵌套深度限制
- 指数大小限制
- 中间结果和值域限制
- 有限且可分类的错误结果

运行：

```bash
python lessons/25_tool_resource_limits/limited_calculator.py
```

这对应 Java 服务中的请求大小限制、递归深度限制、业务值域校验和资源配额。
