# 第 24 课：结构化工具参数与安全校验

Agent 调用工具时，不能只相信模型传入的字符串。本课为计算器工具增加参数 Schema 和白名单校验：

- `expression` 必须是字符串
- 只允许数字、空格和 `+ - * / % ** ( ) .`
- 拒绝函数调用、变量名、导入语句和其他字符
- 校验失败时返回明确错误，不执行表达式

运行：

```bash
python lessons/24_safe_tool/safe_calculator.py
```

这对应 Java 后端常见的 DTO + Bean Validation + 业务层校验模式。
