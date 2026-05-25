# Day16 学习计划：自定义工具开发与 Tool Schema

## 1. 今日定位

今天是 Week3 的第 2 天，主题是自定义工具开发。

Day15 已经用手写 ReAct Demo 看到了 Agent 的最小控制流：

```text
Thought -> Action -> Observation -> Final Answer
```

Day16 要把“工具”从 demo 里拆出来，变成更接近真实工程的模块：

```text
tool schema
-> 参数校验
-> 执行函数
-> 结构化返回
-> 错误处理
```

## 2. 今日目标

1. 理解工具不只是一个函数，而是一个可被 Agent 安全调用的接口。
2. 使用 Pydantic 定义工具输入参数 schema。
3. 设计统一的工具返回结构。
4. 把 Day15 的 calculator 拆成独立工具模块。
5. 创建一个 Week2 RAG 检索工具的第一版接口。
6. 运行一个 Day16 Demo，观察工具调用成功和失败时的返回。

## 3. 今日核心概念

一个生产级 Tool 通常至少包含：

```text
name：工具名
description：工具用途，帮助 Agent 判断何时调用
args_schema：参数结构，帮助 Agent 传对参数
run：执行逻辑
return：结构化结果
error handling：失败时返回可恢复信息
```

工具不是“随便写个 Python 函数”。工具是 Agent 与外部世界交互的边界。

## 4. 时间安排

| 时间 | 学习内容 | 产出 |
| --- | --- | --- |
| 20 分钟 | 理解 Tool Schema | 知道 name/description/args_schema 的作用 |
| 40 分钟 | 创建 `tools` 模块 | `tool_types.py`、`calculator_tool.py` |
| 40 分钟 | 创建 RAG 检索工具接口 | `rag_search_tool.py` |
| 30 分钟 | 运行 Day16 工具调用 Demo | `day16_custom_tools_demo.py` |
| 20 分钟 | 总结工具设计原则 | 更新 `Agent_Week3_Summary.md` |

## 5. 步骤清单

### Step 1：确认 Pydantic

为什么做：

Tool Calling 的核心是结构化参数。Pydantic 可以帮我们定义和校验参数 schema。

运行：

```powershell
python -c "import pydantic; print('pydantic OK')"
```

### Step 2：创建工具基础类型

预计文件：

```text
tools/tool_types.py
```

职责：

- 定义 `ToolResult`
- 定义 `BaseToolSpec`
- 让所有工具有统一返回格式

### Step 3：创建 calculator 工具

预计文件：

```text
tools/calculator_tool.py
```

职责：

- 用 Pydantic 定义 `CalculatorInput`
- 校验数学表达式
- 安全执行基础计算
- 返回结构化结果

### Step 4：创建 RAG 搜索工具接口

预计文件：

```text
tools/rag_search_tool.py
```

职责：

- 用 Pydantic 定义 `RagSearchInput`
- 连接 Week2 Milvus Collection：`week2_pdf_chunks`
- 搜索论文 chunks
- 返回 contexts 列表

如果 Milvus 没有启动，工具要返回失败结果，而不是让程序崩掉。

### Step 5：创建 Day16 Demo

预计文件：

```text
day16_custom_tools_demo.py
```

职责：

- 演示 calculator 成功调用
- 演示 calculator 参数错误
- 演示 rag_search_tool 调用
- 打印工具 schema 和结构化返回

## 6. 验收标准

Day16 完成后，至少满足：

- 能说清楚工具的 `name/description/args_schema/run/return`
- 能运行 `day16_custom_tools_demo.py`
- calculator 工具能成功计算
- calculator 工具能拒绝非法输入
- RAG 工具能在 Milvus 正常时返回 contexts，或在连接失败时返回结构化错误
- 更新 `Agent_Week3_Summary.md`

## 7. 与 Day15 的区别

Day15：

```text
工具直接写在一个 demo 文件里
参数只是字符串
返回只是简单 ToolResult
```

Day16：

```text
工具拆成独立模块
参数由 Pydantic schema 校验
返回包含 metadata 和 error
工具失败也要结构化
```

这就是从“能跑的 demo”走向“可维护 Agent 工程”的第一步。

## 8. 今日最终反馈格式

运行 Day16 demo 后，把这些结果发给我：

```text
1. pydantic 是否可 import
2. calculator 成功案例输出
3. calculator 非法输入输出
4. rag_search_tool 是否成功连接 Milvus
5. RAG 搜索返回了几个 contexts
6. 是否有依赖、连接或 schema 报错
```
