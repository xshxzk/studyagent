# Week3 Agent 与 Tool Calling 学习总结

## Day15: Agent 核心概念与 ReAct 入门

## 1. 今日目标

Day15 是第三周第一天，学习重点从 RAG 检索系统切换到 Agent 开发。

今天的核心目标是先理解 Agent 的控制流，而不是一开始陷入复杂框架。

## 2. Agent 与普通 Chain 的区别

普通 LLM Chain：

```text
用户问题
-> Prompt
-> LLM
-> 答案
```

Agent：

```text
用户目标
-> LLM 判断是否需要工具
-> 调用工具
-> 观察工具结果
-> 根据结果继续推理
-> 最终回答
```

普通 Chain 更像一次性问答；Agent 更像一个带工具箱的任务控制器。

## 3. ReAct 核心循环

ReAct = Reasoning + Acting。

核心结构：

```text
Thought: 我现在需要做什么？
Action: 调用哪个工具？
Action Input: 工具参数是什么？
Observation: 工具返回了什么？
Thought: 这个结果是否足够？
Final Answer: 最终回答用户
```

## 4. 今日代码产出

```text
day15_react_manual_demo.py
```

这个脚本不依赖复杂 Agent 框架，而是用普通 Python 手写一个最小 ReAct 流程。

内置两个工具：

| 工具 | 作用 |
| --- | --- |
| `calculator` | 处理简单数学表达式 |
| `paper_search` | 模拟搜索 Week2 湖冰论文知识 |

## 5. 今日关键理解

Tool 至少需要具备：

```text
name
description
args/input
function
structured return
```

其中 `description` 很重要，因为真实 Agent 会根据工具描述判断什么时候该调用这个工具。

## 6. 与 Week2 的衔接

Week2 已经有：

```text
Milvus Collection: week2_pdf_chunks
Search Script: day14_milvus_search_pdf_chunks.py
```

Week3 的自然升级方向：

```text
把 Milvus PDF 检索逻辑封装成 rag_search_tool
-> Agent 判断何时调用这个工具
-> Tool 返回 contexts
-> Agent 基于 contexts 组织最终答案
```

Day15 先用 `paper_search_stub` 模拟这个过程。Day16 可以正式把 Week2 的 Milvus 检索封装成自定义工具。


## Day16: 自定义工具开发与 Tool Schema

## 1. 今日目标

Day16 的目标是理解 Agent Tool 的组成。今天没有急着接 LLM，也没有直接上复杂 RAG Tool，而是先把一个工具的标准接口拆开学习。

核心问题：

```text
Action 调用的工具，到底应该怎么写？
```

## 2. 今日代码产出

新增/使用文件：

```text
day16_learning_plan.md
day16_step1_tool_result_demo.py
tools/__init__.py
tools/tool_types.py
```

其中 `day16_step1_tool_result_demo.py` 是今天的主 demo，`tools/tool_types.py` 放公共工具返回类型。

## 3. ToolResult：统一工具返回格式

今天先定义了统一返回结构：

```python
@dataclass
class ToolResult:
    success: bool
    content: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
```

含义：

| 字段 | 作用 |
| --- | --- |
| `success` | 工具是否执行成功 |
| `content` | 给人或 LLM 看的简短说明 |
| `data` | 给程序继续处理的结构化数据 |
| `error` | 失败原因 |

关键理解：

```text
工具不能随便返回字符串，应该返回统一结构。
```

这样 Agent 调用工具后，才能判断下一步是继续推理、最终回答，还是处理错误。

## 4. Calculator 工具组成

今天逐步把 calculator 从普通函数升级成 Agent Tool。

一个工具包含：

```text
name
description
args_schema
run
ToolResult
```

对应代码中的对象：

```python
calculator_tool = SimpleTool(
    name="calculator",
    description="Use this tool when you need to calculate a basic arithmetic expression.",
    args_schema=CalculatorInput,
    run=calculator,
)
```

理解：

- `name`：工具叫什么。
- `description`：什么时候该用这个工具。
- `args_schema`：工具需要什么参数。
- `run`：真正执行的函数。
- `ToolResult`：工具执行后统一怎么返回。

## 5. Pydantic 与 args_schema

今天使用 Pydantic 定义参数 schema：

```python
class CalculatorInput(BaseModel):
    expression: str = Field(
        description="A basic arithmetic expression, such as '12 * 3'.",
        min_length=1,
    )
```

理解：

```text
CalculatorInput 就是 calculator 工具的 args_schema。
```

Pydantic 的作用是做输入校验：

- 字段是否存在。
- 类型是否正确。
- 长度等约束是否满足。

例如：

```python
CalculatorInput(expression="12 * 3")
```

可以通过。

```python
CalculatorInput(expression="")
```

会触发 `ValidationError`，因为 `min_length=1`。

## 6. 三层保护

今天的 calculator 工具有三层保护：

### 第一层：args_schema 参数形状校验

```text
expression 必须存在
expression 必须是字符串
expression 不能为空
```

### 第二层：if 安全检查

```python
allowed_chars = set("0123456789+-*/(). ")

if not set(expression) <= allowed_chars:
```

作用：

```text
拒绝 open file、恶意代码、非数学字符等危险输入。
```

### 第三层：try/except 执行保护

```python
try:
    result = eval(expression, {"__builtins__": {}}, {})
except Exception as exc:
```

作用：

```text
防止 12 *、10 / 0 等执行时报错导致程序崩溃。
```

今日记忆：

```text
args_schema 检查参数长什么样；
if 检查内容安不安全；
try/except 防止运行时崩溃。
```

## 7. 今日关键理解

今天最终理解：

```text
Agent Tool = 一个带说明、带参数 schema、带执行逻辑、带统一返回格式的函数接口。
```

普通函数：

```text
程序员知道怎么调用。
```

Agent Tool：

```text
LLM/Agent 也能根据 name、description、args_schema 理解怎么调用。
```

所以工具工程不是简单写函数，而是设计一个 Agent 能安全使用的接口。

## 8. 下一步计划

Day16 后续可以继续做两件事：

1. 把 `SimpleTool` 抽成公共类型，放进 `tools/tool_types.py`。
2. 创建一个轻量 `paper_search_tool`，先模拟 RAG 检索，再逐步替换为 Week2 的 Milvus 搜索。

不急着一步接 LLM。先把工具接口设计稳，再进入真正的 Tool Calling。
