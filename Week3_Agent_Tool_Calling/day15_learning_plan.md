# Day15 学习计划：Agent 核心概念与 ReAct 入门

## 1. 今日定位

今天是第 3 周第 1 天，正式从 Advanced RAG 切换到 Agent 开发与 Tool Calling。

Week2 已经完成了真实 PDF chunks 接入 Milvus 的 RAG 检索链路。Week3 不从零开始，而是把 Week2 的 RAG 检索能力逐步封装成 Agent 可调用的工具。

Day15 的重点不是立刻写复杂 Agent，而是先理解：

```text
LLM Chain
-> Tool Calling
-> ReAct Agent
-> RAG Tool Agent
```

## 2. 今日目标

1. 理解 Agent 和普通 LLM Chain 的区别。
2. 理解 ReAct 的核心循环：Reason + Act + Observe。
3. 明确 Tool 在 Agent 中的工程角色。
4. 运行一个最小可理解的 ReAct Demo。
5. 设计 Week3 的工具目录结构，为 Day16 自定义工具做准备。
6. 形成从 Week2 RAG 到 Week3 Agent 的衔接思路。

## 3. 今日核心概念

### 普通 LLM Chain

```text
用户问题
-> Prompt
-> LLM
-> 答案
```

普通 Chain 的特点是：模型只能基于上下文和自身参数回答，不能主动查工具。

### Agent

```text
用户目标
-> LLM 判断下一步
-> 调用工具
-> 观察工具结果
-> 再判断
-> 最终回答
```

Agent 的关键是：LLM 不只是生成答案，还负责决定“要不要调用工具、调用哪个工具、用什么参数调用”。

### ReAct

ReAct = Reasoning + Acting。

核心循环：

```text
Thought: 我需要先判断问题需要什么信息
Action: 调用某个工具
Observation: 工具返回结果
Thought: 根据结果继续推理
Final Answer: 给出最终答案
```

## 4. 时间安排

| 时间 | 学习内容 | 产出 |
| --- | --- | --- |
| 20 分钟 | 复习 Agent 与 Chain 的区别 | 能说清楚 Agent 为什么需要工具 |
| 30 分钟 | 理解 ReAct 循环 | 画出 Thought/Action/Observation 流程 |
| 40 分钟 | 创建最小 ReAct Demo | `day15_react_manual_demo.py` |
| 30 分钟 | 创建一个假工具并手动调用 | 理解 tool schema 的雏形 |
| 30 分钟 | 规划 Week3 工具结构 | 为 Day16 自定义工具做准备 |
| 20 分钟 | 总结 Day15 | 更新 `Agent_Week3_Summary.md` |

## 5. 步骤清单

### Step 1：检查当前 Python 环境

为什么做：

Week2 安装了 Unstructured，曾经造成 CrewAI 相关依赖冲突。Week3 会用 LangChain、OpenAI SDK、可能还会用 CrewAI/AutoGen。今天先确认环境，不急着安装新包。

你先运行：

```powershell
cd D:\AI_project\AgentGuide\xzkStudyagent\Week3_Agent_Tool_Calling
python --version
python -c "import langchain; print('langchain OK')"
python -c "import openai; print('openai OK')"
```

如果报错，先把错误发给我，不要马上乱装依赖。

### Step 2：创建手写 ReAct Demo

文件位置：

```text
D:\AI_project\AgentGuide\xzkStudyagent\Week3_Agent_Tool_Calling\day15_react_manual_demo.py
```

文件职责：

- 不依赖复杂 Agent 框架。
- 用普通 Python 模拟 ReAct 的 Thought/Action/Observation。
- 内置一个简单工具，例如 `calculator` 或 `paper_search_stub`。
- 让你先理解 Agent 的控制流，而不是一开始被 LangChain Agent 封装挡住。

预计运行命令：

```powershell
python day15_react_manual_demo.py
```

### Step 3：理解 Tool 的工程结构

今天先不做完整 OpenAI Function Calling，但要理解工具至少包含：

```text
name：工具名
description：什么时候该用这个工具
args_schema：工具参数
function：真实执行逻辑
return：结构化返回结果
```

示例：

```text
name: search_week2_rag
description: Search the Week2 Milvus PDF collection for lake ice research context.
args:
  query: str
  top_k: int
return:
  contexts: list[dict]
```

### Step 4：规划 Week3 项目结构

建议结构：

```text
Week3_Agent_Tool_Calling/
  day15_learning_plan.md
  day15_react_manual_demo.py
  tools/
    __init__.py
    calculator_tool.py
    rag_search_tool.py
  Agent_Week3_Summary.md
```

Day15 只创建最小 Demo；Day16 再正式写自定义工具。

### Step 5：更新 Day15 总结

今天结束时，更新：

```text
D:\AI_project\AgentGuide\xzkStudyagent\Week3_Agent_Tool_Calling\Agent_Week3_Summary.md
```

总结要记录：

- Agent 和 Chain 的区别。
- ReAct 的 Thought/Action/Observation/Final Answer。
- Tool 为什么要结构化。
- Week2 RAG 如何变成 Week3 的工具。
- 运行结果和遇到的依赖问题。

## 6. 验收标准

Day15 完成后，至少满足：

- 能说清楚 Agent 与普通 Chain 的区别。
- 能画出 ReAct 循环。
- 能运行一个手写 ReAct Demo。
- 能说清楚 Tool 的 `name/description/args/function/return`。
- 建好 Week3 目录。
- 更新 `Agent_Week3_Summary.md`。

## 7. 与 Week2 的衔接

Week2 产物：

```text
Milvus Collection: week2_pdf_chunks
Script: day14_milvus_search_pdf_chunks.py
```

Week3 目标：

```text
把 Milvus search 封装成 Agent Tool
-> Agent 判断何时调用 RAG Tool
-> Tool 返回 contexts
-> Agent 组织最终回答
```

这就是从 RAG 系统到 Agent 系统的自然过渡。

## 8. 今日注意事项

- 当前项目根目录的 `AGENTS.md` 仍有未提交本地改动，今天不主动处理它。
- Week2 的 `volumes/` 和 `parsed_outputs/` 仍然不提交。
- 如果今天遇到依赖冲突，优先记录原因，不要急着把所有 Agent 框架都装进一个环境。
- 今天优先理解 Agent 控制流，不追求复杂框架。

## 9. 今日最终反馈格式

请把以下结果反馈给我：

```text
1. python --version 输出
2. langchain 是否可 import
3. openai 是否可 import
4. 是否有依赖冲突或报错
```

我会根据你的环境结果继续创建 Day15 的 ReAct Demo。
