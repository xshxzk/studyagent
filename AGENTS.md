# shuhan 的 AI Agent 开发工程师学习指南

## 个人与目录约定

- 我叫 shuhan，现在正在学习 AI Agent。
- 我的主学习资料目录是上一级文件夹：`D:\AI_project\AgentGuide`。
- 如果我问你学习计划，就是默认上AgentGuide中AI Agent 开发工程师的学习计划
- 你可以无条件翻看 `D:\AI_project\AgentGuide` 的内容，用来理解我的学习路线、参考资料和项目背景。
- 如果要删除或修改 `D:\AI_project\AgentGuide` 里的任何内容，必须先经过我的同意。
- 当前项目 `D:\AI_project\AgentGuide\xzkStudyagent` 是我的个人学习与练习区，可以把学习记录、项目代码、总结和工程实践沉淀在这里。

## 当前学习方向

我的当前目标是成为 **AI Agent 开发工程师**，重点不是纯算法研究，而是能把 Agent/RAG/LLM 应用做成可运行、可部署、可评估、可维护的工程系统。

学习时要优先围绕以下能力展开：

1. 后端工程能力：FastAPI、异步 I/O、Pydantic、API 设计、任务队列、错误处理。
2. RAG 系统能力：文档解析、分块、Embedding、向量数据库、混合检索、Rerank、RAG 评估。
3. Agent 开发能力：ReAct、Tool Calling、Function Calling、自定义工具、Memory、规划与工作流。
4. Agent 工程化能力：缓存、并发、降级、重试、状态持久化、权限控制、审计日志。
5. 生产部署能力：Docker、Docker Compose、Redis、Milvus/Chroma、Prometheus、Grafana、LangSmith/LangFuse。
6. 系统设计与面试能力：高并发 RAG、Multi-Agent、LLM Gateway、可观测性、成本优化、ROI 评估。

## 核心学习主线

以 `D:\AI_project\AgentGuide\docs\05-roadmaps\learning-roadmap-development.md` 为主线学习，这是 AI Agent 开发工程师的工程落地版路线。

整体目标：

- 8 周内完成从原型到生产级 Agent 系统的完整训练。
- 至少做出 2 个可以写进简历的完整项目。
- 每个项目都要有需求分析、架构设计、核心代码、部署方式、评估指标和项目总结。
- 简历表达要量化，例如 QPS、P99 延迟、命中率、准确率、成本下降、成功率提升等。

## 8 周学习路线

### 第 1 周：大模型应用开发基础 + Naive RAG

目标：能从 0 搭建一个文档问答 API。

重点：

- FastAPI 路由、异步接口、Pydantic 数据校验。
- LangChain 基础组件：LLM、Prompt Templates、Output Parsers、LCEL。
- RAG 基础流程：Document Loader、Text Splitter、Embedding、Vector Store。
- 本地向量库：FAISS、Chroma。

产出：

- 一个 FastAPI Hello World 服务。
- 一个 LangChain LCEL Demo。
- 一个完整 Naive RAG 文档问答接口。
- 用 Docker 打包并能本地运行。

### 第 2 周：Advanced RAG 与生产级向量数据库

目标：把基础 RAG 升级成更接近生产的检索系统。

重点：

- Query Transformation：HyDE、Multi-Query 等查询改写。
- Hybrid Search：BM25 + 向量检索。
- Rerank：Cohere Rerank 或其他重排模型。
- RAG 评估：RAGAs、DeepEval、TruLens。
- 生产级向量数据库：Milvus、Qdrant，也要继续熟悉 Chroma。
- 复杂文档解析：Unstructured、MinerU、Docling、PyMuPDF。

产出：

- 在 Week1 RAG 基础上增加混合检索、重排和评估。
- 记录优化前后的指标，例如召回率、Faithfulness、Answer Relevancy、延迟。

### 第 3 周：Agent 开发与 Tool Calling

目标：能构建可以调用工具的真实 Agent。

重点：

- Agent 核心：ReAct、Planning、Tool Use、Memory。
- Tool Calling / Function Calling 的 schema 设计。
- 自定义工具开发：天气/API 工具、SQL 查询工具、文件/搜索工具。
- 工具失败处理：重试、降级、结构化错误返回。

产出：

- 至少 3 个自定义工具。
- 一个可以链式调用工具的 Agent。
- 一个结合 RAG + Web 搜索工具的研究助手 Agent。

### 第 4 周：系统性能优化

目标：让 Agent/RAG 系统从“能跑”变成“跑得稳、跑得快、跑得便宜”。

重点：

- 性能瓶颈分析：cProfile、py-spy、Scalene。
- Redis 缓存：LLM 响应缓存、Embedding 缓存、检索结果缓存。
- FastAPI 异步改造：asyncio、aiohttp。
- 批处理：Embedding batch、Reranker batch、LLM batch。
- 推理服务：vLLM、SGLang、Ollama。
- 压测：locust 或 jmeter，关注 QPS、P99、错误率。

产出：

- 一份性能压测报告。
- 对比优化前后的 QPS、P99 延迟、缓存命中率和成本。

### 第 5 周：监控、可观测性与部署

目标：具备 LLM 应用生产化意识。

重点：

- Agent 链路追踪：LangSmith、LangFuse、OpenTelemetry。
- 指标监控：Prometheus、prometheus-fastapi-instrumentator。
- 可视化大盘：Grafana。
- 日志：Python logging、Loguru、structlog，优先 JSON 日志。
- 容器化：Docker、Docker Compose。
- 生产环境模拟：故障定位、日志排查、指标分析。

产出：

- FastAPI + Agent/RAG + Redis + Milvus/Chroma 的 Docker Compose 部署。
- 至少暴露 QPS、延迟、错误率、token 消耗、缓存命中率等指标。
- 一次故障演练和排查记录。

### 第 6 周：Multi-Agent 系统开发

目标：掌握多智能体协作系统的设计与取舍。

重点：

- AutoGen：ConversableAgent、GroupChat、多 Agent 协作。
- CrewAI：Agent、Task、Crew、Process。
- LangGraph：状态机、条件分支、循环、复杂工作流。
- 状态共享、任务编排、失败恢复和中间产物记录。

产出：

- 一个“研究员-程序员-测试员”的软件开发 Multi-Agent Demo。
- 一个角色分工清晰的 CrewAI 或 AutoGen Demo。
- 对比 LangGraph、AutoGen、CrewAI 的适用场景。

### 第 7-8 周：工业级项目实战与面试准备

目标：完成 1-2 个可写进简历的完整工程项目。

推荐项目 1：企业级智能客服 RAG 系统

- 场景：电商客服，回答订单、物流、退款、FAQ 等问题。
- 技术：FastAPI + LangChain/LlamaIndex + 混合检索 + Milvus/Chroma + Redis + Docker Compose。
- 工程要求：数据库精确查询优先，文档检索兜底；有监控、日志、评估和部署。
- 简历亮点：高并发、低延迟、生产级监控、节省客服成本。

推荐项目 2：Agent 驱动的自动化投研/研究报告系统

- 场景：输入公司名或主题，自动完成信息搜集、文档解析、分析和报告生成。
- 技术：CrewAI/AutoGen/LangGraph + RAG + 搜索/API 工具 + 文件读写 + 评估。
- 工程要求：至少 5 个工具；记录每一步中间产出；有异常处理和重试；任务成功率可量化。
- 简历亮点：Multi-Agent 协作、复杂工作流自动化、效率提升。

## 推荐技术栈

RAG 项目优先栈：

```text
后端：FastAPI + Pydantic
LLM 编排：LangChain / LlamaIndex
向量库：Chroma（学习期）→ Milvus/Qdrant（生产期）
检索：BM25 + Embedding + Reranker
评估：RAGAs / DeepEval / TruLens
缓存：Redis
监控：LangSmith / LangFuse + Prometheus + Grafana
部署：Docker + Docker Compose
```

Agent 项目优先栈：

```text
单 Agent：LangChain Tools + Function Calling
复杂工作流：LangGraph
Multi-Agent：AutoGen / CrewAI
记忆：LangChain Memory / Mem0 / Zep
工具：SQL、搜索、文件、API、业务系统工具
可靠性：tenacity 重试、结构化错误、人工确认点、审计日志
```

## 框架选择原则

- 学 Agent 原理：先看 Swarm 或手写简化 ReAct，不要一开始就只会调框架。
- 快速 Demo：CrewAI 上手快，适合角色明确的小项目。
- 复杂工作流：LangGraph 最适合状态管理、条件分支、循环和可控流程。
- Multi-Agent：AutoGen 适合多智能体协作和对话式编排。
- 通用生态：LangChain 必学，但要理解其抽象，不要被框架绑死。
- 高可控场景：Parlant 可关注，适合客服、医疗等规则明确、指令遵循要求高的场景。
- 模块化企业应用：AgentScope 可作为扩展学习。

## Agent 工程化核心原则

从生产实践看，可靠 Agent 不是“完全自主”，而是“边界清晰、可验证、可回滚”的智能工具。

必须牢记：

- 多步骤 Agent 会出现错误累积。每步 95% 成功率，20 步整体成功率会显著下降。
- 长对话会带来 token 成本膨胀。能无状态就无状态，能压缩上下文就压缩。
- Agent 的难点不只是 LLM，而是工具工程。工具要返回结构化、可恢复、低噪声的信息。
- 关键操作要有人类确认点，尤其是写数据库、发消息、支付、删除、部署等动作。
- 生产系统要靠传统工程保证可靠性：权限、事务、幂等、重试、降级、日志、监控、回滚。

## 上下文工程重点

Agent 开发的本质是上下文工程：在正确时间，用正确格式，给 LLM 正确信息。

开发岗优先阅读：

- `docs\02-tech-stack\12-factor-agent-architecture.md`：生产级 Agent 架构。
- `docs\02-tech-stack\17-claude-code-best-practices.md`：工具设计和工程实践。
- `docs\02-tech-stack\14-context-engineering.md`：长上下文失效模式与修复。
- `docs\02-tech-stack\15-agent-memory.md`：短期/长期记忆系统。
- `docs\02-tech-stack\23-lessons-learned.md`：真实项目避坑。

学习时要重点关注：

- System Prompt、User Prompt、Memory、RAG、Tools、Structured Output 的组合。
- 上下文中毒、干扰、混乱、冲突等失败模式。
- 精简、缓存、总结、过滤、结构化、分层等修复技巧。

## 必读与优先资源

入门和实践：

- Datawhale《面向开发者的 LLM 入门教程》
- Datawhale《动手学大模型应用开发》
- Datawhale Hello-Agents
- Datawhale All-in-RAG
- LangChain 官方文档
- LlamaIndex 官方文档
- OpenAI Cookbook
- HuggingFace NLP Course

Agent 核心：

- Lilian Weng: LLM Powered Autonomous Agents
- ReAct 论文
- Reflexion 论文
- Anthropic: Building Effective Agents
- OpenAI: A Practical Guide to Building Agents
- Agentic Design Patterns

RAG 与评估：

- RAG from Scratch
- RAGAs
- DeepEval
- FlashRAG
- Anthropic Contextual Retrieval
- Modular RAG

工程部署：

- FastAPI 官方教程
- Docker / Docker Compose 官方文档
- Redis + FastAPI 实践
- Prometheus Python Client
- Grafana 文档
- LangSmith / LangFuse 文档

## 面试准备重点

开发岗高频主题：

1. 设计一个日均百万查询的企业级 RAG 系统。
2. 如何把 RAG 系统 P99 延迟从 2s 优化到 300ms。
3. 如何降低 Agent/LLM API 成本 70%。
4. LangChain、LlamaIndex、LangGraph、AutoGen、CrewAI 如何选型。
5. 如何设计 Multi-Agent 智能客服系统。
6. 如何实现 Memory 系统和长期记忆。
7. 如何评估 Agent 效果：成功率、准确率、效率、成本、满意度。
8. 如何做异常处理、重试、幂等和降级。
9. 如何做监控、日志、链路追踪和可观测性。
10. 如何处理权限、隐私、幻觉、审计和人工确认。

回答面试题时要始终从工程视角组织：

```text
业务需求 → 指标约束 → 架构设计 → 核心模块 → 关键取舍 → 风险与兜底 → 评估与监控 → 简历量化结果
```

## 学习记录要求

每周结束时要在当前学习项目中沉淀总结，建议包含：

- 本周学了什么。
- 完成了哪些代码和 Demo。
- 遇到的问题和解决方法。
- 关键指标：延迟、QPS、准确率、召回率、成本、成功率等。
- 下周要改进的点。

项目 README 或周总结要尽量工程化表达，不只写“学会了”，要写“实现了什么、怎么实现、效果如何、还有什么风险”。

## 当前已有学习痕迹

当前项目已经有：

- `Week1_RAG_Basic`：FastAPI、LangChain、RAG、Chroma/FAISS、PDF RAG、前端和 Docker 相关练习。
- `Week2_Advance_RAG`：Query Transform、Hybrid Search、Rerank、RAG 评估相关练习。

后续学习时应优先在已有 Week1/Week2 基础上继续升级，而不是每次从零开新项目。这样更接近真实工程迭代，也方便形成完整简历项目。
