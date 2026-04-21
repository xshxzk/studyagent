# 🚀 AI Agent 学习路线第一周：从零构建全栈 RAG 系统 (Day 1 - Day 7 总结)

## 🎯 第一周学习里程碑
本周，我们完成了一次史诗级的全栈技术冲刺，从写第一行后端 API 接口开始，一步步深入 LangChain 核心机制，最终打通了前端网页、大语言模型、向量数据库，并使用 Docker 实现了企业级容器化部署。这不仅是代码的堆砌，更是 AI 产品经理和开发工程师思维的重塑。

---

## 🛠️ 第一部分：核心代码与技术串讲 (Day 1 - Day 7)

### Day 1: FastAPI 基础 —— 打造系统的“服务员”
**核心目标**：将 Python 脚本变成可以被网络访问的 Web 服务。
* **最核心代码**：
    ```python
    @app.post("/ask")
    async def ask_question(query: UserQuery):
    ```
* **原理解析**：`FastAPI` 负责监听特定的网络端口。`@app.post` 定义了前端发送数据的路径和方式。`Pydantic` (如 `UserQuery` 模型) 负责严格校验用户传来的数据格式。这是全栈架构中前后端通信的桥梁。

### Day 2: LangChain 与 LCEL 语法 —— 组装流水线
**核心目标**：使用工业级框架连接和控制大语言模型。
* **最核心代码**：
    ```python
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": "FastAPI优势？"})
    ```
* **原理解析**：这是 LangChain 最现代的 **LCEL (表达式语言)**。它像水管一样，把“提示词组装 -> 发送给大模型 -> 解析提取文字”串联起来，极大地简化了之前复杂的代码结构。

### Day 3: 文档加载与分块 (Chunking) —— 嚼碎知识
**核心目标**：大模型读不完一整本 PDF，需要切成小块。
* **最核心代码**：
    ```python
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    ```
* **原理解析**：`RecursiveCharacterTextSplitter` 是核心。`chunk_size` 决定每块多大，`chunk_overlap` (重叠) 极其重要，它防止一句话被生硬地从中间切断，保证了上下文的连贯性。

### Day 4: Embedding 与向量库 —— 构建机器人的“海马体”
**核心目标**：让计算机理解文字的“语义距离”。
* **最核心代码**：
    ```python
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    ```
* **原理解析**：Embedding 将文字变成高维空间中的坐标（向量）。FAISS (Facebook AI Similarity Search) 则是存放这些坐标的数据库。查找答案的过程，就是算数学里的“计算坐标点之间的直线距离”。

### Day 5: 检索增强生成 (RAG) 闭环 —— 开卷考试
**核心目标**：检索相关片段，连同问题一起喂给大模型。
* **最核心代码**：
    ```python
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt | llm | StrOutputParser()
    )
    ```
* **原理解析**：极其优雅的逻辑闭环。`RunnablePassthrough()` 接收用户的原始问题，`retriever` 去 FAISS 里找资料，两者合并填入 `rag_prompt`，最后交给大模型（LLM）生成答案。

### Day 6-7: 动态 RAG 全栈集成与部署
**核心目标**：让用户在网页上自由上传 PDF 并聊天。
* **最核心代码** (后端状态管理)：
    ```python
    global global_retriever, global_rag_chain
    ```
* **原理解析**：通过 FastAPI 接收上传的文件，实时生成 FAISS 索引并挂载到全局变量上，使得 `/chat` 接口能动态使用用户刚刚上传的知识库。

---

## 🧠 第二部分：核心知识点凝练

1.  **RAG 的本质是什么？**
    RAG（检索增强生成）相当于给大模型进行**“开卷考试”**。防止它胡编乱造（幻觉），保证它的回答基于你提供的私有资料。
2.  **跨语言检索的痛点 (Embedding 错位)**
    如果你用纯英文的 Embedding 模型（如 `all-MiniLM`）去处理中文 PDF，不管你用中文还是英文提问，检索效果都会很差，因为模型不认识中文字符，匹配会彻底失效。**解决办法是更换为多语言模型（如 `BGE-m3`）**。
3.  **Prompt 工程的“遥控器”效应**
    大模型显得“笨”或“死板”，往往是因为 Prompt 限制得太死。例如加入 *“如果在上下文中找不到，可以结合你的知识回答，但必须说明”*，能瞬间提升机器人的情商和智能感。

---

## 🚧 第三部分：工程实战“踩坑与破局”实录

本周在集成辅助工具（Streamlit 和 Docker）时，我们遭遇了真实开发环境中的经典难题。以下是宝贵的排障经验：

### 1. 前端交互 (Streamlit) 的坑
* **问题**：网页一片空白黑屏，终端显示正常。
    * **破局**：文件未保存 (`Ctrl+S`)，或终端运行的文件名 (`day6_frontend.py`) 与实际代码所在文件 (`day7_frontend.py`) 不匹配。
* **问题**：聊天接口频繁报 `400 Error`。
    * **破局**：后端代码中设置了拦截器。用户必须先点击侧边栏的“上传文件”按钮，让服务器“大脑”里有东西后，才能进行聊天对话。

### 2. 容器化部署 (Docker) 的究极网络折磨
将系统打包进 Docker 集装箱时，遇到了国内开发者最痛的网络环境问题。

* **天坑一：Docker 无法拉取基础镜像，显示 `Timeout` 或 `DeadlineExceeded`。**
    * **原因**：国内网络封锁，且 Docker 运行在虚拟机中，无法直接使用宿主机的代理软件（Clash）。
    * **解决过程**：需要在 Docker Desktop 的 Proxies 设置中手动配置代理。
* **天坑二：配置了代理依然报错 `actively refused` (目标拒绝连接)。**
    * **原因**：
        1. 填错了 IP（不能用 `127.0.0.1`，必须用 Docker 专属暗号 `host.docker.internal`）。
        2. 填错了 Clash 端口（将默认的 `7890` 纠正为了实际的 `7897`）。
        3. 代理配置不完整（遗漏了 HTTPS 栏位的配置，必须 HTTP 和 HTTPS 都配置）。
        4. **关键一步**：Clash 代理软件出于安全原因，默认拦截了外部虚拟机流量。
    * **终极破局方案**：
        打开 Clash 的 **“允许局域网 (Allow LAN)”** 开关。或者直接开启 **“虚拟网卡模式 (TUN 模式)”** 实现全局接管，这样 Docker 甚至都不需要配任何参数就能直接连通外网。

### 3. Docker 的存储魔法
* **现象**：拉取 `python:3.10-slim` 镜像时下载了几百 MB。
* **原理解释**：下载的是一个完整的底层 Linux 操作系统。但因为 Docker 拥有“分层缓存机制”，这几百 MB 永远只存一份在中央仓库。以后再创建多少个项目，都可以直接引用，极大地节省了硬盘空间，同时保证了开发环境的绝对一致性。

---
> **写在最后**：从写下 `FastAPI()` 到看到 `docker-compose up` 亮起绿灯，你已经走完了一个 AI 项目从脚本玩具到工业级部署的完整生命周期。准备迎接 Day 8 更广阔的世界吧！
