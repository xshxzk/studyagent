"""
Day 6: RAG + FastAPI 端到端 Web 服务
学习目标：将本地 RAG 脚本封装成标准的 HTTP API 接口
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ================= 1. 基础配置与 FastAPI 初始化 =================
load_dotenv()
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

app = FastAPI(
    title="Geospatial RAG API",
    description="一个基于 LangChain 和 FastAPI 的文档问答后端服务",
    version="1.0"
)

# ================= 2. 全局变量与系统启动事件 =================
# 在工业级开发中，我们通常会在服务器启动时加载模型和数据库，而不是每次请求都加载
rag_chain = None
retriever = None

@app.on_event("startup")
async def startup_event():
    """服务器启动时自动运行的初始化任务"""
    global rag_chain, retriever
    print("🚀 正在启动服务并初始化 RAG 引擎...")

    # 1. 初始化模型
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_NAME", "deepseek-chat"),
        temperature=0.1,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL")
    )

    # 2. 构建基础知识库（实际生产中这里应该是 load_local 读取你之前处理好的硬盘库）
    knowledge_base = [
        "AI Agent 是一个能够自主感知环境、做出决策并采取行动的智能系统。",
        "Agent 的核心能力包括：感知、推理、行动。",
        "RAG 是一种结合信息检索和大语言模型的技术，可以降低幻觉。",
        "使用 FastAPI 可以非常方便地将大模型能力暴露为 RESTful API。"
    ]
    docs = [Document(page_content=text) for text in knowledge_base]
    
    # 分块并存入临时 FAISS 库
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # 3. 组装 LCEL 黄金管道
    rag_prompt = PromptTemplate.from_template("""
    你是一个专业的 AI 助手。请严格基于以下上下文回答问题。
    
    上下文：
    {context}
    
    问题：{question}
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    print("✅ RAG 引擎初始化完毕！API 准备就绪。")


# ================= 3. 定义数据模型 (API 的输入输出格式) =================
class ChatRequest(BaseModel):
    query: str

class SourceDoc(BaseModel):
    content: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]

# ================= 4. 编写核心对话接口 =================
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    接收用户问题，经过 RAG 系统处理后返回结构化答案。
    """
    if rag_chain is None or retriever is None:
        raise HTTPException(status_code=500, detail="RAG 引擎未初始化完成")

    try:
        # 第一步：先单独拿一次检索结果，为了返回给前端展示引用来源
        source_docs = retriever.invoke(request.query)
        formatted_sources = [SourceDoc(content=doc.page_content) for doc in source_docs]

        # 第二步：调用 LCEL 链生成最终回答
        # 注意这里用了 ainvoke (异步调用)，这是在高并发 Web 服务中防止阻塞的关键！
        answer = await rag_chain.ainvoke(request.query)

        # 第三步：拼装 JSON 返回给前端
        return ChatResponse(
            answer=answer,
            sources=formatted_sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"大模型处理出错: {str(e)}")

# 如果你想直接在 VS Code 里运行这个脚本
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)