"""
Day 6: RAG + FastAPI+ PDF 端到端 Web 服务
学习目标：将本地 RAG 脚本封装成标准的 HTTP API 接口
"""
import os
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# ================= 1. 基础配置与 FastAPI 初始化 =================
load_dotenv()
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

app = FastAPI(
    title="动态文档 RAG 助手",
    description="一个基于给定pdf的 LangChain 和 FastAPI 的文档问答后端服务",
    version="1.0"
)
# ================= 2. 核心全局变量 =================
# 【核心逻辑】：用这两个全局变量来保存用户刚刚上传处理好的知识库
global_retriever = None
global_rag_chain = None

# 把大模型和 Embedding 模型提出来作为全局单例（因为它们不需要随着文件改变而改变）
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME", "deepseek-chat"),
    temperature=0.1,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL")
)

rag_prompt = PromptTemplate.from_template("""
请优先基于以下上下文回答。如果在上下文中找不到答案，你可以结合自身的知识进行解答，但必须明确告知用户该部分内容非文档原文。

上下文：
{context}

问题：{question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ================= 3. 数据模型 =================
class ChatRequest(BaseModel):
    query: str

class SourceDoc(BaseModel):
    content: str
    metadata: dict

class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]

# ================= 4. 接口 A：处理上传的文件 =================
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """接收上传的文件，构建 FAISS 库，并更新全局 RAG 链"""
    global global_retriever, global_rag_chain
    # 提取文件后缀并转为小写 (例如 .pdf)
    file_ext = os.path.splitext(file.filename)[1].lower()
    # 1. 存入临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        # 2. 智能加载器匹配
        if file_ext == '.pdf':
            loader = PyPDFLoader(tmp_path)
        elif file_ext in ['.txt', '.md']:
            loader = TextLoader(tmp_path, encoding="utf-8")
        else:
            raise HTTPException(status_code=400, detail="仅支持 PDF, TXT, MD 格式")
        docs = loader.load()
       
        # 3. 文本分块
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        
        # 4. 构建 FAISS 向量库并生成检索器
        vector_store = FAISS.from_documents(chunks, embeddings_model)
        global_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # 5. 【核心步骤】：基于新的检索器，重新组装 LCEL 管道！
        global_rag_chain = (
            {"context": global_retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        return {
            "status": "success",
            "message": f"文件 {file.filename} 解析完毕！共生成 {len(chunks)} 个记忆碎片，RAG 引擎已就绪。"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件处理出错: {str(e)}")
    finally:
        # 6. 清理现场
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ================= 5. 接口 B：问答聊天 =================
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    接收用户问题，使用刚刚生成的全局 RAG 链回答
    """
    global global_retriever, global_rag_chain
    
    # 【拦截机制】：如果用户还没上传过文件，直接拦截
    if global_rag_chain is None or global_retriever is None:
        raise HTTPException(status_code=400, detail="大脑空空！请先调用 /api/upload 接口上传文档。")

    try:
        # 第一步：提取来源文档（给前端展示用）
        source_docs = global_retriever.invoke(request.query)
        formatted_sources = [
            SourceDoc(content=doc.page_content, metadata=doc.metadata) 
            for doc in source_docs
        ]

        # 第二步：调用大模型生成答案
        answer = await global_rag_chain.ainvoke(request.query)

        return ChatResponse(
            answer=answer,
            sources=formatted_sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"大模型处理出错: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)