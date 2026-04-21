from fastapi import FastAPI, HTTPException
# 【核心修改 1】使用最新版 HuggingFace 官方包，避免后续报错
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pydantic import BaseModel

app = FastAPI(title="RAG 检索 API (已升级至最新版)")

# 初始化模型
# 提示：如果你想加载得更快，可以在这前面加上 os.environ["HF_HUB_OFFLINE"] = "1"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 全局变量（实际生产中会使用持久化的数据库目录，比如从本地 load_local 读档）
vector_store = None

class QueryRequest(BaseModel):
    query: str
    k: int = 3

@app.post("/build-index")
def build_index():
    """构建向量索引"""
    global vector_store
    
    # 示例文档
    docs = [
        Document(page_content="AI Agent 是一个能够自主感知环境、做出决策并采取行动的智能系统。", metadata={"id": 1, "topic": "Agent"}),
        Document(page_content="RAG 技术通过检索外部知识库，极大增强了 LLM 回答的准确性。", metadata={"id": 2, "topic": "RAG"}),
        Document(page_content="向量数据库（如 FAISS）专门用于存储和检索高维文本向量。", metadata={"id": 3, "topic": "Database"})
    ]
    
    # 分块 + 向量化 + 存储
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return {"message": f"✓ 成功！索引构建完成，共存入 {len(chunks)} 个向量。"}

@app.post("/search")
def search_similar(request: QueryRequest):
    """相似度搜索"""
    global vector_store
    
    # 防止用户在没建库的情况下直接搜索
    if vector_store is None:
        raise HTTPException(status_code=400, detail="请先调用 /build-index 构建索引库！")
    
    # 【核心修改 2】使用 similarity_search_with_score 获取真实的计算距离
    # FAISS 默认返回的是 L2 距离（欧氏距离的平方），这个值越小，代表文本越相似！
    results_with_scores = vector_store.similarity_search_with_score(request.query, k=request.k)
    
    # 构建结构化的返回结果
    formatted_results = []
    for doc, score in results_with_scores:
        formatted_results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "distance_score": float(score)  # 真实的距离分数
        })
    
    return {
        "query": request.query,
        "results": formatted_results
    }

if __name__ == "__main__":
    import uvicorn
    # 启动服务器
    uvicorn.run(app, host="127.0.0.1", port=8001)