"""
Day 9: 混合检索与重排 (纯净教学版)
学习目标：不用高级框架，一步步看懂数据是怎么流动的
"""

import os
import jieba
from dotenv import load_dotenv, find_dotenv

# 导入的基础包（绝对安全，不会报 ModuleNotFoundError）
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import CohereRerank



# ================= 第一步：环境与数据准备 =================
load_dotenv(find_dotenv())
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

# 1. 准备模型
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. 准备三段遥感学术资料
docs = [
    Document(page_content="青藏高原地区的湖泊冰情物候特征在空间异质性上表现为：海拔越高、纬度越高的区域，其封冻期显著延长，而消融期相应推迟。"),
    Document(page_content="采用时空融合算法（如TKFM框架）结合多源异构卫星数据，能有效构建高时空分辨率的遥感影像，极大提升了地表动态特征的监测精度。"),
    Document(page_content="传统深度学习在处理小样本图像时存在过拟合风险，引入注意力机制的 Transformer 架构能有效提取长距离上下文依赖。")
]

# 3. 用户的问题
query = "湖冰物候受啥影响？"
print(f"\n🙋‍♂️ 用户提问: 【{query}】\n")
print("-" * 50)


# ================= 第二步：分别进行两次独立的检索 =================

# 1. FAISS 向量检索（懂语义，但可能忽略具体生僻字）
vector_store = FAISS.from_documents(docs, embeddings)
faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
faiss_results = faiss_retriever.invoke(query)

print("🔍 FAISS 搜回来的前 3 名：")
for i, d in enumerate(faiss_results):
    print(f"  第{i+1}名: {d.page_content[:20]}...")

# 2. BM25 关键词检索（不懂语义，但死抠字眼，需要 jieba 帮忙切分中文）
def jieba_cut(text):
    return " ".join(jieba.lcut(text))

bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=jieba_cut)
bm25_retriever.k = 3
bm25_results = bm25_retriever.invoke(query)

print("\n🔍 BM25 搜回来的前 3 名：")
for i, d in enumerate(bm25_results):
    print(f"  第{i+1}名: {d.page_content[:20]}...")


# ================= 第三步：手动实现 RRF (倒数秩融合) 算法 =================
print("\n" + "-" * 50)
print("🧠 开始进行 RRF 积分融合...")

# 准备一个计分板
score_board = {}
# 准备一个字典，用来存文档内容对应的原始 Document 对象
content_to_doc = {}

# 定义一个常数 k（工业界一般取 60，用来平滑分数）
K = 60

# 给 FAISS 的结果打分
for rank, doc in enumerate(faiss_results, start=1): # rank 是排名: 1, 2, 3
    content = doc.page_content
    if content not in score_board:
        score_board[content] = 0.0
        content_to_doc[content] = doc
    # RRF 核心公式：加分 = 1 / (排名 + 60)
    score_board[content] += 1.0 / (rank + K)

# 给 BM25 的结果打分
for rank, doc in enumerate(bm25_results, start=1):
    content = doc.page_content
    if content not in score_board:
        score_board[content] = 0.0
        content_to_doc[content] = doc
    score_board[content] += 1.0 / (rank + K)

# 按照总分从高到低，把文档重新排个序
sorted_contents = sorted(score_board.keys(), key=lambda x: score_board[x], reverse=True)
hybrid_docs = [content_to_doc[c] for c in sorted_contents]

print(f"✅ 融合完成！合并去重后共有 {len(hybrid_docs)} 篇文档。")


# ================= 第四步：送给 Cohere 大模型做终极重排 =================
print("\n" + "-" * 50)
print("👑 正在呼叫 Cohere 进行大模型精度重排 (只要最准的 1 篇)...")

reranker = CohereRerank(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    model="rerank-multilingual-v3.0",
    top_n=1  # 只要 Top 1
)

# 压缩/重排文档
final_docs = reranker.compress_documents(hybrid_docs, query=query)

print("\n🎯 最终给用户的答案背景资料：")
for i, d in enumerate(final_docs, 1):
    score = d.metadata.get('relevance_score', '无')
    print(f"  [Top {i}] (Cohere打分: {score}) -> {d.page_content}")
print("=" * 50)