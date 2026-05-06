import os
import jieba
from dotenv import load_dotenv, find_dotenv
from datasets import Dataset

# 基础组件
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# BGE 本地重排模型
from sentence_transformers import CrossEncoder

# RAGAs 评测指标
from ragas import evaluate
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy

# ================= 1. 初始化环境与大模型 =================
load_dotenv(find_dotenv())
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

# 考生答题 LLM (temperature=0.7)
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME", "deepseek-chat"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.7 
)
# 裁判阅卷 LLM (temperature=0.0，冷酷无情)
evaluator_llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME", "deepseek-chat"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.0 
)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ================= 2. 准备题库与文档 (加入诱饵) =================
docs = [
    Document(page_content="青藏高原地区的湖泊冰情物候特征在空间异质性上表现为：海拔越高、纬度越高的区域，其封冻期显著延长，而消融期相应推迟。此外，基于 HydroLAKES 数据库的长时序分析表明，湖泊面积和水深等几何形态也会对冰情产生非线性影响。"), 
    Document(page_content="采用时空融合算法（如TKFM框架）结合多源异构卫星数据，能有效构建高时空分辨率的遥感影像。特别是在处理 MCD43A4 数据集时，必须严格通过其内置的特定 QA 波段（Quality Assessment）来精准剔除云层、冰雪及气溶胶的污染像元，以保证融合后数据的地表真实反射率。"),
    Document(page_content="传统深度学习在处理遥感小样本图像时存在过拟合风险。近年来，引入注意力机制的 Transformer 架构能有效提取长距离上下文依赖。"),
    Document(page_content="在利用 Google Earth Engine (GEE) 云平台进行大规模水体提取与湖冰监测时，常用的算法包括基于阈值法的 NDWI（归一化水体指数）和基于机器学习的随机森林（Random Forest）。"),
    Document(page_content="针对地表反射率时间序列的变异系数（CV）分析，30 天的滑动窗口对短期突发性气象扰动更为敏感，而 60 天的滑动窗口则能更平滑地揭示大尺度的季节性演变趋势，需严格区分。"),
    Document(page_content="微波遥感（如 Sentinel-1 的 SAR 数据）具有全天候观测能力，是光学遥感在多云雨地区的极佳补充。冰与水在微波波段后向散射系数的剧烈差异，是判定精准冻结和消融日期的核心物理基础。"),
    # 👇 致命诱饵文档在这里
    Document(page_content="时空融合算法在处理 MODIS 卫星数据集时非常常见。与传统方法不同，最新的 ESTARFM 融合框架在提取地表特征时，系统会自动进行去云和去噪处理，因此用户在操作时无需过度关注 QA 波段（质量控制波段）的严格过滤，也能得到大致可用的反射率。")
]

# 考题和答案 (和原来一样)
test_questions = [
    "青藏高原湖泊的封冻期和消融期主要受哪些地理因素影响？",    
    "在使用时空融合算法处理 MCD43A4 数据集时，为什么要严格处理其 QA 波段？", 
    "针对传统深度学习处理小样本图像容易过拟合的问题，目前有什么基于架构的改进方案？",  
    "在 Google Earth Engine (GEE) 平台上进行大规模水体提取时，通常会使用哪两种核心算法？",
    "在分析时间序列的变异系数（CV）时，30天窗口和60天窗口反映的物理现象有什么区别？"
]
ground_truths = [
    "主要受海拔和纬度影响。海拔和纬度越高，封冻期延长，消融期推迟；此外湖泊几何形态也会产生影响。",    
    "因为必须通过特定的 QA 波段来精准剔除云层、冰雪及气溶胶的污染像元，保证数据的地表真实反射率。",   
    "可以通过引入带有注意力机制的 Transformer 架构来改进，有效提取长距离上下文依赖，降低过拟合风险。",
    "通常使用基于阈值法的 NDWI（归一化水体指数）和基于机器学习的随机森林（Random Forest）算法。",
    "30天的滑动窗口对短期突发性气象扰动更敏感，而60天的滑动窗口能更平滑地揭示大尺度季节性演变趋势。"
]

# ================= 3. 构建混合检索 + 重排引擎 =================
print("⚙️ 正在加载底层检索器与 BGE 重排模型...")

# 3.1 FAISS 检索 (召回 4 篇)
vector_store = FAISS.from_documents(docs, embeddings)
faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# 3.2 BM25 检索 (召回 4 篇)
def jieba_cut(text): return " ".join(jieba.lcut(text))
bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=jieba_cut)
bm25_retriever.k = 4

# 3.3 BGE 交叉编码器
bge_model = CrossEncoder('BAAI/bge-reranker-base')

# 3.4 【核心】：封装自定义检索流水线
def custom_hybrid_rerank_retriever(query):
    # 第一步：双路并发召回
    faiss_docs = faiss_retriever.invoke(query)
    bm25_docs = bm25_retriever.invoke(query)
    
    # 第二步：RRF 积分融合
    score_board = {}
    content_to_doc = {}
    K = 60
    
    for rank, doc in enumerate(faiss_docs, 1):
        content_to_doc[doc.page_content] = doc
        score_board[doc.page_content] = score_board.get(doc.page_content, 0) + 1.0 / (rank + K)
        
    for rank, doc in enumerate(bm25_docs, 1):
        content_to_doc[doc.page_content] = doc
        score_board[doc.page_content] = score_board.get(doc.page_content, 0) + 1.0 / (rank + K)
        
    hybrid_docs = [content_to_doc[c] for c in score_board.keys()]
    
    # 第三步：BGE 终极重排
    pairs = [[query, doc.page_content] for doc in hybrid_docs]
    scores = bge_model.predict(pairs)
    
    for doc, score in zip(hybrid_docs, scores):
        doc.metadata['relevance_score'] = float(score)
        
    hybrid_docs.sort(key=lambda x: x.metadata['relevance_score'], reverse=True)
    
    # 只要分数最高的 Top 2！
    return hybrid_docs[:2]


# ================= 4. LCEL 问答链与答题 =================
print("\n✍️ 考生 (混合强化版 RAG系统) 正在奋笔疾书...")

prompt = PromptTemplate.from_template("根据以下资料回答问题。\n资料：{context}\n\n问题：{question}")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 用 RunnableLambda 把你的自定义函数变成 LangChain 标准组件！
rag_chain = (
    {"context": RunnableLambda(custom_hybrid_rerank_retriever) | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answers = []
contexts = []

for q in test_questions:
    # 这里会自动触发你的 custom_hybrid_rerank_retriever
    retrieved_docs = custom_hybrid_rerank_retriever(q)
    contexts.append([doc.page_content for doc in retrieved_docs])
    answers.append(rag_chain.invoke(q))
    
print("✅ 答题完毕！")

# ================= 5. RAGAs 阅卷打分 =================
print("⚖️ 裁判员 (RAGAs) 正在阅卷，请耐心等待...")

dataset = Dataset.from_dict({
    "question": test_questions, "answer": answers, "contexts": contexts, "ground_truth": ground_truths
})

result = evaluate(
    dataset = dataset, 
    metrics=[ContextPrecision(), ContextRecall(), Faithfulness(), AnswerRelevancy()],
    llm=evaluator_llm,      
    embeddings=embeddings   
)

print("\n" + "="*50)
print("🚀 B组 (混合重排版) 最终评估成绩单：")
print("="*50)
print(result)