import os
from dotenv import load_dotenv, find_dotenv
from datasets import Dataset

# 导入你的 RAG 核心组件 (这里用最基础的 FAISS 和 LLM)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 导入 RAGAs 评测四大指标
from ragas import evaluate
# 导入 RAGAs 评测四大指标的“类 (Class)”
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerRelevancy, 
)
# ================= 1. 初始化环境与基础组件 =================
load_dotenv(find_dotenv())
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

# 初始化大模型 (假设你用的是 DeepSeek 或 OpenAI)
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME", "deepseek-chat"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.0 
)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 准备那 3 篇文档并构建基础 FAISS 检索器
docs = [
    Document(page_content="青藏高原地区的湖泊冰情物候特征在空间异质性上表现为：海拔越高、纬度越高的区域，其封冻期显著延长，而消融期相应推迟。此外，基于 HydroLAKES 数据库的长时序分析表明，湖泊面积和水深等几何形态也会对冰情产生非线性影响。"), 
    Document(page_content="采用时空融合算法（如TKFM框架）结合多源异构卫星数据，能有效构建高时空分辨率的遥感影像，极大提升了地表动态特征的监测精度。特别是在处理 MCD43A4 数据集时，必须严格通过其内置的特定 QA 波段（Quality Assessment）来精准剔除云层、冰雪及气溶胶的污染像元，以保证融合后数据的地表真实反射率。"),
    Document(page_content="传统深度学习在处理遥感小样本图像时存在过拟合风险。近年来，引入注意力机制的 Transformer 架构能有效提取长距离上下文依赖。结合大语言模型（LLM）的检索增强生成（RAG）技术，研究人员正探索将多模态数据与遥感先验知识图谱深度融合，以实现地学知识的自动化问答与智能解译。"),
    Document(page_content="在利用 Google Earth Engine (GEE) 云平台进行大规模水体提取与湖冰监测时，常用的算法包括基于阈值法的 NDWI（归一化水体指数）和基于机器学习的随机森林（Random Forest）。通过编写 GEE 脚本配合本地的 Python 自动化调度，可以高效完成 TB 级卫星影像的批量下载与预处理。"),
    Document(page_content="针对地表反射率时间序列的变异系数（Coefficient of Variation, CV）分析，统计绘图窗口大小的选择对结果解释至关重要。最新的实验校正指出，30 天的滑动窗口对短期突发性气象扰动更为敏感，而 60 天的滑动窗口则能更平滑地揭示大尺度的季节性演变趋势，在分析时需严格区分两者的物理意义。"),
    Document(page_content="微波遥感（如 Sentinel-1 的 SAR 数据）具有全天候、全天时的观测能力，能够穿透云层，是光学遥感（如 Landsat-8/9 或 Sentinel-2）在多云雨地区的极佳补充。在湖冰冻融周期的监测中，冰与水在微波波段后向散射系数的剧烈差异，是判定精准冻结和消融日期的核心物理基础。"),
    # 诱饵文档（专门用来骗 FAISS 的）
    Document(page_content="时空融合算法在处理 MODIS 卫星数据集时非常常见。与传统方法不同，最新的 ESTARFM 融合框架在提取地表特征时，系统会自动进行去云和去噪处理，因此用户在操作时无需过度关注 QA 波段（质量控制波段）的严格过滤，也能得到大致可用的反射率。")
]
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2}) # 基础版只拿前 2 篇

# ================= 2. 手搓题库 (进阶版) =================
test_questions = [
    # 基础概念题
    "青藏高原湖泊的封冻期和消融期主要受哪些地理因素影响？",    
    # 细节干扰题（考验系统能否精准抓取 QA 波段处理细节）
    "在使用时空融合算法处理 MCD43A4 数据集时，为什么要严格处理其 QA 波段？", 
    # 深度学习理论题
    "针对传统深度学习处理小样本图像容易过拟合的问题，目前有什么基于架构的改进方案？",  
    # 平台与工程实现题
    "在 Google Earth Engine (GEE) 平台上进行大规模水体提取时，通常会使用哪两种核心算法？",
    # 统计学参数辨析题（极度考验大模型的防幻觉能力）
    "在分析时间序列的变异系数（CV）时，30天窗口和60天窗口反映的物理现象有什么区别？"
]

ground_truths = [
    "主要受海拔和纬度影响。海拔和纬度越高的区域，湖泊封冻期显著延长，消融期推迟；此外，湖泊的几何形态（如面积和水深）也会产生非线性影响。",    
    "因为必须通过特定的 QA 波段来精准剔除云层、冰雪及气溶胶的污染像元，这样才能保证融合后数据的地表真实反射率。",   
    "可以通过引入带有注意力机制的 Transformer 架构来改进，因为它能有效提取长距离上下文依赖，降低过拟合风险。",
    "通常使用基于阈值法的 NDWI（归一化水体指数）算法和基于机器学习的随机森林（Random Forest）算法。",
    "30天的滑动窗口对短期突发性气象扰动更为敏感，而60天的滑动窗口则能更平滑地揭示大尺度的季节性演变趋势。"
]

answers = []
contexts = []

# ================= 3. 让 RAG 系统开始“考试答题” =================
print("✍️ 考生 (RAG系统) 正在奋笔疾书答题中...")

# 构建一个极其简单的 LCEL 问答链
prompt = PromptTemplate.from_template("根据以下资料回答问题。\n资料：{context}\n\n问题：{question}")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 循环答题
for q in test_questions:
    # 1. 记录系统搜回来的资料 (Contexts)
    retrieved_docs = retriever.invoke(q)
    # RAGAs 要求 contexts 必须是字符串列表格式：["文档1的内容", "文档2的内容"]
    doc_contents = [doc.page_content for doc in retrieved_docs]
    contexts.append(doc_contents)
    
    # 2. 记录系统最终生成的回答 (Answer)
    ans = rag_chain.invoke(q)
    answers.append(ans)
    
print("✅ 答题完毕！准备交卷！\n")

# ================= 4. 整理试卷，交给 RAGAs 裁判打分 =================
print("⚖️ 裁判员 (RAGAs) 正在阅卷，请耐心等待 (约需1-2分钟)...")

# 必须按 RAGAs 规定的字典格式拼装
data = {
    "question": test_questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}

# 转换成 HuggingFace Dataset 格式
dataset = Dataset.from_dict(data)

# 开始打分！必须带上 () 来初始化这些指标对象
result = evaluate(
    dataset = dataset, 
    metrics=[
        ContextPrecision(),
        ContextRecall(),
        Faithfulness(),
        AnswerRelevancy(),
    ],
    llm=llm,      # <--- 把无情的冷血裁判塞进去
    embeddings=embeddings
)

print("\n" + "="*50)
print("🏆 A组 (FAISS 基础版) 最终评估成绩单：")
print("="*50)
# result 本身就是一个包含了各维度分数的字典
print(result)