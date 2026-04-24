"""
Day 8: 高阶 RAG - Query Transformation (查询改写与 HyDE 实战)
学习目标：手撕多路查询扩展 (Multi-Query) 和 假设性文档嵌入 (HyDE) 的底层逻辑
"""

import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ================= 1. 基础环境与模型初始化 =================
load_dotenv(find_dotenv())
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

# 初始化大模型 (稍微提高一点 temperature，让模型在改写问题时更有创造力)
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME", "deepseek-chat"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.7 
)

# 初始化 Embedding
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ================= 2. 模拟构建本地知识库 =================
# 这里我们故意放几句非常书面化、极其严谨的学术长句。
docs = [
    Document(page_content="青藏高原地区的湖泊冰情物候特征在空间异质性上表现为：海拔越高、纬度越高的区域，其封冻期显著延长，而消融期相应推迟。"),
    Document(page_content="采用时空融合算法（如TKFM框架）结合多源异构卫星数据，能有效构建高时空分辨率的遥感影像，极大提升了地表动态特征的监测精度。"),
    Document(page_content="传统深度学习在处理小样本图像时存在过拟合风险，引入注意力机制的 Transformer 架构能有效提取长距离上下文依赖。")
]
# 快速建库
vector_store = FAISS.from_documents(docs, embeddings)

print("=" * 70)
print("Day 8: RAG 检索进阶 —— 查询改写 (Query Transformation)")
print("=" * 70)

# 用户非常简短、随意的原始提问
user_query = "湖冰物候受啥影响？"
print(f"\n🙋‍♂️ 用户的原始短问题: 【{user_query}】\n")


# ================= 3. 基础 RAG 检索 (反面教材) =================
print("❌ 【测试一】基础 Naive RAG 直接检索")
# 用户的短问题和论文的严谨长句在向量空间里差异很大，往往搜不出最好的结果
naive_results = vector_store.similarity_search_with_score(user_query, k=1)
print(f"基础检索命中的文档 (距离得分: {naive_results[0][1]:.4f}):")
print(f"  -> {naive_results[0][0].page_content}")


# ================= 4. 高级策略 A：Multi-Query (多路扩展) =================
print("\n" + "-" * 70)
print("✨ 【高级策略 A】Multi-Query 多路召回扩展")

# 1. 写一个针对“提问改写”的 Prompt
multi_query_prompt = PromptTemplate(
    input_variables=["question"],
    template="""你是一个遥感与空间数据分析领域的资深专家。用户的原始查询可能太简短或太口语化。
    请将用户的以下原始问题，改写并扩展为 3 个不同角度、表述更严谨的学术问题，以便我们在论文数据库中进行检索。
    请直接分行输出这 3 个问题，不要带有任何编号和废话。
    
    原始问题: {question}"""
)

# 2. 构建 LCEL 数据流：Prompt -> 大模型 -> 字符串提取 -> 按换行符切分成 Python 列表
multi_query_chain = (
    multi_query_prompt 
    | llm 
    | StrOutputParser() 
    | (lambda x: [q.strip() for q in x.split('\n') if q.strip()])
)

# 3. 执行多路改写
generated_queries = multi_query_chain.invoke({"question": user_query})
print("🧠 大模型为你扩展的 3 个严谨检索词：")
for i, q in enumerate(generated_queries, 1):
    print(f"   [{i}] {q}")

print("\n🔍 分别对 3 个扩展问题进行检索并打分：")
# 4. 【核心修改】：通过循环，逐个核对扩展问题的检索效果
all_results = []
for i, q in enumerate(generated_queries, 1):
    # 对每一个扩展后的学术问题进行独立检索
    res = vector_store.similarity_search_with_score(q, k=1)
    doc, score = res[0]
    
    # 将结果存入列表，方便后续可能的去重处理
    all_results.append(doc.page_content)
    
    print(f"   👉 扩展词 [{i}] 命中结果 (距离得分: {score:.4f}):")
    print(f"      内容预览: {doc.page_content[:60]}...")
# 4. 工程进阶：去重处理（虽然本例只有一条正确答案，但在处理长论文时非常重要）
unique_contents = list(set(all_results))
print(f"\n✅ 多路召回完成，合并去重后共获得 {len(unique_contents)} 条相关背景资料。")
print("-" * 70)


# ================= 5. 高级策略 B：HyDE (假设性文档嵌入) =================
print("\n" + "-" * 70)
print("✨ 【高级策略 B】HyDE (假设性文档嵌入盲猜法)")

# 1. 写一个让大模型“胡编乱造”的 Prompt
hyde_prompt = PromptTemplate(
    input_variables=["question"],
    template="""请写一段简短的学术论文摘要或段落，来回答以下问题。
    这段文字将用于向量数据库的相似度检索，所以请尽量使用专业的学术词汇（如空间异质性、高程、纬度等）。
    直接输出正文，不要有“本文探讨了”之类的开场白。
    
    问题: {question}"""
)

# 2. 构建 HyDE 的 LCEL 数据流
hyde_chain = hyde_prompt | llm | StrOutputParser()

# 3. 让大模型生成“假论文”
fake_document = hyde_chain.invoke({"question": user_query})
print(f"🧠 大模型盲猜生成的“假学术段落”:\n   {fake_document}\n")

# 4. 用这段“假论文”去 FAISS 里搜“真论文”
# 因为假论文的行文风格、专业词汇和真论文极其相似，所以距离得分会非常漂亮！
hyde_results = vector_store.similarity_search_with_score(fake_document, k=1)
print(f"🎯 HyDE 策略检索命中的真实文档 (距离得分: {hyde_results[0][1]:.4f}):")
print(f"  -> {hyde_results[0][0].page_content}")
print("=" * 70)
