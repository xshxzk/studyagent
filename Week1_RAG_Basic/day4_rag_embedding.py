"""
Day 4: RAG 核心 - Embedding & 向量存储学习 (已升级至 LangChain 最新版)
学习目标：理解文本向量化原理，掌握向量存储和检索
"""

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
# 【核心修改 1】使用最新的 langchain_huggingface 官方包
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
import os

print("=" * 70)
print("Day 4: RAG Embedding & 向量存储")
print("=" * 70)

# ===== 方式1：理解 Embedding 基础 =====
print("\n【方式1】Embedding 基础概念演示")
print("-" * 70)

# 创建示例文本
texts = [
    "AI Agent 是一个能够自主感知环境、做出决策并采取行动的系统。",
    "Agent 的核心能力包括感知、推理和行动。",
    "RAG（检索增强生成）是一种结合了信息检索和大语言模型的技术。",
    "RAG 的工作流程是：先从知识库中检索相关信息，然后用这些信息增强 LLM 的生成过程。",
    "这样做的好处是增强 LLM 的知识，不用微调模型，支持实时更新知识库。",
    "向量数据库是 RAG 的核心组件，用于存储和检索文本向量。",
    "Embedding 模型将文本转换为高维向量，相似文本的向量距离近。",
    "FAISS 和 Chroma 是常用的向量数据库，支持高效的相似度搜索。"
]

print("\n示例文本:")
for i, text in enumerate(texts, 1):
    print(f"{i}. {text}")

# 【核心修改 2】使用新的 HuggingFaceEmbeddings 类初始化模型
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print(f"\n✓ Embedding 模型加载成功")
print(f"  - 模型: all-MiniLM-L6-v2")
# 注意：这里不再硬编码 384，因为后续通过代码会自动打印实际维度

# 生成向量
vectors = embeddings.embed_documents(texts)
print(f"✓ 向量化完成，共 {len(vectors)} 个向量")
print(f"  - 第一个向量维度: {len(vectors[0])}")
print(f"  - 向量范围: [{min(vectors[0]):.3f}, {max(vectors[0]):.3f}]")

# 计算相似度
def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# 比较前两个文本的相似度
sim_1_2 = cosine_similarity(vectors[0], vectors[1])
sim_1_3 = cosine_similarity(vectors[0], vectors[2])

print(f"\n相似度计算:")
print(f"  - 文本1 vs 文本2: {sim_1_2:.3f} (都关于 Agent)")
print(f"  - 文本1 vs 文本3: {sim_1_3:.3f} (Agent vs RAG)")

# ===== 方式2：向量存储 - FAISS =====
print("\n\n" + "=" * 70)
print("【方式2】向量存储 - FAISS")
print("-" * 70)

# 创建 Document 对象（LangChain 标准格式）
documents = [
    Document(page_content=text, metadata={"id": i, "topic": "AI Agent" if i <= 2 else "RAG"})
    for i, text in enumerate(texts, 1)
]

print(f"✓ 创建 {len(documents)} 个 Document 对象")

# 使用 FAISS 构建向量数据库
faiss_store = FAISS.from_documents(documents, embeddings)

print(f"✓ FAISS 向量数据库创建成功")
print(f"  - 存储向量数量: {faiss_store.index.ntotal}")

# 执行相似度搜索
query = "什么是 AI Agent？"
print(f"\n🔍 查询: '{query}'")

# 搜索最相似的 3 个文档
results = faiss_store.similarity_search(query, k=3)

print(f"✓ 检索结果 (Top 3):")
for i, doc in enumerate(results, 1):
    similarity = cosine_similarity(
        embeddings.embed_query(query),
        embeddings.embed_query(doc.page_content)
    )
    print(f"  {i}. 相似度: {similarity:.3f}")
    print(f"     内容: {doc.page_content}")
    print(f"     元数据: {doc.metadata}")

# ===== 方式3：向量存储 - Chroma =====
print("\n\n" + "=" * 70)
print("【方式3】向量存储 - Chroma")
print("-" * 70)

# 使用 Chroma 构建向量数据库
chroma_store = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")

print(f"✓ Chroma 向量数据库创建成功")
print(f"  - 持久化目录: ./chroma_db")

# 执行相似度搜索
query2 = "RAG 有什么优势？"
print(f"\n🔍 查询: '{query2}'")

results2 = chroma_store.similarity_search(query2, k=3)

print(f"✓ 检索结果 (Top 3):")
for i, doc in enumerate(results2, 1):
    similarity = cosine_similarity(
        embeddings.embed_query(query2),
        embeddings.embed_query(doc.page_content)
    )
    print(f"  {i}. 相似度: {similarity:.3f}")
    print(f"     内容: {doc.page_content}")

# ===== 方式4：对比 FAISS vs Chroma =====
print("\n\n" + "=" * 70)
print("【方式4】FAISS vs Chroma 对比分析")
print("-" * 70)

comparison_data = {
    "特性": ["安装复杂度", "内存占用", "检索速度", "持久化", "云端支持", "元数据过滤", "推荐场景"],
    "FAISS": ["中等", "低", "极快", "需要额外配置", "不支持", "不支持", "高性能检索"],
    "Chroma": ["简单", "中等", "快", "自动", "部分支持", "支持", "原型开发"]
}

print("\n对比表格:")
print(f"{'特性':<15} {'FAISS':<12} {'Chroma':<12}")
print("-" * 40)
for i, feature in enumerate(comparison_data["特性"]):
    faiss_val = comparison_data["FAISS"][i]
    chroma_val = comparison_data["Chroma"][i]
    print(f"{feature:<15} {faiss_val:<12} {chroma_val:<12}")

print("\n💡 选择建议:")
print("  - FAISS：生产环境，高性能，大规模数据")
print("  - Chroma：开发原型，易用，功能丰富")

# ===== 方式5：完整 RAG 管道预览 =====
print("\n\n" + "=" * 70)
print("【方式5】完整 RAG 管道 (Day 3 + Day 4)")
print("-" * 70)

# 模拟完整的 RAG 流程
def simple_rag_pipeline(query, documents, embeddings_model, vector_store):
    """简化的 RAG 管道"""
    
    # 1. 文本分块 (Day 3)
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"✓ 步骤1: 文本分块完成，共 {len(chunks)} 块")
    
    # 2. 向量化存储 (Day 4)
    if vector_store == "faiss":
        db = FAISS.from_documents(chunks, embeddings_model)
    else:
        db = Chroma.from_documents(chunks, embeddings_model)
    print(f"✓ 步骤2: 向量存储完成，使用 {vector_store}")
    
    # 3. 检索相关文档
    retrieved_docs = db.similarity_search(query, k=2)
    print(f"✓ 步骤3: 检索完成，找到 {len(retrieved_docs)} 个相关文档")
    
    # 4. 构建上下文 (模拟)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    print(f"✓ 步骤4: 上下文构建完成，长度 {len(context)} 字符")
    
    # 5. 生成回答 (这里用模拟，实际会调用 LLM)
    answer = f"基于检索到的信息，我的回答是：[模拟] {query} 的答案..."
    print(f"✓ 步骤5: 回答生成完成")
    
    # 返回包含各步骤结果的字典
    return {
        "query": query,
        "retrieved_docs": [doc.page_content for doc in retrieved_docs],
        "context": context,
        "answer": answer
    }

# 测试完整管道
print("\n🔄 运行完整 RAG 管道测试:")
test_query = "AI Agent 的核心能力是什么？"
result = simple_rag_pipeline(test_query, documents, embeddings, "faiss")

print(f"\n查询: {result['query']}")
print(f"检索文档数: {len(result['retrieved_docs'])}")
print(f"生成的回答: {result['answer']}")

# ===== 方式6：持久化与加载 =====
print("\n\n" + "=" * 70)
print("【方式6】向量数据库持久化")
print("-" * 70)

# FAISS 持久化
faiss_store.save_local("./faiss_index")
print("✓ FAISS 索引已保存到 ./faiss_index")

# 加载 FAISS
loaded_faiss = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
print("✓ FAISS 索引加载成功")

# Chroma 已经自动持久化
print("✓ Chroma 已自动持久化到 ./chroma_db")

# 测试加载后的检索
test_results = loaded_faiss.similarity_search("Agent 能力", k=1)
print(f"✓ 加载后检索测试: {test_results[0].page_content[:50]}...")

# ===== 总结 =====
print("\n\n" + "=" * 70)
print("Day 4 学习总结")
print("=" * 70)

print("""
✅ 今天学到的核心概念：

1. 【Embedding 原理】：文本 → 向量，相似文本向量距离近

2. 【向量存储】：
   - FAISS：高性能，适合生产
   - Chroma：易用，适合开发

3. 【相似度搜索】：余弦相似度，找最相关的文档

4. 【RAG 管道】：分块 → 向量化 → 存储 → 检索 → 生成

5. 【持久化】：保存和加载向量数据库

📌 下一步（Day 5）：
  - 学习检索增强生成：把检索结果喂给 LLM
  - 学习 Prompt 工程：如何构造 RAG 的提示词
  - 完成端到端 RAG 问答系统

💡 实践任务：
  1. 找一个真实的文档（PDF 或 TXT）
  2. 用今天的代码完成分块 + 向量化 + 存储
  3. 测试不同的查询，观察检索效果
  4. 比较 FAISS 和 Chroma 的性能差异
""")

# # 清理演示文件 (为了保持你的工程目录干净)
# import shutil
# if os.path.exists("./faiss_index"):
#     shutil.rmtree("./faiss_index")
# if os.path.exists("./chroma_db"):
#     shutil.rmtree("./chroma_db")
# print(f"\n✓ 清理演示文件完成")
# 测试不同类型的查询
test_queries = [
    "什么是 AI Agent？",
    "RAG 的优势是什么？",
    "向量数据库的作用？"
]

for q in test_queries:
    results = faiss_store.similarity_search(q, k=1)
    print(f"\n查询: {q}")
    for doc in results:
        print(f"  - {doc.page_content[:50]}...")