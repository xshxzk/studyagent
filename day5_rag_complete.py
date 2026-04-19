"""
Day 5: RAG 完整系统 - 检索增强生成学习
学习目标：构建端到端的 RAG 问答系统，掌握检索 + 生成的完整流程
"""

import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 加载环境变量
load_dotenv()
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

print("=" * 70)
print("Day 5: RAG 完整系统 - 检索增强生成")
print("=" * 70)

# ===== 方式1：理解 RAG 核心流程 =====
print("\n【方式1】RAG 核心流程演示")
print("-" * 70)

# 1. 准备知识库文档
knowledge_base = [
    "AI Agent 是一个能够自主感知环境、做出决策并采取行动的智能系统。",
    "Agent 的核心能力包括：感知（从环境中获取信息）、推理（基于知识进行思考）、行动（执行决策）。",
    "RAG（Retrieval-Augmented Generation）是一种结合信息检索和大语言模型的技术。",
    "RAG 的工作流程：用户提问 → 检索相关文档 → 构建上下文 → LLM 生成回答。",
    "RAG 的优势：增强知识准确性、支持实时更新、降低幻觉风险。",
    "LangChain 是构建 RAG 系统的优秀框架，提供完整的工具链。",
    "FastAPI 可以快速构建 RAG 的 Web API 接口。",
    "向量数据库如 FAISS、Chroma 用于高效存储和检索文本向量。",
    "Embedding 模型将文本转换为向量，相似文本的向量距离近。",
    "Prompt 工程在 RAG 中非常重要，需要合理组织检索结果和用户查询。"
]

print("✓ 知识库文档准备完成，共", len(knowledge_base), "条")

# 2. 文档加载和分块
documents = [Document(page_content=text) for text in knowledge_base]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

print(f"✓ 文本分块完成，共 {len(chunks)} 块")

# 3. 向量化存储
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)

print(f"✓ 向量存储完成，共 {vector_store.index.ntotal} 个向量")

# 4. 初始化 LLM
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo"),
    temperature=0.1,  # 降低随机性，提高准确性
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL")
)

print("✓ LLM 初始化完成")

# ===== 方式2：基础 RAG 实现 =====
print("\n\n" + "=" * 70)
print("【方式2】基础 RAG 实现")
print("-" * 70)

# RAG Prompt 模板
rag_prompt = PromptTemplate.from_template("""
你是一个专业的 AI 助手。请基于以下提供的上下文信息回答用户的问题。

上下文信息：
{context}

用户问题：{question}

请按照以下要求回答：
1. 直接基于上下文信息回答，不要添加外部知识
2. 如果上下文信息不足以回答，请明确说明
3. 回答要简洁明了，重点突出
4. 保持客观中立的语气

回答：
""")

def basic_rag_query(question, k=3):
    """基础 RAG 查询函数"""
    
    # 检索相关文档
    retrieved_docs = vector_store.similarity_search(question, k=k)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # 构建提示词
    prompt = rag_prompt.format(context=context, question=question)
    
    # LLM 生成回答
    response = llm.invoke(prompt)
    
    return {
        "question": question,
        "retrieved_docs": [doc.page_content for doc in retrieved_docs],
        "context": context,
        "answer": response.content,
        "num_docs": len(retrieved_docs)
    }

# 测试基础 RAG
test_question = "什么是 AI Agent？"
result = basic_rag_query(test_question)

print(f"🔍 查询: {result['question']}")
print(f"📄 检索到 {result['num_docs']} 个相关文档")
print(f"🤖 回答: {result['answer']}")

# ===== 方式3：使用 LangChain LCEL 构建 RAG 链 =====
print("\n\n" + "=" * 70)
print("【方式3】LangChain LCEL RAG 链")
print("-" * 70)

# 检索器
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# RAG 链：检索 → 格式化上下文 → LLM 生成
def format_docs(docs):
    """格式化检索到的文档"""
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# 测试 LCEL RAG 链
test_question2 = "RAG 有什么优势？"
answer2 = rag_chain.invoke(test_question2)

print(f"🔍 查询: {test_question2}")
print(f"🤖 LCEL 回答: {answer2}")

# ===== 方式4：RAG 性能对比 =====
print("\n\n" + "=" * 70)
print("【方式4】RAG vs 直接 LLM 对比")
print("-" * 70)

def direct_llm_query(question):
    """直接用 LLM 回答（无检索）"""
    prompt = f"请回答以下问题：{question}"
    response = llm.invoke(prompt)
    return response.content

# 对比测试
test_questions = [
    "什么是 AI Agent？",
    "LangChain 是什么框架？",
    "向量数据库的作用是什么？"
]

print("对比结果:")
print(f"{'问题':<20} {'直接LLM':<30} {'RAG系统':<30}")
print("-" * 80)

for q in test_questions:
    direct_answer = direct_llm_query(q)[:25] + "..."
    rag_answer = rag_chain.invoke(q)[:25] + "..."
    
    print(f"{q:<20} {direct_answer:<30} {rag_answer:<30}")

print("\n💡 观察:")
print("  - RAG 回答更基于事实，更准确")
print("  - 直接 LLM 可能产生幻觉或泛化回答")

# ===== 方式5：RAG 评估指标 =====
print("\n\n" + "=" * 70)
print("【方式5】RAG 系统评估")
print("-" * 70)

def evaluate_rag(question, expected_answer):
    """简化的 RAG 评估"""
    
    # 获取 RAG 回答
    rag_answer = rag_chain.invoke(question)
    
    # 检索评估：计算检索到的文档是否包含关键信息
    retrieved_docs = retriever.get_relevant_documents(question)
    retrieval_score = 1.0 if any(expected_answer.lower() in doc.page_content.lower() for doc in retrieved_docs) else 0.0
    
    # 生成评估：检查回答是否包含预期关键词
    generation_score = 1.0 if any(word in rag_answer.lower() for word in expected_answer.lower().split()) else 0.0
    
    return {
        "question": question,
        "expected": expected_answer,
        "rag_answer": rag_answer,
        "retrieval_score": retrieval_score,
        "generation_score": generation_score,
        "overall_score": (retrieval_score + generation_score) / 2
    }

# 评估测试
eval_result = evaluate_rag(
    question="什么是 RAG？",
    expected_answer="检索增强生成"
)

print("评估结果:")
print(f"  - 问题: {eval_result['question']}")
print(f"  - 预期答案: {eval_result['expected']}")
print(f"  - RAG 回答: {eval_result['rag_answer'][:50]}...")
print(f"  - 检索得分: {eval_result['retrieval_score']}")
print(f"  - 生成得分: {eval_result['generation_score']}")
print(f"  - 总体得分: {eval_result['overall_score']}")

# ===== 方式6：RAG 优化技巧 =====
print("\n\n" + "=" * 70)
print("【方式6】RAG 优化技巧")
print("-" * 70)

print("🔧 检索优化:")
print("  - 调整 chunk_size: 太小丢失上下文，太大检索不准")
print("  - 增加 overlap: 防止信息在块边界丢失")
print("  - 使用 rerank: 对检索结果重新排序")
print("  - 多查询策略: 用不同方式表达查询")

print("\n🔧 生成优化:")
print("  - 改进 prompt: 明确指令和格式要求")
print("  - 控制温度: 降低随机性提高准确性")
print("  - 添加示例: few-shot learning")
print("  - 后处理: 过滤或改写回答")

# 演示优化效果
optimized_retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # 增加检索数量

optimized_rag_chain = (
    {"context": optimized_retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

test_optimized = optimized_rag_chain.invoke("AI Agent 的核心能力是什么？")
print(f"\n🎯 优化后回答: {test_optimized}")

# ===== 总结 =====
print("\n\n" + "=" * 70)
print("Day 5 学习总结")
print("=" * 70)

print("""
✅ 今天学到的核心概念：

1. 【RAG 完整流程】：查询 → 检索 → 上下文构建 → LLM 生成 → 回答

2. 【Prompt 工程】：如何组织检索结果和用户查询
   - 系统指令明确角色
   - 上下文信息清晰标注
   - 回答要求具体可操作

3. 【LangChain LCEL】：用管道操作符 | 构建复杂链
   - RunnablePassthrough 传递数据
   - 自定义函数处理文档

4. 【性能评估】：检索质量 + 生成质量
   - 检索：是否找到相关信息
   - 生成：回答的准确性和相关性

5. 【优化策略】：检索参数调优 + Prompt 改进

📌 下一步（Day 6）：
  - 学习 FastAPI 集成：将 RAG 做成 Web 服务
  - 学习高级 RAG：多查询、rerank 等
  - 部署到生产环境

💡 实践任务：
  1. 用自己的文档测试 RAG 系统
  2. 尝试不同的 prompt 模板，观察效果
  3. 调整检索参数，优化回答质量
  4. 实现简单的评估指标
""")

print("\n🎉 Day 5 完成！你已经掌握了 RAG 的核心技术栈！")