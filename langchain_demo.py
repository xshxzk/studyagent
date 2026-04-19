"""
Day 2: LangChain核心概念学习 (更新版 - 使用现代LCEL语法)
学习目标：理解LLM、Prompt、Chain等核心概念，掌握LCEL管道式调用
"""
import os
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
from langchain_openai import ChatOpenAI  # 使用ChatOpenAI，更稳定
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser  # 用于解析输出
from langchain_core.runnables import RunnablePassthrough  # 用于传递数据


# 初始化LLM (推荐用ChatOpenAI替代OpenAI)
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME"),         # 自动读取 "deepseek-chat"
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY"),   # 自动读取 "sk-..."
    openai_api_base=os.getenv("OPENAI_BASE_URL")  # 核心：把请求地址指向 DeepSeek 服务器
)

# # ===== 方式1：直接调用LLM（最简单）=====
# print("=" * 50)
# print("方式1: 直接调用LLM")
# print("=" * 50)

# # 直接问一个问题
# response = llm.invoke("什么是AI Agent？")
# print(f"LLM回答: {response.content}\n")

# # ===== 方式2：使用Prompt模板 =====
# print("=" * 50)
# print("方式2: 使用Prompt模板")
# print("=" * 50)

# # 定义一个Prompt模板
# prompt_template = PromptTemplate(
#     input_variables=["topic", "language"],
#     template="用{language}解释什么是{topic}。只需要一句话。"
# )

# # 生成提示词
# prompt = prompt_template.format(topic="RAG", language="中文")
# print(f"生成的提示词: {prompt}")

# response = llm.invoke(prompt)
# print(f"LLM回答: {response.content}\n")

# # ===== 方式3：使用LCEL简单链（替代LLMChain）=====
# print("=" * 50)
# print("方式3: 使用LCEL简单链（现代推荐方式）")
# print("=" * 50)

# # 创建提示词模板
# prompt = PromptTemplate(
#     input_variables=["topic"],
#     template="你是一个AI专家。用三句话解释{topic}的核心概念。"
# )

# # 使用LCEL管道：prompt -> llm -> 解析输出
# chain = prompt | llm | StrOutputParser()

# # 运行链并获取结果
# result = chain.invoke({"topic": "大语言模型"})
# print(f"LLM回答:\n{result}\n")

# # ===== 方式4：使用LCEL管道（最佳实践）=====
# print("=" * 50)
# print("方式4: 使用LCEL管道（最现代的方式）")
# print("=" * 50)

# # 定义提示词模板
# prompt = PromptTemplate(
#     input_variables=["question"],
#     template="作为AI助手，请回答这个技术问题：{question}"
# )

# # 使用LCEL的管道操作符 |，添加输出解析
# chain = prompt | llm | StrOutputParser()

# # 运行
# result = chain.invoke({"question": "FastAPI有什么优势？"})
# print(f"LLM回答:\n{result}\n")

# # ===== 方式5：多步链（链式组合）=====
# print("=" * 50)
# print("方式5: 多步链（组合多个LLM调用）")
# print("=" * 50)

# # 第一步：生成学习计划
# prompt1 = PromptTemplate(
#     input_variables=["topic"],
#     template="为学习{topic}创建一个3天的学习计划。"
# )

# # 第二步：详细说明第一天的内容
# prompt2 = PromptTemplate(
#     input_variables=["plan"],
#     template="这是学习计划：\n{plan}\n\n现在详细解释第一天应该做什么？"
# )

# # 组合链：先执行prompt1|llm，输出作为plan，再传给prompt2|llm
# chain = (
#     {"plan": prompt1 | llm | StrOutputParser()}
#     | RunnablePassthrough.assign(plan=lambda x: x["plan"])  # 传递plan
#     | prompt2
#     | llm
#     | StrOutputParser()
# )

# # 运行
# result = chain.invoke({"topic": "LangChain框架"})
# print(f"详细计划:\n{result}\n")

# ===== 方式6：添加工具调用（进阶扩展）=====
print("=" * 50)
print("方式6: 添加工具调用（进阶扩展）")
print("=" * 50)

from langchain_core.tools import tool

# 定义一个简单工具（计算器）
@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except:
        return "计算错误"

# 将工具绑定到LLM
llm_with_tools = llm.bind_tools([calculator])

# 创建带工具的链
prompt_with_tool = PromptTemplate(
    input_variables=["query"],
    template="请回答问题，如果需要计算就用工具：{query}"
)

chain_with_tool = prompt_with_tool | llm_with_tools

# 运行（工具调用会自动处理）
response = chain_with_tool.invoke({"query": "计算 2 + 3 * 4"})
print(f"带工具的回答: {response.content}")
if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"工具调用: {tool_call['name']}({tool_call['args']})")
        result = calculator.invoke(tool_call['args'])
        print(f"工具结果: {result}")