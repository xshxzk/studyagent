"""
Day 3: RAG 基础 - 文档加载 & 文本分块学习 (现代 LangChain 修复版)
学习目标：把本地的专业学术 PDF 变成大模型能看懂的“记忆碎片”
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 1. 基础环境配置与网络防线
load_dotenv()
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

# 2. 导入最新版的 LangChain 工具包 (告别红线报错)
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

print("=" * 70)
print("Day 3: RAG 文档加载与文本分块")
print("=" * 70)

# ===== 方式1：简单文本分块演示 =====
print("\n【方式1】基础文本分块 - 理解分块的核心作用")
print("-" * 70)

sample_text = """
AI Agent 是一个能够自主感知环境、做出决策并采取行动的系统。

Agent 的核心能力包括：
1. 感知：从环境中获取信息
2. 推理：基于知识和目标进行思考
3. 行动：执行决策并改变环境

RAG（检索增强生成）是一种结合了信息检索和大语言模型的技术。
它的工作流程是：先从知识库中检索相关信息，然后用这些信息增强 LLM 的生成过程。

这样做的好处是：
- 增强 LLM 的知识，不用微调模型
- 支持实时更新知识库
- 减少 LLM 幻觉
"""

print("\n【分割方式】RecursiveCharacterTextSplitter（递归分割 - 推荐）")
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "，", " "],  # 按优先级递归分割
    chunk_size=60,    # 每块最多 150 个字符
    chunk_overlap=20  # 块之间重叠 30 个字符（防止一句话被生硬切断）
)
chunks = splitter.split_text(sample_text)
print(f"分块数量: {len(chunks)}")
for i, chunk in enumerate(chunks, 1):
    print(f"\n块 {i}:")
    print(f"  长度: {len(chunk)} 字符")
    print(f"  内容: {chunk}")


# ===== 方式2：定义一个多格式文件加载的万能函数 =====
print("\n\n" + "=" * 70)
print("【方式2】通用文件加载器 (实战级代码)")
print("-" * 70)

def load_and_chunk_file(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    通用文件加载和分块函数
    支持: .txt, .pdf, .md
    """
    file_ext = Path(file_path).suffix.lower()
    print(f"\n📄 正在处理文件: {file_path}")
    
    try:
        # 1. 自动识别文件类型并加载
        if file_ext == ".txt" or file_ext == ".md":
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_ext == ".pdf":
            loader = PyPDFLoader(file_path) # 专门用于读取 PDF
        else:
            print(f"❌ 不支持的文件格式: {file_ext}")
            return None
        
        # 将文件吞进内存
        documents = loader.load()
        print(f"✓ 加载成功，该文档共有 {len(documents)} 页")
        
        # 2. 对文档进行分块
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        # split_documents 专门用来处理带页码等元数据的文件对象
        split_docs = splitter.split_documents(documents)
        print(f"✓ 分块成功，被切成了 {len(split_docs)} 个文字块")
        
        return split_docs
    
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        return None

# ===== 🎯 终极实战：读取你的遥感 PDF 论文 =====
print("\n\n" + "=" * 70)
print("【终极实战】读取真实的本地 PDF")
print("-" * 70)

# 👇👇👇 PM 请注意：把这里的名字换成你电脑里那篇英文 PDF 的真实名字！ 👇👇👇
# 请确保这篇 PDF 和当前这个 Python 代码放在同一个文件夹里
my_pdf_file = "2023 - Multi-sensor detection of spring breakup phenology of Canada's lakes.pdf"  

# 检查文件是否存在
if os.path.exists(my_pdf_file):
    # 开始执行切块操作！对于英文学术论文，chunk_size 设为 1000 字符比较合适
    my_chunks = load_and_chunk_file(my_pdf_file, chunk_size=1000, chunk_overlap=100)
    
    if my_chunks:
        print("\n🔍 让我们偷看一下前 3 个数据块长什么样：")
        for i, doc in enumerate(my_chunks[:3], 1):
            print(f"\n--- 第 {i} 块数据 ---")
            # 打印文字内容的前 200 个字符
            print(f"文本内容: {doc.page_content[:200]}...\n")
            # 打印元数据（你能看到它精准记住了这是论文的第几页！）
            print(f"元数据(出处): {doc.metadata}")
else:
    print(f"\n⚠️ 提示：没找到名为 '{my_pdf_file}' 的文件。")
    print("请把你的 PDF 论文拖到这个文件夹里，然后修改代码第 92 行的文件名！")