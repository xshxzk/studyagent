"""
RAG Day 3.5: FastAPI + LangChain 文档处理接口 (现代修复版)
"""
import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv

# 1. 基础环境与网络防线
load_dotenv()
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

# 2. 现代版 LangChain 导入 (告别报错)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# 初始化 FastAPI 应用
app = FastAPI(title="GeoAI 文档解析 API", description="将 PDF 转化为大模型记忆碎片")

@app.post("/upload-and-chunk")
async def upload_and_chunk(file: UploadFile = File(...)):
    """接收前端上传的文件，后台自动完成加载与分块，返回处理报告"""
    
    # 提取文件后缀并转为小写 (例如 .pdf)
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    # 将前端传来的文件，安全地存入服务器的临时文件夹
    # 注意：加上 suffix=file_ext 让 Loader 能认出这是 PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # 3. 智能加载器匹配
        if file_ext == '.pdf':
            loader = PyPDFLoader(tmp_path)
        elif file_ext in ['.txt', '.md']:
            loader = TextLoader(tmp_path, encoding="utf-8")
        else:
            return {"error": f"暂不支持 {file_ext} 格式，请上传 PDF 或 TXT。"}
        
        # 执行加载 (把文件吞进内存)
        docs = loader.load()
        
        # 4. 文本分块 (使用针对学术文献推荐的参数)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  
            chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)
        
        # 5. 返回结构化 JSON 报告给前端
        return {
            "status": "success",
            "filename": file.filename,
            "total_pages": len(docs),
            "total_chunks": len(chunks),
            "preview": [
                {
                    "chunk_id": i + 1,
                    "length": len(c.page_content),
                    # 截取前 150 个字作为预览给前端展示
                    "content_preview": c.page_content[:150] + "...",
                    # 【PM重点】：把元数据（比如出自第几页）一起发给前端
                    "metadata": c.metadata 
                }
                for i, c in enumerate(chunks[:3])  # 为了不卡顿，只预览前 3 块
            ]
        }
        
    except Exception as e:
         return {"error": f"处理出错: {str(e)}"}
         
    finally:
        # 6. 清理现场：处理完立刻删掉临时文件，防止服务器硬盘被撑爆！
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    import uvicorn
    # 启动服务器
    uvicorn.run(app, host="127.0.0.1", port=8000)