import streamlit as st
import requests

# ================= 1. 页面与侧边栏设置 =================
st.set_page_config(page_title="GeoAI 智能助手", page_icon="🤖", layout="centered")

with st.sidebar:
    st.image("https://api.dicebear.com/7.x/bottts/svg?seed=GeoAI", width=100) 
    st.title("⚙️ 控制台")
    st.markdown("---")
    
    # 【新增UI】：文件上传区域
    st.markdown("### 📄 1. 知识库构建")
    uploaded_file = st.file_uploader("请上传你要提问的 PDF 或 TXT 论文", type=["pdf", "txt", "md"])
    
    if st.button("🚀 开始解析并注入大脑", use_container_width=True):
        if uploaded_file is not None:
            with st.spinner("正在努力阅读并提取记忆碎片..."):
                try:
                    # 组装文件数据，发送 POST 请求给 FastAPI 的 /api/upload 接口
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post("http://backend:8000/api/upload", files=files, timeout=60)
                    
                    if response.status_code == 200:
                        st.success("✅ " + response.json().get("message", "文件解析成功！现在可以提问了。"))
                    else:
                        st.error(f"❌ 解析失败: {response.json().get('detail', '未知错误')}")
                except requests.exceptions.ConnectionError:
                    st.error("🚨 无法连接到后端！请检查 FastAPI 是否在运行。")
        else:
            st.warning("⚠️ 请先在上方选择一个文件！")
            
    st.markdown("---")
    st.markdown("### 💬 2. 对话管理")
    if st.button("🗑️ 清空对话历史", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 GeoAI 智能知识库助手")

# ================= 2. 状态初始化 =================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= 3. 渲染历史对话 =================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 查看参考来源"):
                for i, src in enumerate(message["sources"], 1):
                    st.info(f"**[{i}]** {src['content']}")

# ================= 4. 处理用户输入 =================
if prompt := st.chat_input("请提问，例如：这篇论文研究了什么？"):
    
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("正在从本地知识库检索并思考..."):
            try:
                # 请求 FastAPI 的 /api/chat 接口
                response = requests.post(
                    "http://127.0.0.1:8000/api/chat",
                    json={"query": prompt},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "大模型没有返回答案")
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("📚 查看参考来源 (点击展开)"):
                            for i, src in enumerate(sources, 1):
                                st.info(f"**[{i}]** {src['content']}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                elif response.status_code == 400:
                    st.warning("⚠️ " + response.json().get("detail", "请求错误"))
                else:
                    st.error(f"⚠️ 服务器返回错误码: {response.status_code}")
            
            except requests.exceptions.ConnectionError:
                st.error("🚨 无法连接到后端！请检查你的 FastAPI 是否还在运行。")