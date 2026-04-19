from fastapi import FastAPI

# 创建FastAPI应用实例
app = FastAPI(
    title="My First AI Agent API",
    description="学习FastAPI的第一个应用",
    version="1.0.0"
)

# 定义根路由
@app.get("/")
async def read_root():
    """根路径，返回欢迎信息"""
    return {"message": "欢迎来到AI Agent开发世界！", "status": "success"}

# 定义带参数的路由
@app.get("/hello/{name}")
async def say_hello(name: str):
    """问候路由，接受名字参数"""
    return {"message": f"你好，{name}！准备开始AI Agent开发了吗？"}

# 定义POST路由
from pydantic import BaseModel

class UserQuery(BaseModel):
    question: str
    user_id: int = 0

@app.post("/ask")
async def ask_question(query: UserQuery):
    """模拟问答接口"""
    # 这里暂时返回固定回答，后续会接LLM
    answer = f"收到你的问题：{query.question}。我会用AI来回答！"
    return {
        "answer": answer,
        "user_id": query.user_id,
        "timestamp": "2024-01-01T00:00:00Z"
    }

# 新增路由1：GET - 查询用户信息
@app.get("/user/{user_id}")
async def get_user_info(user_id: int):
    """根据用户ID查询用户信息"""
    # 模拟用户数据（实际项目中会从数据库查询）
    users_db = {
        1: {"name": "小明", "email": "xiaoming@example.com", "level": "VIP"},
        2: {"name": "小红", "email": "xiaohong@example.com", "level": "普通"},
        3: {"name": "小刚", "email": "xiaogang@example.com", "level": "高级"}
    }
    
    if user_id in users_db:
        return {
            "user_id": user_id,
            "user_info": users_db[user_id],
            "status": "found"
        }
    else:
        return {
            "user_id": user_id,
            "message": "用户不存在",
            "status": "not_found"
        }

# 新增路由2：POST - 提交用户反馈
class FeedbackRequest(BaseModel):
    user_id: int
    rating: int  # 1-5星评分
    comment: str
    category: str = "general"  # 反馈类别

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """提交用户反馈"""
    # 验证评分范围
    if not (1 <= feedback.rating <= 5):
        return {
            "error": "评分必须在1-5之间",
            "status": "invalid_rating"
        }
    
    # 模拟保存反馈（实际项目中会存数据库）
    feedback_id = f"fb_{feedback.user_id}_{feedback.rating}"
    
    return {
        "feedback_id": feedback_id,
        "message": "感谢您的反馈！",
        "submitted_data": {
            "user_id": feedback.user_id,
            "rating": feedback.rating,
            "comment": feedback.comment,
            "category": feedback.category
        },
        "status": "submitted",
        "timestamp": "2024-01-01T00:00:00Z"
    }

# 运行应用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_demo:app", host="0.0.0.0", port=8000, reload=True)