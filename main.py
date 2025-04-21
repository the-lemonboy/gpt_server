from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from memory_store import get_memory
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

app = FastAPI()

chat = ChatOpenAI(streaming=True, model="deepseek-chat", temperature=0.7)

@app.post("/chat")
async def chat_api(request: Request):
    try:
        body = await request.json()
        session_id = body.get("session_id", "")
        user_host = body.get("user_host", "")  # 修改为userHost
        user_input = body["message"]
        
        # 这里可以添加session_id和user_host的处理逻辑
        # 为每个请求创建新的memory和chain实例
        memory = get_memory(user_host, session_id)
        chain = ConversationChain(
            llm=chat,
            memory=memory,
            verbose=True
        )

        async def event_stream():
            try:
                response = await chain.acall({"input": user_input})
                yield f"data: {response['response']}\n\n"
            except Exception as e:
                yield f"data: [ERROR] {str(e)}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Connection": "keep-alive"}
        )
    except Exception as e:
        return {"error": str(e)}