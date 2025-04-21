from langchain.memory import ConversationBufferMemory
from langchain.memory import FileChatMessageHistory
import os
import hashlib

def get_memory(user_host: str, session_id: str, is_new_session: bool = False):
    # 创建存储目录
    os.makedirs("./chat_histories", exist_ok=True)
    
    # 生成唯一会话key
    session_key = hashlib.md5(f"{user_host}_{session_id}".encode()).hexdigest()
    
    # 使用文件存储对话历史
    message_history = FileChatMessageHistory(
        file_path=f"./chat_histories/{session_key}.json"
    )
    
    # 如果是新会话，清空历史记录
    if is_new_session:
        message_history.clear()
    
    return ConversationBufferMemory(
        chat_memory=message_history,
        return_messages=True
    )