from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
import hashlib
import redis

def test_redis_connection(url: str):
    """测试Redis连接是否成功"""
    try:
        r = redis.Redis.from_url(url)
        return r.ping()
    except Exception as e:
        return False

def get_memory(user_host: str, session_id: str, is_new_session: bool = False):
    # 生成唯一会话key
    session_key = hashlib.md5(f"{user_host}_{session_id}".encode()).hexdigest()
    
    redis_url = "redis://redis-17542.c323.us-east-1-2.ec2.redns.redis-cloud.com:17542"
    if not test_redis_connection(redis_url):
        raise ConnectionError("无法连接到Redis服务器")
    
    # 使用Redis存储对话历史
    message_history = RedisChatMessageHistory(
        url=redis_url,
        ttl=3600,
        session_id=session_key
    )
    
    # 如果是新会话，清空历史记录
    if is_new_session:
        message_history.clear()
    
    return ConversationBufferMemory(
        chat_memory=message_history,
        return_messages=True
    )