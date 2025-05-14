from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
import redis
import json
from logger_config import get_module_logger

# 获取模块日志记录器
logger = get_module_logger("main")

# 加载环境变量
load_dotenv()



# Redis连接池配置
REDIS_POOL = None
def get_redis_connection(retries=3, delay=2):
    """获取Redis连接，支持重试和超时设置"""
    global REDIS_POOL
    
    # 如果连接池已存在且有效，直接使用
    if REDIS_POOL and REDIS_POOL.connection_kwargs.get('password') == os.getenv("REDIS_PASSWORD"):
        conn = redis.Redis(connection_pool=REDIS_POOL)
        try:
            if conn.ping():
                return conn
        except:
            pass
            
    for attempt in range(retries):
        try:
            # 创建或更新连接池
            REDIS_POOL = redis.ConnectionPool(
                host='redis-17542.c323.us-east-1-2.ec2.redns.redis-cloud.com',
                port=17542,
                decode_responses=True,
                username="default",
                password=os.getenv("REDIS_PASSWORD"),  # 从环境变量获取Redis密码
                max_connections=50,
                socket_timeout=15,
                socket_connect_timeout=15,
                health_check_interval=30,
                retry_on_timeout=True
            )
            conn = redis.Redis(connection_pool=REDIS_POOL)
            # 测试连接
            if conn.ping():
                logger.info("Redis连接成功")
                return conn
        except redis.AuthenticationError as e:
            logger.error(f"Redis认证失败，请检查密码是否正确: {str(e)}")
            raise
        except redis.ConnectionError as e:
            if attempt < retries - 1:
                logger.warning(f"Redis连接失败，第{attempt+1}次重试...")
                from time import sleep
                sleep(delay)
                continue
            logger.error(f"Redis连接失败: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Redis连接未知错误: {str(e)}")
            return None
    return None

# 全局存储模式标志
USE_LOCAL_MODE = False

# 尝试连接Redis，如果失败则使用内存存储
try:
    redis_conn = get_redis_connection()
    if not redis_conn:
        logger.warning("Redis连接失败，将使用内存存储作为备选方案")
        # 设置一个标志，表示使用本地模式
        USE_LOCAL_MODE = True
    else:
        USE_LOCAL_MODE = False
except Exception as e:
    logger.error(f"Redis初始化错误: {str(e)}")
    USE_LOCAL_MODE = True
    redis_conn = None

# 初始化LLM
try:
    chat = ChatOpenAI(
    streaming=True,
    model="deepseek-chat",
    temperature=0.2,
    openai_api_base="https://api.deepseek.com/v1"
)
    logger.info("LLM初始化成功")
except Exception as e:
    if "Incorrect API key" in str(e):
        logger.error("API密钥无效，请检查.env文件中的OPENAI_API_KEY配置")
    elif "Connection error" in str(e):
        logger.error(f"无法连接到API服务，请检查网络或API地址: {os.getenv('OPENAI_API_BASE')}")
    else:
        logger.error(f"LLM初始化失败: {str(e)}")
    chat = None

from contextlib import asynccontextmanager

app = FastAPI()

@app.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    user_input = body["message"]
    session_id = body.get("session_id", "default")
    user_host = body.get("userHost", "unknown")
    
    logger.info(f"收到用户请求: {user_input[:50]}...")
    
    # 使用消息历史 - 根据Redis连接状态选择存储方式
    if not USE_LOCAL_MODE:
        try:
            # 使用与/history接口完全相同的Redis连接参数
            host = 'redis-17542.c323.us-east-1-2.ec2.redns.redis-cloud.com'
            port = 17542
            redis_url = f"redis://default:{os.getenv('REDIS_PASSWORD')}@{host}:{port}"
            # 使用与/history接口相同的session_key格式
            session_key = f"{user_host}_{session_id}"
            logger.info(f"使用会话键: {session_key}")
            
            # 检查Redis中是否已存在该会话键
            redis_key = f"message_store:{session_key}"
            try:
                if redis_conn and redis_conn.exists(redis_key):
                    logger.info(f"在Redis中找到已存在的会话: {session_key}")
                else:
                    logger.info(f"在Redis中创建新会话: {session_key}")
            except Exception as e:
                logger.warning(f"检查Redis会话时出错: {str(e)}")
                
            message_history = RedisChatMessageHistory(
                url=redis_url,
                session_id=session_key
            )
            logger.info(f"为用户 {user_host} 创建Redis会话 {session_id}")
            # 记录Redis键名，便于调试
            redis_key = f"message_store:{session_key}"
            logger.info(f"Redis存储键: {redis_key}")
        except Exception as e:
            logger.error(f"创建Redis会话历史失败: {str(e)}")
            # 失败时回退到内存存储
            from langchain_community.chat_message_histories import ChatMessageHistory
            message_history = ChatMessageHistory()
            logger.warning(f"为用户 {user_host} 创建内存会话 {session_id}")
    else:
        # 本地模式 - 使用内存存储
        from langchain_community.chat_message_histories import ChatMessageHistory
        message_history = ChatMessageHistory()
        logger.info(f"本地模式: 为用户 {user_host} 创建内存会话 {session_id}")
        
    # 创建带记忆的对话链
    memory = ConversationBufferMemory(
        chat_memory=message_history,
        return_messages=True
    )
    conversation = ConversationChain(
        llm=chat,
        verbose=True,
        memory=memory
    )

    async def event_stream():
        try:
            logger.info(f"开始处理用户输入: {user_input[:50]}...")
            # 检查LLM是否初始化成功
            if chat is None:
                logger.error("LLM模型初始化失败")
                yield f"data: 系统错误：语言模型初始化失败，请联系管理员。\n\n"
                return
                
            # 直接使用ConversationChain处理用户输入
            try:
                # 先添加用户消息到历史记录
                message_history.add_user_message(user_input)
                
                # 使用ConversationChain处理用户输入
                response = conversation.predict(input=user_input)
                
                # 添加AI响应到历史记录
                message_history.add_ai_message(response)
                
                # 记录消息添加情况
                logger.info(f"已将用户消息和AI响应添加到历史记录，当前历史记录长度: {len(message_history.messages)}")
                
                yield f"data: {response}\n\n"
            except Exception as e:
                logger.error(f"处理请求时出错: {str(e)}")
                yield f"data: [ERROR] 处理您的请求时出现错误，请稍后再试。\n\n"
                
        except Exception as e:
            logger.error(f"处理请求时出错: {str(e)}")
            yield f"data: [ERROR] 处理您的请求时出现错误，请稍后再试。\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
    )

# 在应用启动时检查Redis连接状态
if not USE_LOCAL_MODE and redis_conn:
    try:
        if redis_conn.ping():
            print("Redis连接成功")
            # 打印Redis连接参数，便于调试
            print(f"Redis连接参数: host={REDIS_POOL.connection_kwargs.get('host')}, port={REDIS_POOL.connection_kwargs.get('port')}")
            # 打印环境变量中的Redis配置
            print(f"环境变量Redis配置: REDIS_HOST={os.getenv('REDIS_HOST')}, REDIS_PORT={os.getenv('REDIS_PORT')}")
        else:
            print("Redis连接失败，但应用将继续在本地模式下运行")
    except Exception as e:
        print(f"Redis连接测试失败: {str(e)}")
        print("应用将在本地模式下运行")
else:
    print("应用将在本地模式下运行，不使用Redis")

@app.post("/clear-redis")
@app.get("/clear-redis")
async def clear_redis_data():
    """
    清空Redis中的所有数据
    """
    try:
        global USE_LOCAL_MODE, redis_conn
        
        # 检查是否处于本地模式
        if USE_LOCAL_MODE:
            logger.warning("当前处于本地模式，无法清空Redis数据")
            return {"status": "error", "message": "当前处于本地模式，无法清空Redis数据"}
            
        # 检查Redis连接状态
        if not redis_conn or not redis_conn.ping():
            logger.warning("Redis连接不可用，尝试重新连接...")
            redis_conn = get_redis_connection()
            if not redis_conn:
                return {"status": "error", "message": "无法连接到Redis服务器"}
        
        # 获取所有键并删除
        all_keys = redis_conn.keys("*")
        if all_keys:
            logger.info(f"正在删除 {len(all_keys)} 个Redis键")
            redis_conn.delete(*all_keys)
            logger.info("Redis数据已清空")
            return {"status": "success", "message": f"已成功删除 {len(all_keys)} 个键"}
        else:
            logger.info("没有找到需要删除的Redis键")
            return {"status": "success", "message": "Redis中没有任何数据需要删除"}
    except Exception as e:
        logger.error(f"清空Redis数据失败: {str(e)}")
        return {"status": "error", "message": f"清空Redis数据失败: {str(e)}"}

@app.get("/redis-data")
async def get_redis_data():
    """
    获取Redis中的所有数据并以易读格式返回
    """
    try:
        global USE_LOCAL_MODE, redis_conn
        
        # 检查是否处于本地模式
        if USE_LOCAL_MODE:
            logger.warning("当前处于本地模式，无法获取Redis数据")
            return {"status": "error", "message": "当前处于本地模式，无法获取Redis数据"}
            
        # 检查Redis连接状态
        if not redis_conn or not redis_conn.ping():
            logger.warning("Redis连接不可用，尝试重新连接...")
            redis_conn = get_redis_connection()
            if not redis_conn:
                return {"status": "error", "message": "无法连接到Redis服务器"}
        
        # 获取所有键
        all_keys = redis_conn.keys("*")
        logger.info(f"在Redis中找到 {len(all_keys)} 个键")
        
        # 获取每个键的类型和值
        result = {}
        for key in all_keys:
            # 处理键名，确保是字符串
            key_str = key.decode('utf-8') if isinstance(key, bytes) else str(key)
            
            # 获取键类型并确保是字符串
            key_type_raw = redis_conn.type(key)
            key_type = key_type_raw.decode('utf-8') if isinstance(key_type_raw, bytes) else str(key_type_raw)
            
            # 根据键类型获取值
            if key_type == "string":
                value = redis_conn.get(key)
                result[key_str] = {"type": key_type, "value": value}
            elif key_type == "list":
                values = redis_conn.lrange(key, 0, -1)
                result[key_str] = {"type": key_type, "value": values, "length": len(values)}
            elif key_type == "hash":
                values = redis_conn.hgetall(key)
                result[key_str] = {"type": key_type, "value": values}
            elif key_type == "set":
                values = redis_conn.smembers(key)
                result[key_str] = {"type": key_type, "value": list(values)}
            elif key_type == "zset":
                values = redis_conn.zrange(key, 0, -1, withscores=True)
                result[key_str] = {"type": key_type, "value": values}
            else:
                result[key_str] = {"type": key_type, "value": "未知类型数据"}
        
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"获取Redis数据失败: {str(e)}")
        return {"status": "error", "message": f"获取Redis数据失败: {str(e)}"}

@app.get("/history")
async def get_history(session_id: str, user_host: str = "unknown"):
    """
    获取指定会话的历史聊天记录
    """
    try:
        global USE_LOCAL_MODE, redis_conn
        
        # 详细记录当前存储模式
        logger.info(f"当前存储模式: {'本地内存' if USE_LOCAL_MODE else 'Redis'}")
        
        # 检查Redis连接状态
        if not USE_LOCAL_MODE and not redis_conn:
            logger.warning("Redis连接不可用，自动切换到本地模式")
            USE_LOCAL_MODE = True
            
        # 根据存储模式获取消息历史
        if not USE_LOCAL_MODE:
            try:
                # 确保使用与/chat接口完全相同的Redis连接参数
                host = 'redis-17542.c323.us-east-1-2.ec2.redns.redis-cloud.com'
                port = 17542
                redis_url = f"redis://default:{os.getenv('REDIS_PASSWORD')}@{host}:{port}"
                logger.info(f"尝试连接Redis: {redis_url}")
                
                if not redis_conn or not redis_conn.ping():
                    logger.info("Redis连接不可用，尝试重新连接...")
                    redis_conn = get_redis_connection()
                    if not redis_conn:
                        raise Exception("Redis连接失败")
                    logger.info("Redis重新连接成功")
                else:
                    logger.info("Redis连接正常")
                
                # 使用与/chat接口完全相同的session_key格式
                session_key = f"{user_host}_{session_id}"
                logger.info(f"使用会话键: {session_key}")
                
                # 检查Redis中是否存在该会话数据
                redis_key = f"message_store:{session_key}"
                if redis_conn.exists(redis_key):
                    logger.info(f"在Redis中找到会话 {session_key} 的数据")
                    
                    # 获取Redis中的原始数据
                    raw_data = redis_conn.lrange(redis_key, 0, -1)
                    
                    # 解析JSON格式的消息
                    messages = []
                    for item in raw_data:
                        try:
                            data = json.loads(item)
                            if isinstance(data, dict) and 'type' in data:
                                if data['type'] == 'human':
                                    content = data.get('data', {}).get('content', '')
                                    messages.append({"type": "human", "content": content})
                                elif data['type'] == 'ai':
                                    content = data.get('data', {}).get('content', '')
                                    messages.append({"type": "ai", "content": content})
                        except json.JSONDecodeError as e:
                            logger.warning(f"解析JSON消息失败: {str(e)}")
                            continue
                    
                    logger.info(f"成功解析 {len(messages)} 条消息")
                    return {"status": "success", "messages": messages}
                else:
                    logger.warning(f"未在Redis中找到会话 {session_key} 的数据")
                    return {"status": "error", "message": "未找到会话历史数据"}
                
            except Exception as e:
                logger.error(f"获取Redis历史记录失败: {str(e)}")
                return {"status": "error", "message": f"获取历史记录失败: {str(e)}"}
        else:
            # 本地模式 - 返回空数组
            logger.info("本地模式: 返回空历史记录")
            return {"status": "success", "messages": []}
            
    except Exception as e:
        logger.error(f"获取历史记录时发生未知错误: {str(e)}")
        return {"status": "error", "message": f"系统错误: {str(e)}"}

    except Exception as e:
        logger.error(f"获取历史记录失败: {str(e)}")
        return {"status": "error", "message": f"获取历史记录失败: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)