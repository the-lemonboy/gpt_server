from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.chains import ConversationChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import os
from dotenv import load_dotenv
import redis

# 加载环境变量
load_dotenv()

app = FastAPI()

# Redis连接配置 - 修改为官方推荐方式
redis_conn = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    decode_responses=True,
    username="default",
    password=os.getenv("REDIS_PASSWORD")
)

# 初始化LLM
chat = ChatOpenAI(
    streaming=True,
    model="deepseek-chat",
    temperature=0.2
)

def load_pdf_to_vectorstore():
    """直接加载pdf_files/demo.pdf文件到向量数据库"""
    try:
        pdf_path = "pdf_files/demo.pdf"
        if not os.path.exists(pdf_path):
            raise FileNotFoundError("默认PDF文件不存在")
            
        pdf_reader = PdfReader(pdf_path)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages])
        if not text.strip():
            raise ValueError("PDF内容为空或无法提取")
            
        # 分割文本
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(text)
        
        # 创建嵌入向量
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        # 存储到向量数据库
        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        # 验证向量数据库是否存储成功
        test_docs = vectorstore.similarity_search("测试", k=1)
        if not test_docs or not test_docs[0].page_content.strip():
            raise RuntimeError("向量数据库存储失败")
            
        return vectorstore
    except Exception as e:
        print(f"加载PDF失败: {str(e)}")
        return None

@app.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    user_input = body["message"]
    session_id = body.get("session_id", "default")
    user_host = body.get("userHost", "unknown")

    # 加载向量数据库
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # 如果chroma_db不存在，自动加载demo.pdf
    if not os.path.exists("./chroma_db"):
        vectorstore = load_pdf_to_vectorstore()
        if not vectorstore:
            return StreamingResponse(
                iter(["data: 无法加载默认PDF文件\n\n"]),
                media_type="text/event-stream"
            )
    else:
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
    
    # 创建检索链
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    # 保留聊天历史
    message_history = RedisChatMessageHistory(
        session_id=f"{user_host}_{session_id}",
        url=f"redis://default:{os.getenv('REDIS_PASSWORD')}@{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}"
    )

    async def event_stream():
        try:
            # 检查向量数据库是否已加载内容
            if not os.path.exists("./chroma_db"):
                yield f"data: 请先上传PDF文档以初始化知识库。\n\n"
                return
                
            # 检查是否是询问PDF内容的请求
            if "介绍" in user_input or "内容" in user_input or "summary" in user_input.lower():
                docs = vectorstore.similarity_search("", k=3)
                summary = "\n".join([f"• {doc.page_content[:200]}..." for doc in docs])
                yield f"data: 以下是PDF文档的主要内容摘要：\n{summary}\n\n"
                return
                
            # 检查问题是否与PDF内容相关
            relevant_docs = vectorstore.similarity_search(user_input, k=1)
            if not relevant_docs or relevant_docs[0].page_content.strip() == "":
                yield f"data: 抱歉，我只能回答与上传PDF文档内容相关的问题。\n\n"
                return
                
            # 计算相似度分数
            scores = vectorstore.similarity_search_with_score(user_input, k=1)
            if not scores or scores[0][1] > 0.8:  # 放宽相似度阈值
                yield f"data: 抱歉，这个问题与PDF内容关联度不足，请询问与文档更相关的问题。\n\n"
                return
                
            response = await qa_chain.acall({"query": user_input})
            yield f"data: {response['result']}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        chunk_size=8192  # 增加缓冲区大小
    )

# 在Redis连接配置后立即测试
if redis_conn.ping():
    print("Redis连接成功")
else:
    print("Redis连接失败")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)