from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import os

# 加载环境变量
load_dotenv()

# 读取环境变量
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")
model_name = os.getenv("MODEL_NAME",'gpt-5.6-luna')

llm = init_chat_model(
    model=model_name,
    model_provider='openai',
    api_key=api_key,
    base_url=base_url,
    timeout=30,
    max_retries=1
)


response = llm.invoke("什么是RAG？")

print(response.content)