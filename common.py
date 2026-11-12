from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI
import psycopg
from pgvector.psycopg import register_vector
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGEngine,PGVectorStore

# 加载配置文件
load_dotenv()

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://rag:rag@localhost:5433/rag",
)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

EMBEDDING_OPENAI_API_KEY = os.environ.get("EMBEDDING_OPENAI_API_KEY")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")

EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", 1024))

CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-5.6-luna")

BASE_URL = os.environ.get("BASE_URL")

EMBEDDING_BASE_URL = os.environ.get('EMBEDDING_BASE_URL')

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not OPENAI_API_KEY or not EMBEDDING_BASE_URL:
    raise RuntimeError("请先在 .env 中设置 API_KEY(问答和嵌入)")

#加载大模型实例(嵌入大模型)
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=EMBEDDING_OPENAI_API_KEY,
    base_url=EMBEDDING_BASE_URL,
     check_embedding_ctx_length=False,  # 关键修复
)


# 问答大模型
chat_client = OpenAI(
    api_key= OPENAI_API_KEY,
    base_url=BASE_URL
)


def get_db_connection() -> psycopg.Connection:
    """建立连接，并注册 pgvector 的 Python 类型适配器。"""
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn

# 全局变量
pg_engine = None
vector_store = None


async def store_documents(splits):
    global pg_engine, vector_store

    if pg_engine is None:
        pg_engine = PGEngine.from_connection_string(
            url=DATABASE_URL
        )

        await pg_engine.ainit_vectorstore_table(
            table_name="lc_documents",
            vector_size=EMBEDDING_DIM
        )

    if vector_store is None:
        vector_store = await PGVectorStore.create(
            engine=pg_engine,
            table_name="lc_documents",
            embedding_service=embeddings
        )

    await vector_store.aadd_documents(splits)





# def embed_texts(texts: list[str]) -> list[list[float]]:
#     """批量生成 embedding，返回顺序与输入 texts 保持一致。"""
#     if not texts:
#         return []

#     response = embed_client.embeddings.create(
#         model=EMBEDDING_MODEL,
#         input=texts,
#     )
#     vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

#     if any(len(vector) != EMBEDDING_DIM for vector in vectors):
#         actual_dim = len(vectors[0]) if vectors else 0
#         raise ValueError(
#             f"embedding 维度不匹配：配置为 {EMBEDDING_DIM}，实际为 {actual_dim}。"
#             "请检查 EMBEDDING_MODEL、EMBEDDING_DIM 和数据库 vector 列。"
#         )

#     return vectors