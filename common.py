from __future__ import annotations

import os
from collections.abc import Iterable

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import psycopg
from pgvector.psycopg import register_vector
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGEngine,PGVectorStore
from sqlalchemy import text
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import asyncio

# 加载配置文件
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
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
llm = init_chat_model(
    api_key= OPENAI_API_KEY,
    base_url=BASE_URL,
    model=CHAT_MODEL,
    temperature=30,
    timeout=30
)


def get_db_connection() -> psycopg.Connection:
    """建立连接，并注册 pgvector 的 Python 类型适配器。"""
    conn = psycopg.connect(_sync_database_url(DATABASE_URL))
    register_vector(conn)
    return conn


pg_engine = None
vector_store = None

TABLE_NAME = "lc_documents"
SCHEMA_NAME = "public"


def _async_database_url(url: str) -> str:
    """将同步 PostgreSQL URL 转成 PGEngine 可用的异步 URL。"""
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://") :]
    if url.startswith("postgres://"):
        return "postgresql+asyncpg://" + url[len("postgres://") :]
    return url


def _sync_database_url(url: str) -> str:
    """将 SQLAlchemy 异步 URL 转成 psycopg 可用的连接 URL。"""
    if url.startswith("postgresql+asyncpg://"):
        return "postgresql://" + url[len("postgresql+asyncpg://") :]
    if url.startswith("postgresql+psycopg://"):
        return "postgresql://" + url[len("postgresql+psycopg://") :]
    return url


async def _table_exists(engine, table_name: str, schema_name: str = SCHEMA_NAME) -> bool:
    """使用 to_regclass 检查表是否存在（不抛异常）。"""
    async def check() -> bool:
        async with engine._pool.connect() as conn:
            result = await conn.execute(
                text("SELECT to_regclass(:full_name)"),
                {"full_name": f"{schema_name}.{table_name}"},
            )
            return result.scalar_one_or_none() is not None

    # PGEngine.from_connection_string() owns a background event loop.  The
    # pool must be used on that loop as well.
    return await engine._run_as_async(check())


def _validate_split_options(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须大于等于 0 且小于 chunk_size")


def load_and_split_documents(
    directory: str | os.PathLike[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """递归加载指定目录下的 .txt / .md / .pdf 文件并切分。"""
    _validate_split_options(chunk_size, chunk_overlap)
    directory = str(directory)
    loaders = [
        DirectoryLoader(
            directory,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
            silent_errors=True,
        ),
        DirectoryLoader(
            directory,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
            silent_errors=True,
        ),
        DirectoryLoader(
            directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            silent_errors=True,
        ),
    ]

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )
    return splitter.split_documents(docs)


async def store_documents(
    directory: str | os.PathLike[str] | Iterable | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    *,
    splits: Iterable | None = None,
):
    """切分并将文档向量写入 PostgreSQL，或直接写入已有切片。"""
    global pg_engine, vector_store

    # 兼容旧调用：store_documents(splits) 和 store_documents(splits=splits)。
    if splits is None and directory is not None and not isinstance(
        directory, (str, bytes, os.PathLike)
    ):
        splits = directory
        directory = None

    if splits is not None and directory is not None:
        raise ValueError("directory 和 splits 只能传一个")

    if splits is None:
        if directory is None:
            raise ValueError("必须提供 directory 或 splits")
        splits = load_and_split_documents(directory, chunk_size, chunk_overlap)
    else:
        splits = list(splits)

    if not splits:
        source = directory if directory is not None else "splits"
        print(f"[store_documents] {source} 下没有可处理的文档")
        return []

    # 1) 初始化引擎（单例）
    if pg_engine is None:
        pg_engine = PGEngine.from_connection_string(
            url=_async_database_url(DATABASE_URL)
        )

    # 2) 仅在表不存在时创建（幂等检查）
    if not await _table_exists(pg_engine, TABLE_NAME):
        await pg_engine.ainit_vectorstore_table(
            table_name=TABLE_NAME,
            vector_size=EMBEDDING_DIM,
            schema_name=SCHEMA_NAME,
        )
        print(f"[store_documents] 表 {TABLE_NAME} 不存在，已创建")
    else:
        print(f"[store_documents] 表 {TABLE_NAME} 已存在，复用")

    # 3) 初始化向量库（单例）
    if vector_store is None:
        vector_store = await PGVectorStore.create(
            engine=pg_engine,
            table_name=TABLE_NAME,
            schema_name=SCHEMA_NAME,
            embedding_service=embeddings,
        )

    # 4) PGVectorStore 会负责生成 embedding 并和文档一起写入数据库。
    ids = await vector_store.aadd_documents(splits)
    print(f"[store_documents] 已写入 {len(splits)} 个切片到 {TABLE_NAME}")
    return ids

async def main():
    """一个包装了所有异步逻辑的入口函数。"""
    try:
        await store_documents(BASE_DIR / 'data', chunk_size=500, chunk_overlap=100)
    finally:
        # 关键步骤：确保在事件循环结束前，清理并重置全局状态
        global pg_engine, vector_store
        if pg_engine is not None:
            await pg_engine.close()
            pg_engine = None
            vector_store = None
            print("[main] 数据库连接池已释放，全局变量已重置。")

if __name__ == "__main__":
    asyncio.run(main())
