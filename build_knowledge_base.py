from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from common import store_documents
import config

from dotenv import load_dotenv
import asyncio

load_dotenv()

EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", 1024))


async def do_pdf(path: str):
    """做指定目录下的pdf文件拆分"""

    # 加载文档
    loader = PyPDFLoader(path)
    documents = loader.load()

    # print(f'加载文档的数量：{len(documents)}')
    # print(documents[0])

    # 文档切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = config.CHUNK_SIZE,
        chunk_overlap= config.OVERLAP,
    )

    splits = text_splitter.split_documents(documents=documents)

    await store_documents(splits=splits)

asyncio.run(do_pdf('/home/wangxing/document.pdf'))