from common import llm
from langchain.chat_models import BaseChatModel
from langchain.agents import AgentState,create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage,AIMessage,ToolMessage
from dataclasses import dataclass
from pydantic import Field
from langchain_postgres import PGEngine, PGVectorStore
import os
import config


from dotenv import load_dotenv
from common import embeddings

load_dotenv()

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://rag:rag@localhost:5433/rag",
)

EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", 1024))
class My_Agent:

    def __init__(self,llm):

        # 赋值大语言模型
        self.llm = llm

        engine = PGEngine.from_connection_string(url = DATABASE_URL)

        self.store = PGVectorStore.create_sync(
            embedding_service=embeddings,
            engine=engine,
            table_name='lc_documents'
        )

        # 初始化Agent对象
        self.agent = create_agent(

            model=self.llm,
            system_prompt="你是一个智能助手，需要严谨的回答用户的问题，且只能通过知识库检索的内容进行回答，否则要明确表示，不知道答案"
        )

        


    # @tool
    def search_knowledge_base(self, question: str) -> list[str]:
        """
        去数据库中查询最相似内容

        Args:
            question:用户问题

        Returns:
            与用户问题比较相似的几个片段(按照相似度由高到低排序)，也有可能是空
    
        """

        results = self.store.similarity_search_with_score(question, k = config.K)


        filtered = []

        for doc, dist in results:
            similarity = 1 - dist / 2      # 余弦距离[0,2] → 相似度[1,0]
            if similarity >= 0.7:
                filtered.append((doc, similarity))

        # 最终返回 list[str]：只要文本内容
        texts: list[str] = [doc.page_content for doc, sim in filtered]
        
        return texts



if __name__ == '__main__':

    agent = My_Agent(
        llm = llm
    )

    texts = agent.search_knowledge_base("数据包协议")

    for text in texts:

        print(text + '\n' + " ==================")