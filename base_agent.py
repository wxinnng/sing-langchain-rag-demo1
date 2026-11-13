from common import _async_database_url, embeddings, llm
from langchain.agents import AgentState, create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage, RemoveMessage
from langchain_postgres import PGEngine, PGVectorStore
from langchain.agents.middleware import before_model
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field
from typing import Any
from dotenv import load_dotenv
import os
import config
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://rag:rag@localhost:5433/rag",
)

EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", 1024))


class AIResponse(BaseModel):
    """AI回答格式"""

    question: str = Field(description="用户的问题")
    answer: str = Field(description="回答的内容")
    time: str = Field(description="回答的时间")
    is_included: str = Field(description="知识库里面是否包含相关的内容")


class My_Agent:

    def __init__(self, llm):
        self.llm = llm

        engine = PGEngine.from_connection_string(url=_async_database_url(DATABASE_URL))
        self.store = PGVectorStore.create_sync(
            embedding_service=embeddings,
            engine=engine,
            table_name='lc_documents',
        )

        # ① 先定义工具闭包，捕获 self
        @tool
        def search_knowledge_base(question: str) -> list[str]:
            """从数据库中搜索相似的内容"""
            results = self.store.similarity_search_with_score(question, k=config.K)
            threshold = config.LIMIT_SIMILAR_RATE

            texts = []
            for doc, score in results:
                # 前提：score 是 cosine distance ∈ [0, 2]
                cosine_similarity = 1 - score
                if cosine_similarity >= threshold:
                    texts.append(doc.page_content)
            return texts

        # ② 再定义中间件，同样捕获 self（虽然这里没用到）
        @before_model
        def clear_message(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            """保留首条信息和最近的10条信息，移除中间冗余的信息"""
            messages = state['messages']
            print(f'===============消息数量：{len(messages)} ====================')

            if len(messages) < 11:
                return None

            first_message = messages[0]
            recent_message = messages[-10:]
            new_messages = [first_message] + recent_message

            return {
                'messages': [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    *new_messages,
                ],
            }

        # ③ 最后创建 agent，此时两个函数都已定义
        self.agent = create_agent(
            model=self.llm,
            system_prompt=(
                "你是一个智能助手，需要严谨的回答用户的问题，"
                "且只能通过知识库检索的内容进行回答，否则要明确表示，不知道答案"
            ),
            tools=[search_knowledge_base],
            middleware=[clear_message],
            response_format=AIResponse,
        )

    def invoke(self, question: str) -> AIResponse:
        response = self.agent.invoke(
            {'messages': [HumanMessage(content=question)]}
        )
        print(type(response['structured_response']))
        return response['structured_response']


if __name__ == '__main__':
    agent = My_Agent(llm=llm)
    response = agent.invoke("给我介绍一下青岛")

    print(response)