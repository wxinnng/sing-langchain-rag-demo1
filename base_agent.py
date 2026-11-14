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
import asyncio
from collections.abc import AsyncIterator
from dotenv import load_dotenv
import os
import config
from langgraph.checkpoint.memory import InMemorySaver

from datetime import datetime, timezone

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
    time: str = Field(description="回答的具体的时间")
    is_included: str = Field(description="知识库里面是否包含相关的内容")

StreamEvent = dict[str, Any]

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

        self.checkpointer = InMemorySaver()

        # ③ 最后创建 agent，此时两个函数都已定义
        self.agent = create_agent(
            model=self.llm,
            system_prompt=(
                "你是一个智能助手，需要严谨的回答用户的问题，且只能通过知识库检索的内容进行回答，否则要明确表示，"
                "不知道答案,如果问你聊天历史，知道可以直接回答，不用从知识库里面检索。"
            ),
            tools=[search_knowledge_base],
            middleware=[clear_message],
            response_format=AIResponse,
            checkpointer=self.checkpointer,
        )

        self.stream_agent = create_agent(
            model=self.llm,
            system_prompt=(
                "你是一个智能助手，需要严谨的回答用户的问题，且只能通过知识库检索的内容进行回答，否则要明确表示，"
                "不知道答案,如果问你聊天历史，知道可以直接回答，不用从知识库里面检索。"
            ),
            tools=[search_knowledge_base],
            middleware=[clear_message],
            checkpointer=self.checkpointer,
        )

    # 同步调用
    def invoke(self, question: str, session_id: str) -> AIResponse:
        response = self.agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=self._run_config(session_id),
        )

        return response["structured_response"]

    @staticmethod
    def _extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content

        if not isinstance(content, (list, tuple)):
            return ""

        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(block.get("text"), str):
                    parts.append(block["text"])
                continue

            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)

        return "".join(parts)

    @staticmethod
    def _has_retrieval_hit(output: Any) -> bool:
        if output is None:
            return False

        if isinstance(output, (list, tuple)):
            return bool(output)

        if isinstance(output, dict):
            for key in ("content", "texts", "documents", "results"):
                value = output.get(key)
                if isinstance(value, list) and value:
                    return True
            return False

        content = getattr(output, "content", None)
        if content is not None:
            return bool(content)

        return bool(output)



    # 异步调用
    async def ainvoke(self, question: str, session_id: str) -> AIResponse:
        response = await self.agent.ainvoke(
            {
                'messages':[HumanMessage(content=question)],
            },
            config=self._run_config(session_id)
        )

        return response["structured_response"]

    # 流式输出
    async def astream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncIterator[StreamEvent]:
        answer_parts: list[str] = []
        retrieval_hit = False

        try:
            async for event in self.stream_agent.astream_events(
                {"messages": [HumanMessage(content=question)]},
                config=self._run_config(session_id),
                version="v2",
            ):
                event_name = event.get("event")
                event_data = event.get("data") or {}

                if event_name == "on_chat_model_stream":
                    chunk = event_data.get("chunk")
                    content = getattr(chunk, "content", None)
                    text = self._extract_text(content)

                    if text:
                        answer_parts.append(text)
                        yield {
                            "event": "token",
                            "data": {"text": text},
                        }

                elif event_name == "on_tool_end":
                    tool_output = event_data.get("output")
                    if self._has_retrieval_hit(tool_output):
                        retrieval_hit = True

            answer = "".join(answer_parts)
            yield {
                "event": "done",
                "data": {
                    "question": question,
                    "answer": answer,
                    "time": datetime.now(timezone.utc).isoformat(),
                    "is_included": "是" if retrieval_hit else "否",
                    "session_id": session_id,
                },
            }
        except asyncio.CancelledError:
            # 浏览器主动关闭连接时，不能继续向下游发送事件。
            raise
        except Exception:
            # 不要把模型、数据库或密钥相关的内部错误直接发送给客户端。
            yield {
                "event": "error",
                "data": {
                    "code": "STREAM_ERROR",
                    "message": "回答生成失败，请稍后重试。",
                },
            }




    # 清除一个会话
    def clear_session(self, session_id: str) -> None:
        delete_thread = getattr(self.checkpointer, "delete_thread", None)
        if not callable(delete_thread):
            raise NotImplementedError("当前 checkpointer 不支持删除会话")
        delete_thread(session_id)

    @staticmethod
    def _run_config(session_id: str) -> dict[str, Any]:

        session_id = session_id.strip()

        if not session_id:
            raise ValueError("session_id 不能为空")

        return {
            "configurable": {
                "thread_id": session_id,
            },
        }


if __name__ == "__main__":
    agent = My_Agent(llm=llm)
    response = agent.invoke("给我介绍一下青岛", session_id="local-demo")

    print(response)
