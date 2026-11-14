import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from base_agent import My_Agent, AIResponse
from common import llm

logger = logging.getLogger(__name__)


class RequestBody(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="问题",
    )
    session_id: str = Field(
        min_length=1,
        max_length=128,
        description="会话 ID",
    )

app = FastAPI()

# Allow the plain static frontend to call the API during local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


def encode_sse(event: str, data: dict[str, Any]) -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    )

llm = llm

my_agent = My_Agent(llm=llm)

@app.get('/')
async def hello():
    return 'hello world'


@app.get('/v1/{query}')
async def do_short_query(query: str):
    return query


@app.post("/v1", response_model=AIResponse)
async def query(request_body: RequestBody):
    question = request_body.question.strip()
    session_id = request_body.session_id.strip()

    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id 不能为空")

    return await my_agent.ainvoke(
        question=question,
        session_id=session_id,
    )

@app.post("/v1/chat", response_model=AIResponse)
async def v1_chat(request_body: RequestBody):
    question = request_body.question.strip()
    session_id = request_body.session_id.strip()

    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id 不能为空")

    return await my_agent.ainvoke(
        question=question,
        session_id=session_id,
    )
@app.post("/v1/chat/stream")
async def chat_stream(request_body: RequestBody):
    question = request_body.question.strip()
    session_id = request_body.session_id.strip()

    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id 不能为空")

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for event in my_agent.astream(
                question=question,
                session_id=session_id,
            ):
                yield encode_sse(
                    event=event["event"],
                    data=event["data"],
                )
        except asyncio.CancelledError:
            logger.info("客户端取消流式请求，session_id=%s", session_id)
            raise
        except Exception:
            logger.exception(
                "流式回答失败，session_id=%s",
                session_id,
            )
            yield encode_sse(
                event="error",
                data={
                    "code": "STREAM_ERROR",
                    "message": "回答生成失败，请稍后重试。",
                },
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.delete("/v1/chat/{session_id}")
async def delete_session(session_id: str):
    session_id = session_id.strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id 不能为空")

    try:
        my_agent.clear_session(session_id)
    except NotImplementedError:
        raise HTTPException(
            status_code=501,
            detail="当前 checkpointer 不支持删除会话",
        )

    return {
        "session_id": session_id,
        "deleted": True,
    }
