# 后端流式输出详细实现指南

本文专门说明如何在当前项目上实现后端流式输出。目标是：用户发送问题后，模型生成的回答可以逐步返回给浏览器，而不是等完整回答生成后一次性返回 JSON。

本文基于当前工作区的代码状态编写。当前 `base_agent.py` 已经有 `ainvoke()`、`astream()` 的占位方法，`main.py` 已经有 `/v1/chat`、`/v1/chat/stream` 和删除会话的占位路由。本文只修改文档，不修改这些代码。

## 1. 目标和最终调用链

最终请求链路应当是：

```text
浏览器
  |
  | POST /v1/chat/stream
  | {"question": "...", "session_id": "..."}
  v
FastAPI
  |
  | StreamingResponse(text/event-stream)
  v
My_Agent.astream()
  |
  | agent.astream_events(version="v2")
  v
LangGraph Agent
  |---------------------------|
  v                           v
知识库搜索工具                 ChatModel
                              |
                              v
                  token / done / error
                              |
                              v
                          浏览器逐步显示
```

实现需要同时处理四件事：

1. Agent 层必须使用异步流式接口。
2. FastAPI 必须返回 `StreamingResponse`。
3. 服务端必须把模型事件转换为稳定的 SSE 事件格式。
4. 同一个 `session_id` 必须转换成同一个 LangGraph `thread_id`。

只实现 Agent 流式调用但仍然返回普通 JSON，或者只把 FastAPI 改成 `StreamingResponse` 但内部仍然调用 `invoke()`，都不会产生真正的流式效果。

## 2. 当前代码中需要先修正的问题

### 2.1 `checkpointer` 必须保存为实例属性

当前 `base_agent.py` 中是：

```python
checkpointer = InMemorySaver()
```

它是 `__init__()` 内的局部变量。`create_agent()` 可以使用它，但类的其他方法无法访问它，因此后续无法实现 `clear_session()`。

应改成：

```python
self.checkpointer = InMemorySaver()
```

然后传入 Agent：

```python
self.agent = create_agent(
    ...,
    checkpointer=self.checkpointer,
)
```

### 2.2 `config=config` 是错误的

当前文件同时存在：

```python
import config
```

以及：

```python
config=config
```

这里传入的是项目的 `config.py` 模块，不是 LangGraph 所需的运行配置。LangGraph 需要的是包含 `configurable.thread_id` 的字典：

```python
run_config = {
    "configurable": {
        "thread_id": session_id,
    }
}
```

后续 `invoke()`、`ainvoke()` 和 `astream_events()` 都必须使用这个运行配置。

### 2.3 现有的 `response_format=AIResponse` 不适合直接作为流式文本

当前 Agent 使用：

```python
response_format=AIResponse
```

这个设置适合在调用结束时得到完整的结构化结果，但流式事件中可能出现 JSON 字段、结构化输出片段或工具调用参数，而不是用户想看到的自然语言回答。

推荐采用两个 Agent：

- `self.agent`：保留 `response_format=AIResponse`，用于普通 JSON 接口 `/v1/chat`。
- `self.stream_agent`：不设置 `response_format`，用于 `/v1/chat/stream`，只流式输出自然语言文本。

两个 Agent 可以共用工具和同一个 checkpointer。这样普通调用和流式调用都能读取同一个会话历史。

如果暂时不需要 `/v1/chat`，也可以只保留一个不设置 `response_format` 的 Agent，然后由服务端自行组装 `AIResponse`。但这会改变现有普通接口的行为，建议先使用两个 Agent，迁移风险更低。

## 3. 依赖和配置检查

### 3.1 是否需要新增依赖

当前项目使用的 FastAPI 自带的 Starlette `StreamingResponse` 即可实现 SSE，不需要额外安装 `sse-starlette`。

`requirements.txt` 中至少需要确保以下包可以被实际导入：

```text
fastapi
uvicorn[standard]
langchain
langgraph
langchain-openai
langchain-postgres
```

当前 `requirements.txt` 中代码实际导入了 `langchain_postgres`，但依赖文件没有显式列出 `langchain-postgres`。这不是流式 API 的特殊依赖，但在新环境安装项目时会导致启动失败，建议补充一个与当前 LangChain/LangGraph 兼容的版本范围。

### 3.2 模型必须支持流式输出

`common.py` 中通过 `init_chat_model()` 创建模型：

```python
llm = init_chat_model(...)
```

需要确认当前 `CHAT_MODEL` 对应的服务商和模型支持 streaming。判断标准不是模型名字中有没有“stream”，而是 LangChain 模型对象是否支持：

```python
await llm.ainvoke(...)
llm.astream(...)
```

如果模型服务商的 OpenAI 兼容接口没有实现流式响应，Agent 代码即使使用 `astream_events()`，也可能等到完整结果后才产生事件。

### 3.3 `temperature` 需要重新确认

当前 `common.py` 中是：

```python
temperature=30
```

很多模型接口要求温度在 `0` 到 `2` 之间，`30` 可能会被拒绝或产生异常行为。流式改造前建议先改为服务商允许的值，问答场景通常可从 `0.2` 或 `0.3` 开始测试。该项不是流式协议要求，但如果模型初始化失败，后续无法验证流式接口。

## 4. 修改 `base_agent.py`

下面的修改分为五部分：导入、初始化 checkpointer、创建流式 Agent、补充运行配置、实现异步和流式方法。

### 4.1 修改导入部分

文件：`base_agent.py`

当前已有：

```python
from typing import Any
```

建议改成或补充为：

```python
import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any
```

`asyncio` 用于处理客户端取消连接，`AsyncIterator` 用于标注异步生成器，`datetime` 用于在服务端生成完成时间。

### 4.2 增加 SSE 事件的数据约定

Agent 层不要直接拼接 HTTP 字符串，先返回普通字典，路由层统一转换成 SSE。这样 Agent 可以独立测试，HTTP 协议也不会和模型逻辑耦合。

建议在 `AIResponse` 后面增加类型别名：

```python
StreamEvent = dict[str, Any]
```

事件统一使用以下结构：

```python
{
    "event": "token",
    "data": {
        "text": "青岛",
    },
}
```

支持三种事件：

```text
token  增量文本
done   本次回答已经完成，携带最终元数据
error  本次回答失败
```

### 4.3 把 checkpointer 改为实例属性

在 `My_Agent.__init__()` 中，将当前代码：

```python
checkpointer = InMemorySaver()
```

改为：

```python
self.checkpointer = InMemorySaver()
```

然后修改现有普通 Agent：

```python
self.agent = create_agent(
    model=self.llm,
    system_prompt=(
        "你是一个智能助手，需要严谨的回答用户的问题，且只能通过知识库检索的内容进行回答，否则要明确表示，"
        "不知道答案。如果问你聊天历史，知道可以直接回答，不用从知识库里面检索。"
    ),
    tools=[search_knowledge_base],
    middleware=[clear_message],
    response_format=AIResponse,
    checkpointer=self.checkpointer,
)
```

注意，这个普通 Agent 仍然保留 `response_format=AIResponse`，用于非流式接口。

### 4.4 创建一个不带结构化输出的流式 Agent

在普通 Agent 创建代码后面增加：

```python
self.stream_agent = create_agent(
    model=self.llm,
    system_prompt=(
        "你是一个智能助手，需要严谨地回答用户的问题。"
        "你只能依据知识库检索到的内容回答事实问题；如果知识库没有相关内容，必须明确说明不知道。"
        "如果用户是在追问聊天历史，可以根据已有会话上下文回答，不必重复检索。"
    ),
    tools=[search_knowledge_base],
    middleware=[clear_message],
    checkpointer=self.checkpointer,
)
```

这里故意不设置 `response_format=AIResponse`。流式 Agent 的模型最终输出应该是正常的自然语言文本。

两个 Agent 使用同一个 `InMemorySaver`，因此都能使用：

```text
同一个 session_id -> 同一个 thread_id -> 同一份聊天历史
```

如果当前 LangGraph 版本不允许两个 Agent 共享一个 checkpointer，可以采用备用方案：只保留 `stream_agent`，普通接口也改为从最终 `AIMessage` 手动组装 `AIResponse`。先以项目实际安装版本的测试结果为准。

### 4.5 增加运行配置方法

在 `My_Agent` 类中增加：

```python
    @staticmethod
    def _run_config(session_id: str) -> dict[str, Any]:
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id 不能为空")

        return {
            "configurable": {
                "thread_id": session_id,
            }
        }
```

所有调用都通过这个方法生成配置，避免某个接口漏传 `thread_id`。

### 4.6 修正同步调用

将当前的同步方法：

```python
    def invoke(self, question: str, session_id: str) -> AIResponse:
        response = self.agent.invoke(
            {'messages': [HumanMessage(content=question)]},
            config=config
        )

        return response['structured_response']
```

改为：

```python
    def invoke(self, question: str, session_id: str) -> AIResponse:
        response = self.agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=self._run_config(session_id),
        )

        return response["structured_response"]
```

这里最重要的变化是：

```python
config=self._run_config(session_id)
```

不能再写成：

```python
config=config
```

### 4.7 实现异步普通调用

当前占位代码是：

```python
    def ainvoke(self, question: str, seesion_id: str) -> AIResponse:
        pass
```

需要修改拼写、增加 `async`，并实现为：

```python
    async def ainvoke(self, question: str, session_id: str) -> AIResponse:
        response = await self.agent.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            config=self._run_config(session_id),
        )

        return response["structured_response"]
```

FastAPI 的 `async def` 路由应该调用这个方法，而不是在事件循环中直接调用同步的 `invoke()`。

### 4.8 增加文本提取辅助方法

不同模型适配器的 `chunk.content` 可能是字符串，也可能是 content block 列表。不要直接假设它永远是字符串。

在 `My_Agent` 类中增加：

```python
    @staticmethod
    def _extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content

        if not isinstance(content, list):
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
```

这个方法只提取文本 block，不会把工具调用参数当成用户回答显示出来。

### 4.9 增加工具结果判断方法

当前知识库工具返回的是 `list[str]`。为了在最终 `done` 事件中填写 `is_included`，增加一个保守的判断方法：

```python
    @staticmethod
    def _has_retrieval_hit(output: Any) -> bool:
        if output is None:
            return False

        if isinstance(output, list):
            return len(output) > 0

        if isinstance(output, dict):
            for key in ("content", "texts", "documents", "results"):
                value = output.get(key)
                if isinstance(value, list) and value:
                    return True
            return False

        return bool(output)
```

这比让模型自己生成 `is_included` 更稳定。后续如果把检索工具升级为返回来源和 score，只需要扩展这个方法。

### 4.10 实现流式 Agent 方法

将当前占位代码：

```python
    def astream(self, question: str, session_id: str) -> AIResponse:
        pass
```

替换为下面的异步生成器：

```python
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
```

这段代码的行为是：

1. `on_chat_model_stream` 到来时提取增量文本。
2. 每个文本片段立即 yield 一个 `token` 事件。
3. `on_tool_end` 到来时根据工具输出判断是否命中知识库。
4. Agent 事件结束后拼接完整回答，发送一个 `done` 事件。
5. 出现异常时发送 `error` 事件。

### 4.11 关于 `on_chat_model_stream` 的过滤

上面的实现使用 `_extract_text()` 过滤工具调用参数。首次接入时，建议临时打印事件的以下字段：

```python
print(
    event.get("event"),
    event.get("name"),
    event.get("metadata"),
)
```

确认当前模型实际产生的事件后再删除调试日志。不同版本可能在 `metadata["langgraph_node"]` 中提供 `model`、`tools` 等节点名。

如果发现多个模型节点都产生文本，可以增加过滤：

```python
metadata = event.get("metadata") or {}
if metadata.get("langgraph_node") != "model":
    continue
```

但不要一开始就强依赖这个字段，因为不同 LangGraph 版本或自定义节点可能没有完全相同的 metadata。

### 4.12 实现清除会话

当前已有：

```python
    def clear_session(self, session_id):
        pass
```

如果当前版本的 `InMemorySaver` 支持 `delete_thread()`，可以实现为：

```python
    def clear_session(self, session_id: str) -> None:
        self.checkpointer.delete_thread(session_id)
```

如果安装的 LangGraph 版本没有这个方法，不要猜测内部字段结构。可以先不暴露删除接口，或者升级到提供线程删除能力的版本，再根据当前版本 API 实现。

## 5. 修改 `main.py`

### 5.1 修改导入

当前导入：

```python
from fastapi import FastAPI
```

建议改为：

```python
import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
```

项目已有的导入保留：

```python
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from base_agent import My_Agent, AIResponse
from common import llm
```

增加日志对象：

```python
logger = logging.getLogger(__name__)
```

### 5.2 修改请求模型

当前是：

```python
class RequestBody(BaseModel):
    question: str = Field(description="问题")
    session_id: str = Field(description="唯一的ID")
```

建议增加长度约束：

```python
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
```

如果希望自动去除首尾空格，可以在进入 Agent 前执行：

```python
question = request_body.question.strip()
session_id = request_body.session_id.strip()
```

空字符串在清理后应返回 HTTP 400，而不是发给模型。

### 5.3 增加 SSE 编码函数

在路由定义前增加：

```python
def encode_sse(event: str, data: dict[str, Any]) -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    )
```

SSE 要求：

- `event:` 指定事件名。
- `data:` 放 JSON 字符串。
- 每条事件最后必须有一个空行。

下面是一个完整事件：

```text
event: token
data: {"text": "青岛"}

```

不能直接写：

```python
yield event
```

因为 Agent 返回的是 Python 字典，浏览器无法按 SSE 协议解析。

### 5.4 修改 CORS 配置

当前只允许 `GET` 和 `POST`：

```python
allow_methods=["GET", "POST"],
```

如果保留删除会话接口，需要改为：

```python
allow_methods=["GET", "POST", "DELETE"],
```

流式 POST 本身不需要特殊的 CORS 配置，但浏览器会先发送 OPTIONS 预检请求，FastAPI 的 CORSMiddleware 应保留。

### 5.5 实现普通异步聊天接口

当前占位代码：

```python
@app.post('/v1/chat')
async def v1_chat(request_body: RequestBody):
    pass
```

建议实现为：

```python
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
```

这个接口不是流式接口，但它很重要：可以用来确认 `session_id`、checkpointer 和普通异步调用先工作正常。

### 5.6 实现流式接口

当前占位代码：

```python
@app.post('/v1/chat/stream')
async def chat_stream(request_body: RequestBody):
    pass
```

建议实现为：

```python
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
```

这里有几个关键点：

- `event_generator()` 必须是异步生成器。
- 每次 `yield` 都必须是完整的 SSE 文本。
- `media_type` 必须是 `text/event-stream`。
- `Cache-Control` 和 `X-Accel-Buffering` 用于减少中间层缓存和缓冲。
- `CancelledError` 要重新抛出，让模型调用尽快停止。
- 内部异常记录到服务端日志，客户端只收到通用错误，不能泄露数据库连接串或 API Key。

### 5.7 实现删除会话接口

当前占位代码：

```python
@app.delete('/v1/chat/{id}')
async def delete(id: str):
    pass
```

建议改为：

```python
@app.delete("/v1/chat/{session_id}")
async def delete_session(session_id: str):
    session_id = session_id.strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id 不能为空")

    try:
        my_agent.clear_session(session_id)
    except AttributeError:
        raise HTTPException(
            status_code=501,
            detail="当前 checkpointer 不支持删除会话",
        )

    return {
        "session_id": session_id,
        "deleted": True,
    }
```

如果当前 `InMemorySaver` 的版本没有 `delete_thread()`，应根据实际版本处理，而不是假设删除成功。这个接口可以暂时不开放，流式功能不依赖它。

### 5.8 处理旧的 `/v1` 接口

当前旧接口仍然是：

```python
@app.post('/v1')
async def query(request_body: RequestBody):
    ...
    return my_agent.invoke(request_body.question)
```

由于 `RequestBody` 现在要求 `session_id`，旧代码会出现两个问题：

1. 旧前端没有发送 `session_id`，请求会被 FastAPI 返回 422。
2. 旧路由调用 `invoke()` 时没有传 `session_id`。

有两个选择。

推荐选择：保留接口，但改为兼容调用：

```python
@app.post("/v1", response_model=AIResponse)
async def query(request_body: RequestBody):
    return await my_agent.ainvoke(
        question=request_body.question.strip(),
        session_id=request_body.session_id.strip(),
    )
```

另一种选择是删除旧接口，并同步修改前端使用 `/v1/chat/stream`。在前端还没有迁移完成前，不建议直接删除旧接口。

## 6. SSE 协议约定

### 6.1 token 事件

模型每产生一段文本就返回：

```text
event: token
data: {"text":"青岛"}

```

前端只需要把 `data.text` 追加到当前回答节点。

### 6.2 done 事件

回答结束后返回一次：

```text
event: done
data: {"question":"请介绍青岛","answer":"青岛是一座...","time":"2026-11-13T10:00:00+00:00","is_included":"是","session_id":"demo-1"}

```

`done` 事件用于：

- 标记请求完成。
- 保存完整回答。
- 显示知识库命中状态。
- 显示生成时间。
- 恢复发送按钮。

### 6.3 error 事件

发生异常时返回：

```text
event: error
data: {"code":"STREAM_ERROR","message":"回答生成失败，请稍后重试。"}

```

一个请求通常只能以 `done` 或 `error` 之一结束。发送 `error` 后不要再发送 `done`。

### 6.4 心跳事件

如果模型很久没有输出，反向代理可能认为连接空闲。需要时可以在 `event_generator()` 中增加心跳：

```text
: keep-alive

```

SSE 中以冒号开头的行是注释，浏览器不会当作业务事件。第一版可以不加，遇到代理超时再增加。

## 7. `response_format` 的两种处理方式

### 7.1 推荐方式：普通接口和流式接口使用两个 Agent

这是本文示例采用的方式：

```text
/v1/chat
  -> self.agent
  -> response_format=AIResponse
  -> 完整 JSON

/v1/chat/stream
  -> self.stream_agent
  -> 无 response_format
  -> token + done SSE
```

优点是普通接口的现有返回结构不变，流式接口也不会把 JSON 片段显示给用户。

### 7.2 备选方式：只保留一个普通文本 Agent

如果不需要严格的结构化回答，可以删除：

```python
response_format=AIResponse
```

然后普通接口从最终消息中取出文本：

```python
response = await self.agent.ainvoke(...)
messages = response.get("messages", [])
answer = messages[-1].content if messages else ""
```

再由服务端构造：

```python
AIResponse(
    question=question,
    answer=answer,
    time=...,
    is_included=...,
)
```

这个方式更简单，但需要自己处理最终消息类型、工具调用结果和命中状态。当前项目建议先采用 7.1 的双 Agent 方式。

## 8. 启动和测试

### 8.1 安装依赖

当前环境如果没有依赖，先在项目虚拟环境中安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果安装后仍提示 `No module named langchain_postgres`，补充安装与当前 LangChain 兼容的 `langchain-postgres`，并把它写入 `requirements.txt`。

### 8.2 启动后端

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

不要用普通的 `curl` 测 SSE，否则终端可能缓冲输出。使用 `-N`：

```bash
curl -N -X POST http://localhost:8000/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"请介绍一下青岛","session_id":"curl-demo-1"}'
```

预期可以看到多条事件，例如：

```text
event: token
data: {"text":"青岛"}

event: token
data: {"text":"是一座"}

event: done
data: {"question":"请介绍一下青岛", "answer":"...", "is_included":"是", "session_id":"curl-demo-1"}
```

如果只在最后一次性看到全部内容，依次排查：

1. 模型适配器是否真的支持 streaming。
2. Agent 是否使用了 `astream_events()` 而不是 `invoke()`。
3. 路由是否返回了 `StreamingResponse`。
4. `media_type` 是否为 `text/event-stream`。
5. 是否有 Nginx、网关或开发代理缓冲响应。
6. `curl` 是否使用了 `-N`。

### 8.3 验证会话记忆

连续发送两个请求，并复用同一个 `session_id`：

```bash
curl -N -X POST http://localhost:8000/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"介绍一下青岛","session_id":"memory-demo-1"}'

curl -N -X POST http://localhost:8000/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"它有哪些特点？","session_id":"memory-demo-1"}'
```

再换一个 `session_id`，确认新会话不能读取 `memory-demo-1` 的历史。

### 8.4 验证错误和取消

需要验证：

- 空问题返回 HTTP 400 或 FastAPI 的校验错误。
- 模型服务不可用时收到 `error` 事件。
- 浏览器关闭页面或点击停止后，后端日志能看到取消，而不是一直生成。
- 发生错误后前端按钮仍然可以再次发送。

## 9. 常见错误

### 错误一：把异步生成器加上 `await`

错误：

```python
events = await my_agent.astream(...)
```

异步生成器不能这样等待。正确方式是在 `async for` 中读取：

```python
async for event in my_agent.astream(...):
    ...
```

### 错误二：在 `async def` 中调用同步 `invoke()`

错误：

```python
@app.post("/v1/chat/stream")
async def chat_stream(...):
    result = my_agent.invoke(...)
```

这会阻塞事件循环，而且仍然不是流式。应使用 `astream()`。

### 错误三：每个请求重新创建 Agent 或 checkpointer

错误：

```python
@app.post(...)
async def route(...):
    agent = My_Agent(llm=llm)
```

这样会导致连接池、工具和内存 checkpointer 在请求间无法稳定复用。当前项目应在模块加载或 FastAPI lifespan 中初始化一次。

### 错误四：忘记传 `thread_id`

即使使用了 `InMemorySaver`，如果每次调用都不传：

```python
{"configurable": {"thread_id": session_id}}
```

也无法区分不同会话，记忆功能不会按预期工作。

### 错误五：一次 `read()` 当作一条 SSE 事件

这是前端问题，但后端协议必须允许网络分包。一次网络读取可能包含半条事件、完整一条事件或多条事件。前端必须使用缓冲区按空行拆分，不能直接对每次读取的数据 `JSON.parse()`。

### 错误六：把内部异常原样返回

错误：

```python
yield encode_sse("error", {"message": str(exc)})
```

`str(exc)` 可能包含数据库地址、SQL、模型服务 URL 或其他内部信息。详细错误写服务端日志，客户端只接收稳定的错误码和用户可读提示。

## 10. 建议的文件修改清单

实现后端流式输出时，至少需要处理以下位置：

| 文件 | 修改内容 | 是否必须 |
| --- | --- | --- |
| `base_agent.py` | 实例化并保存 checkpointer | 必须 |
| `base_agent.py` | 修正 `config=config`，统一生成 `thread_id` 配置 | 必须 |
| `base_agent.py` | 增加不带 `response_format` 的流式 Agent | 推荐 |
| `base_agent.py` | 实现 `async ainvoke()` | 推荐 |
| `base_agent.py` | 实现异步 `astream()` | 必须 |
| `base_agent.py` | 增加文本提取、工具命中判断 | 必须 |
| `main.py` | 增加 `StreamingResponse`、JSON 编码和日志导入 | 必须 |
| `main.py` | 实现 `/v1/chat/stream` | 必须 |
| `main.py` | 让 `/v1/chat` 调用 `ainvoke()` | 推荐 |
| `main.py` | 修正旧 `/v1` 的 `session_id` 调用 | 兼容旧前端时必须 |
| `main.py` | 修改 CORS 的 DELETE 方法 | 使用删除接口时必须 |
| `requirements.txt` | 显式加入 `langchain-postgres` | 建议 |
| `common.py` | 确认模型支持 streaming，检查 temperature | 必须验证 |

## 11. 推荐落地顺序

按下面顺序实施，可以减少问题同时出现时的排查难度：

1. 先修正 `checkpointer` 实例属性和 `run_config`。
2. 实现 `ainvoke()`，用 `/v1/chat` 验证同一 `session_id` 的历史记忆。
3. 增加 `stream_agent`，先用 `curl -N` 验证 `token` 事件。
4. 增加 `done`、`error` 事件和服务端异常日志。
5. 修改前端读取 SSE，逐段更新回答区域。
6. 最后处理会话删除、PostgreSQL 持久化和反向代理缓冲。

完成后，后端最小可用状态应满足：

```text
POST /v1/chat/stream
  -> 立即返回 text/event-stream
  -> 持续返回 token
  -> 最后返回 done
  -> 出错时返回 error
  -> 同一 session_id 可以读取历史
```
