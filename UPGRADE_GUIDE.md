# LangChain RAG Demo 升级方案

本文基于当前项目的代码结构编写，目标是说明后续如何加入多轮会话记忆、流式输出，并逐步处理现有工程问题。本文只提供改造方案，不直接修改项目代码。

## 1. 当前项目结构

当前请求链路如下：

```text
frontend/chat.html
        |
frontend/app.js  -- POST /v1，等待完整 JSON
        |
main.py          -- my_agent.invoke(question)
        |
base_agent.py    -- create_agent + 知识库搜索工具
        |
common.py        -- ChatModel、Embedding、PGVectorStore
        |
PostgreSQL       -- lc_documents 向量表
```

各文件的职责基本清晰：

| 文件 | 当前职责 | 后续主要改造方向 |
| --- | --- | --- |
| `common.py` | 加载模型、连接 PostgreSQL、切分和写入文档 | 拆分配置与资源初始化，补充生命周期管理 |
| `base_agent.py` | 创建 Agent、定义检索工具、生成结构化回答 | 接入 checkpointer、会话 ID、异步调用和流式事件 |
| `main.py` | FastAPI 路由和请求模型 | 增加会话接口、SSE 流式接口、异常处理 |
| `frontend/app.js` | 发送问题并等待 JSON | 保存会话 ID、读取流、逐 token 更新回答 |
| `build_knowledge_base.py` | 导入 PDF 并写入向量库 | 增加文档去重、来源元数据和可重复执行能力 |

## 2. 现状问题定位

### 2.1 当前没有真正启用聊天记忆

`base_agent.py` 第 14 行虽然导入了 `InMemorySaver`，但它没有传给 `create_agent`，所以没有任何 checkpointer 参与 Agent 状态保存。

`My_Agent.invoke()` 每次都只传入一条新的消息：

```python
{'messages': [HumanMessage(content=question)]}
```

而 `main.py` 也没有接收或传递用户、会话等标识。因此第二次 HTTP 请求无法知道第一次对话属于谁。当前的 `clear_message` 只是对本次运行中已经存在的消息做裁剪，不会自动产生跨请求记忆。

需要区分两类数据：

```text
知识库记忆：文档切片 + 向量，存放在 lc_documents
聊天记忆：用户消息、AI 消息、工具调用，按 thread_id 保存
```

向量库解决“回答知识从哪里来”，checkpointer 解决“当前用户之前说过什么”，两者不能互相替代。

### 2.2 当前没有流式输出

后端调用的是同步的 `invoke()`，FastAPI 返回完整的 `AIResponse` JSON。前端使用 `response.json()`，所以必须等模型完整生成后才会显示回答。

要实现流式输出，至少需要同时改动三层：

1. Agent 使用 `astream()` 或 `astream_events()`。
2. FastAPI 使用 `StreamingResponse`，建议采用 `text/event-stream`。
3. 浏览器使用 `fetch()` 的 `response.body.getReader()` 持续读取数据，而不是 `response.json()`。

只改其中一层，用户仍然看不到真正的流式效果。

### 2.3 其他应一并处理的小问题

- `main.py` 在 `async def query()` 中调用同步的 `my_agent.invoke()`，会阻塞 FastAPI 的事件循环。升级时应改为 `ainvoke()`/`astream()`，或明确放到线程池。
- `AIResponse` 的 `question`、`time`、`is_included` 都是必填字段，但异常分支中的 `AIResponse(answer="请先输入问题！")` 缺少其他字段，可能再次触发 Pydantic 校验错误。
- `RequestBody` 已由 FastAPI 校验，`request_body is None` 基本没有意义；空字符串应通过字段约束或显式异常返回处理。
- `time` 和 `is_included` 目前由模型结构化生成。时间应由服务端生成，是否命中知识库应由检索结果决定，不能完全交给模型猜测。
- `common.py` 中 `temperature=30` 数值异常偏高。需要结合当前模型服务商确认其范围，通常问答场景应使用较低温度，例如 `0` 到 `0.7`。
- `search_knowledge_base()` 假定返回的 score 是 cosine distance，并直接使用 `1 - score`。应确认 `PGVectorStore` 当前版本返回的是距离还是相似度，并通过实际数据验证阈值 `0.7` 的含义。
- `clear_message` 按消息条数裁剪，可能把 tool call 和 tool result 拆开。后续应按 token 数裁剪，或做摘要，并保持完整的消息块。
- `langchain_postgres`、`langchain_text_splitters`、`asyncpg` 等实际使用的包应在 `requirements.txt` 中显式声明，不要只依赖间接依赖。
- Agent 初始化时创建的数据库引擎需要在 FastAPI shutdown 时释放。否则开发热重载或多次启动时可能留下连接池。

## 3. 第一阶段：接入会话记忆

### 3.1 先使用 `InMemorySaver` 验证功能

本地开发阶段可以先使用 LangGraph 的内存 checkpointer，验证调用链是否正确。核心思路是：

```python
checkpointer = InMemorySaver()

self.agent = create_agent(
    model=self.llm,
    system_prompt=..., 
    tools=[search_knowledge_base],
    middleware=[clear_message],
    response_format=AIResponse,
    checkpointer=checkpointer,
)
```

每次调用都必须使用同一个 `thread_id`：

```python
config = {
    "configurable": {
        "thread_id": session_id,
    }
}

response = self.agent.invoke(
    {"messages": [HumanMessage(content=question)]},
    config=config,
)
```

其中 `session_id` 是前端生成并持续保存的会话标识。它不是用户问题，也不应该放到 prompt 中。

建议将 Agent 的方法改为下面的职责形式：

```text
invoke(question, session_id)       非流式调用
ainvoke(question, session_id)      异步非流式调用
astream(question, session_id)      流式调用
clear_session(session_id)          清除一个会话
```

`thread_id` 的作用域必须明确：

- 同一个会话的连续提问使用同一个 `thread_id`。
- 新建会话使用新的 `thread_id`。
- 不同用户不能共用同一个 `thread_id`。
- 如果后续加入登录体系，建议使用“用户 ID + 会话 ID”的服务端校验，而不是信任前端任意传入的 ID。

### 3.2 修改请求协议

当前请求只有 `question`，建议升级为：

```json
{
  "session_id": "browser-session-uuid",
  "question": "它的主要特点是什么？"
}
```

建议使用独立的新接口，保留旧的 `POST /v1` 作为兼容入口：

```text
POST /v1/chat          普通 JSON 回答
POST /v1/chat/stream   SSE 流式回答
DELETE /v1/chat/{id}   清除指定会话
```

前端第一次打开页面时生成 UUID，并放到 `localStorage`。点击“新会话”时生成新的 UUID；页面刷新时复用原 UUID。后端仍然要校验长度、格式和访问权限。

### 3.3 记忆裁剪策略

接入 checkpointer 后，历史消息会越来越长，必须限制上下文大小。建议按以下顺序升级：

1. 先限制最近若干轮完整对话，确保一个 AI 回复及其工具消息不会被拆开。
2. 再根据模型上下文窗口按 token 数裁剪，而不是简单使用 `len(messages)`。
3. 历史较长时，将旧消息压缩成一条“会话摘要”，同时保留最近几轮原文。
4. 摘要和最近消息都通过同一个 `thread_id` 保存。

不要简单保留 `messages[0] + messages[-10:]`。当前首条消息通常是第一轮用户问题，不能代表完整的会话摘要；而工具调用消息存在成对关系，随意裁剪可能导致后续模型无法理解上下文。

### 3.4 处理追问和代词

启用记忆后，用户可能提问：

```text
用户：介绍一下青岛。
用户：它的旅游资源有哪些？
```

第二个问题直接拿去做向量检索，命中率可能低。后续可以在检索前增加“问题改写”步骤：根据历史消息将当前问题改写为独立问题，再将独立问题传给向量检索工具。

推荐流程：

```text
历史消息 + 当前问题
        |
问题改写为独立查询
        |
向量检索 lc_documents
        |
基于检索结果生成回答
```

第一版也可以先让 Agent 根据历史自主调用检索工具，等确认基础记忆工作正常后再加入问题改写，避免一次引入太多变量。

## 4. 第二阶段：实现后端流式输出

### 4.1 推荐 SSE 协议

当前前端是通过 HTTP POST 发送问题，因此推荐使用 POST + SSE 响应，而不是依赖只能 GET 的原生 `EventSource`。

响应头建议设置：

```text
Content-Type: text/event-stream; charset=utf-8
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
```

每条事件以空行分隔，建议定义为：

```text
event: token
data: {"text":"青岛"}

event: token
data: {"text":"是一座"}

event: done
data: {"answer":"青岛是一座...","is_included":"是","time":"...","session_id":"..."}
```

发生异常时返回：

```text
event: error
data: {"code":"MODEL_ERROR","message":"回答生成失败"}
```

不要把 Python 对象的 `repr()` 直接写入流中，所有 `data` 内容都应使用 JSON 编码。

### 4.2 Agent 层的流式设计

推荐优先使用 `astream_events(version="v2")`，因为它可以区分模型输出、工具调用和链路结束事件。示意流程如下：

```python
async for event in self.agent.astream_events(
    {"messages": [HumanMessage(content=question)]},
    config={"configurable": {"thread_id": session_id}},
    version="v2",
):
    if event["event"] == "on_chat_model_stream":
        # 提取模型增量文本，转换成 token 事件
        yield make_sse("token", {"text": extract_text(event)})
    elif event["event"] == "on_chain_end":
        # 找到最终 structured_response，生成 done 事件
        yield make_sse("done", build_final_payload(event))
```

实际实现时需要先打印一次事件类型和字段，确认当前 LangChain 版本及所用模型适配器返回的数据结构，再写提取逻辑。不同模型服务商的 chunk 内容可能是字符串，也可能是包含多个 content block 的列表。

### 4.3 结构化回答和流式文本的关系

当前 Agent 使用了 `response_format=AIResponse`。结构化输出适合在回答结束时校验，不适合直接把完整 JSON 当作用户可读文本逐字符展示。

建议采用“双通道”设计：

- `token` 事件只发送 `answer` 的增量文本。
- `done` 事件发送最终的 `question`、`answer`、`time`、`is_included` 和 `session_id`。
- `time` 在服务端生成。
- `is_included` 根据检索结果或工具返回状态计算。
- 如果最终结构化输出校验失败，发送 `error` 事件并记录日志。

如果当前模型或 Agent 版本无法稳定地在 `response_format` 下流式返回，可以先让模型输出普通文本，服务端在结束时补充元数据；等流式链路稳定后再恢复结构化输出校验。

### 4.4 FastAPI 路由形态

后端路由的目标形态可以是：

```python
@app.post("/v1/chat/stream")
async def stream_chat(request: ChatRequest):
    return StreamingResponse(
        my_agent.astream(
            question=request.question,
            session_id=request.session_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
```

这里的 `astream()` 应返回异步生成器，每次 yield 一条完整的 SSE 事件，而不是返回一个已经拼好的字符串。

需要注意客户端断开连接：流式生成器中应捕获取消异常，及时停止模型调用和数据库操作，避免用户关闭页面后后台仍然持续消耗模型额度。

## 5. 第三阶段：前端读取流

前端不能继续使用：

```javascript
const data = await response.json();
```

应改为读取 `response.body`：

```javascript
const response = await fetch(`${API_BASE}/v1/chat/stream`, {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({session_id, question}),
  signal: abortController.signal,
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = '';

while (true) {
  const {value, done} = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, {stream: true});
  // 按空行拆分 SSE 事件，保留不完整的半条事件到下一轮
  // token 事件追加到当前回答，done/error 事件结束本次请求
}
```

前端应补充以下状态：

- 当前 `session_id`。
- 当前请求的 `AbortController`。
- 当前回答节点，收到 token 后只更新这个节点。
- “生成中”“已完成”“失败”“已取消”四种状态。
- 用户点击新会话后清空消息列表并生成新的 `session_id`。

解析 SSE 时不要假设一次 `reader.read()` 就刚好得到一条事件。网络分包可能把一条事件拆成多次，也可能一次返回多条事件。

## 6. 第四阶段：从内存记忆升级为 PostgreSQL 持久化

`InMemorySaver` 只适合本地单进程验证：服务重启后历史消失，多 worker 进程之间也无法共享。需要持久化时，使用项目已经声明的 `langgraph-checkpoint-postgres`，让聊天状态与 PostgreSQL 绑定。

建议流程：

1. 确认 `DATABASE_URL` 能被 checkpointer 使用。
2. 按当前安装的 LangGraph 版本选择同步或异步 `PostgresSaver`。
3. 应用启动时执行一次 checkpointer 的初始化/建表操作。
4. 将同一个 checkpointer 传给 `create_agent`。
5. FastAPI shutdown 时关闭 checkpointer 和数据库连接池。
6. 不要在每次请求中重新创建 checkpointer。

持久化聊天记忆与 `lc_documents` 最好在概念上分开管理。可以共用 PostgreSQL 实例，但应分别记录：

- 会话标识、用户标识、创建时间、最后活跃时间。
- LangGraph checkpoint 数据。
- 文档向量、来源文件、版本和内容哈希。

不要只依赖 checkpoint 表来实现会话列表、重命名、删除和权限控制；这些产品字段应有自己的会话表。

## 7. 检索链路的建议升级

### 7.1 让工具返回来源信息

当前检索工具只返回 `list[str]`，回答无法向用户展示来源，也不便于判断命中情况。建议后续返回结构化结果：

```json
[
  {
    "content": "文档片段...",
    "source": "data/青岛简介.txt",
    "page": 1,
    "score": 0.84
  }
]
```

模型上下文可以使用 `content`，服务端元数据则用于：

- 判断 `is_included`。
- 返回引用来源。
- 记录检索质量。
- 排查阈值过高或过低的问题。

### 7.2 文档写入需要幂等

当前重复运行 `build_knowledge_base.py` 可能重复写入相同切片。升级时应给每个切片增加稳定标识，例如：

```text
hash(source_path + page + chunk_index + content)
```

并根据文档版本或内容哈希执行 upsert。元数据至少保留来源文件、页码、切片序号、导入时间和内容哈希。

### 7.3 不要把知识库命中完全交给模型判断

当检索结果为空或全部低于阈值时，服务端应明确标记“未命中”；系统提示词再要求模型基于检索内容回答。这样 `is_included` 和前端状态才是稳定的，而不是模型自由生成“是/否”。

## 8. 推荐的改造顺序

### 阶段 A：先修正接口和配置

- 增加 `ChatRequest(session_id, question)`。
- 给问题增加非空、长度限制。
- 修正错误响应模型，避免构造缺少必填字段的 `AIResponse`。
- 将 `temperature`、超时、检索数量和阈值移到统一配置。
- 确认依赖包和实际代码版本一致。

### 阶段 B：实现单进程多轮记忆

- 接入 `InMemorySaver`。
- 给 `create_agent` 配置 `checkpointer`。
- 所有 `invoke`/`ainvoke` 调用传入同一个 `thread_id`。
- 前端保存并发送 `session_id`。
- 增加新会话和清除会话能力。

### 阶段 C：实现后端流式接口

- 增加 `astream_events` 封装。
- 统一生成 `token`、`done`、`error` SSE 事件。
- 将同步 `invoke` 替换为异步调用。
- 处理客户端断开和模型异常。

### 阶段 D：实现前端流式渲染

- 用 `ReadableStream` 读取响应。
- 正确解析可能被拆包的 SSE 数据。
- 逐步追加回答文本。
- 增加停止生成、错误提示和新会话按钮。

### 阶段 E：持久化和质量提升

- 将 `InMemorySaver` 替换为 PostgreSQL checkpointer。
- 增加会话表、权限校验和历史会话列表。
- 增加检索来源、稳定 ID 和文档去重。
- 使用 token 预算或摘要管理长上下文。
- 增加日志、耗时、首 token 时间、检索命中率等指标。

## 9. 验收标准

完成基础升级后，至少应通过以下测试：

### 记忆

1. 同一个 `session_id` 先问“介绍一下青岛”，再问“它有哪些特点”，第二问能正确理解“它”。
2. 新建另一个 `session_id` 后，不能读取第一个会话的历史。
3. 服务重启后，如果使用内存 checkpointer，历史消失是预期行为；如果使用 PostgreSQL checkpointer，历史仍然存在。
4. 长对话不会无限增长，也不会出现孤立的 tool call/tool result 消息。

### 流式

1. 后端响应头为 `text/event-stream`。
2. 前端能在完整回答结束前显示第一段文本。
3. 一次网络读取包含多条事件，或一条事件被拆成多次读取时，解析都正确。
4. 模型异常能显示错误，按钮能恢复可用状态。
5. 用户取消请求后，服务端不再继续无意义地生成。

### RAG

1. 命中文档时返回稳定的命中状态和来源信息。
2. 未命中时不会凭空声称知识库中存在答案。
3. 重复导入同一个文件不会无限增加重复向量。
4. 修改文档后可以识别新版本并重新建立对应切片。

## 10. 最终目标结构

完成上述升级后，项目的主要调用关系应变成：

```text
浏览器
  |  session_id + question
  v
FastAPI /v1/chat/stream
  |  StreamingResponse
  v
Agent astream_events
  |----------------------|
  v                      v
PostgreSQL checkpointer  PGVectorStore
聊天历史                 文档检索
  |                      |
  +------ 模型生成回答 ---+
             |
       token / done / error
             |
           浏览器
```

最小可行路径是先完成“`InMemorySaver` + `session_id`”，确认多轮记忆可用后，再完成“`astream_events` + SSE + 前端 `ReadableStream`”。持久化、来源引用、问题改写和文档幂等可以在这条主链路稳定后继续增加。
