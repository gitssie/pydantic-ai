"""Simple chat app example build with FastAPI.

Run with:

    uv run -m pydantic_ai_examples.chat_app
"""

from __future__ import annotations as _annotations

import asyncio
import json
import sqlite3
import uuid
from collections.abc import AsyncIterator
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, TypeVar, cast

from duckduckgo_search import DDGS
import fastapi
from httpx import AsyncClient, AsyncHTTPTransport
import logfire
from fastapi import Depends, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from bs4 import BeautifulSoup
import markdownify

from pydantic_ai import RunContext
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.common_tools.tavily import tavily_search_tool
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai_examples.code_agent import CodeAgent
from typing_extensions import LiteralString, ParamSpec, TypedDict
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
    PartStartEvent,
    PartDeltaEvent,
    TextPartDelta,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ToolCallPart,
)
from pydantic_ai.agent import Agent
from pydantic_ai.mcp import MCPServerHTTP

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

# 使用指定的模型名称
model_name = 'gemini-2.0-flash'
print(f'Using model: {model_name}')
# 设置代理
proxy = "http://172.16.20.19:3213"
# 使用正确的 Gemini 模型格式和代理设置
transport = AsyncHTTPTransport(proxy=proxy)
custom_http_client = AsyncClient(transport=transport, timeout=30)
# 设置Gemini API密钥
api_key = ""
model = GeminiModel( 
    model_name,
    provider=GoogleGLAProvider(api_key=api_key, http_client=custom_http_client),
)

model = OpenAIModel(
    model_name='gemini-2.0-flash',
    provider=OpenAIProvider(base_url='https://generativelanguage.googleapis.com/v1beta/openai/',api_key=api_key, http_client=custom_http_client),
)

model = OpenAIModel(
    model_name='deepseek-chat',
    provider=OpenAIProvider(base_url='https://api.deepseek.com/v1',api_key=api_key),
)


# 创建Playwright MCP服务器
playwright_mcp_server = MCPServerHTTP(url='http://localhost:3000/sse')
python_mcp_server = MCPServerHTTP(url='http://localhost:3001/sse')

# 创建代理实例，将MCP服务器添加到代理中
agent = CodeAgent[Any, Any](
    model,
    instrument=False,
    output_type=str,
    retries=2,
    mcp_servers=[python_mcp_server],  # 添加MCP服务器
    tools=[tavily_search_tool(api_key="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")],
)
THIS_DIR = Path(__file__).parent

@agent.tool_plain
def get_lat_lng(location_description: str
) -> dict[str, float]:
    """Get the latitude and longitude of a location.

    Args:
        ctx: The context.
        location_description: A description of a location.
    """
    return {'lat': 39.9, 'lng': 116.4074}




@agent.tool
def get_weather(ctx: RunContext[Any], lat: float, lng: float) -> dict[str, Any]:
    """Get the weather at a location.

    Args:
        ctx: The context.
        lat: Latitude of the location.
        lng: Longitude of the location.
    """
    return {'temperature': '21 °C', 'description': 'Sunny'}

@agent.tool
async def fetch_url_as_markdown(ctx: RunContext[Any], url: str, max_length: int = 1500) -> str:
    """获取URL的内容并以Markdown格式返回。
    
    使用BeautifulSoup解析HTML内容并使用markdownify将HTML转换为Markdown。
    
    Args:
        ctx: 运行上下文。
        url: 要获取的URL地址。
        max_length: 返回的Markdown内容的最大长度，默认为1500字符。
    
    Returns:
        str: 页面内容的Markdown格式表示。
    """
    try:
        # 使用已经配置好的HTTP客户端
        response = await custom_http_client.get(url, follow_redirects=True)
        response.raise_for_status()
        
        # 获取内容类型
        content_type = response.headers.get('content-type', '').lower()
        
        # 根据内容类型处理响应
        if 'text/html' in content_type or 'application/xhtml+xml' in content_type:
            try:
                # 使用BeautifulSoup解析HTML
                html_content = response.text
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # 提取标题
                title = soup.title.text.strip() if soup.title else "无标题"
                
                # 尝试找到主要内容区域
                main_content = None
                for selector in ['main', 'article', '#content', '.content', '.main-content']:
                    if soup.select(selector):
                        main_content = soup.select(selector)[0]
                        break
                
                # 如果没有找到主要内容区域，使用body
                body = main_content if main_content else (soup.body if soup.body else soup)
                
                # 移除一些不必要的元素
                for tag in body.find_all(['script', 'style', 'nav', 'footer', 'aside']):
                    tag.decompose()
                
                # 创建最终的Markdown
                result_markdown: str = f"# {title}\n\n"
                result_markdown += f"## URL\n[{url}]({url})\n\n"
                result_markdown += "## 网页内容\n"
                
                # 使用cast确保类型检查器知道这是一个字符串
                try:
                    body_markdown: str = cast(str, markdownify.markdownify(str(body), heading_style="ATX"))  # type: ignore
                    if len(body_markdown) > max_length:
                        result_markdown += body_markdown[:max_length] + f"\n\n*内容已截断，完整内容超过{max_length}字符...*"
                    else:
                        result_markdown += body_markdown
                except Exception:
                    # 如果markdownify转换失败，使用简单的文本提取
                    text_content = body.get_text(separator='\n', strip=True)
                    result_markdown += f"*HTML转Markdown失败，显示纯文本内容：*\n\n{text_content[:max_length]}"
                    if len(text_content) > max_length:
                        result_markdown += f"\n\n*内容已截断，完整内容超过{max_length}字符...*"
                
                return result_markdown
            except Exception as parse_error:
                # HTML解析错误，返回原始内容的一部分
                return f"# HTML解析错误 - {url}\n\n解析HTML时出错：{str(parse_error)}\n\n```\n{response.text[:500]}...\n```"
            
        elif 'application/json' in content_type:
            # 对于JSON内容，格式化输出
            try:
                json_data = response.json()
                result_markdown: str = f"# JSON内容 - {url}\n\n"
                result_markdown += "```json\n"
                json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
                if len(json_str) > max_length:
                    result_markdown += json_str[:max_length] + "\n...(内容已截断)..."
                else:
                    result_markdown += json_str
                result_markdown += "\n```\n"
                
                return result_markdown
            except json.JSONDecodeError:
                return f"# JSON解析错误 - {url}\n\n无法解析响应为有效的JSON格式\n\n```\n{response.text[:500]}...\n```"
        else:
            # 对于其他类型的内容
            result_markdown: str = f"# 内容 - {url}\n\n"
            result_markdown += f"## 内容类型\n{content_type}\n\n"
            result_markdown += f"## 响应内容（前{min(500, max_length)}字符）\n```\n{response.text[:min(500, max_length)]}...\n```\n"
            
            return result_markdown
    except Exception as e:
        return f"# 错误\n获取URL '{url}' 时发生错误：\n\n```\n{str(e)}\n```"

@asynccontextmanager
async def lifespan(_app: fastapi.FastAPI):
    async with Database.connect() as db:
        yield {'db': db}


app = fastapi.FastAPI(lifespan=lifespan)
logfire.instrument_fastapi(app)


@app.get('/')
async def index() -> FileResponse:
    return FileResponse((THIS_DIR / 'chat_app.html'), media_type='text/html')


@app.get('/chat_app.ts')
async def main_ts() -> FileResponse:
    """Get the raw typescript code, it's compiled in the browser, forgive me."""
    return FileResponse((THIS_DIR / 'chat_app.ts'), media_type='text/plain')


async def get_db(request: Request) -> Database:
    return request.state.db


@app.get('/chat/')
async def get_chat(database: Database = Depends(get_db)) -> Response:
    msgs = await database.get_messages()
    return Response(
        b'\n'.join(json.dumps(to_chat_message(m), ensure_ascii=False).encode('utf-8') for m in msgs),
        media_type='text/plain',
    )


class ChatMessage(TypedDict):
    """Format of messages sent to the browser."""

    role: Literal['user', 'model']
    timestamp: str
    content: str
    uid: str  # 添加 uid 字段，用于关联同一轮对话的消息


def to_chat_message(m: ModelMessage, conversation_uid: str = "") -> ChatMessage:
    # 如果没有提供对话ID，生成一个默认值
    uid = conversation_uid if conversation_uid else str(uuid.uuid4())
    
    # 初始化content为空字符串
    content = ""
    timestamp = datetime.now(tz=timezone.utc)
    role = "model"  # 默认为模型角色
    
    if isinstance(m, ModelRequest):
        role = "user"
        
        # 扫描所有部分查找内容
        for part in m.parts:
            if isinstance(part, UserPromptPart):
                assert isinstance(part.content, str)
                content = part.content
                timestamp = part.timestamp
                break
                
    elif isinstance(m, ModelResponse):
        role = "model"
        
        # 扫描所有部分查找内容
        for part in m.parts:
            timestamp = m.timestamp  # 使用消息时间戳
            
            if isinstance(part, TextPart):
                content = part.content
                break
            elif isinstance(part, ToolCallPart):
                # 格式化工具调用信息
                content = f"工具: {part.tool_name}"
                break
    
    # 返回格式化的消息
    return {
        'role': role,
        'timestamp': timestamp.isoformat(),
        'content': content,
        'uid': uid
    }


@app.post('/chat/')
async def post_chat(
    prompt: Annotated[str, fastapi.Form()], database: Database = Depends(get_db)
) -> StreamingResponse:
    async def stream_messages():
        """Streams new line delimited JSON `Message`s to the client."""
        # 为本轮对话生成唯一ID
        conversation_uid = str(uuid.uuid4())
        
        # stream the user prompt so that can be displayed straight away
        yield (
            json.dumps(
                {
                    'role': 'user',
                    'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                    'content': prompt,
                    'uid': conversation_uid  # 添加对话ID
                },
                ensure_ascii=False  # 不使用ASCII编码，保持中文原样
            ).encode('utf-8')
            + b'\n'
        )
        
        # 获取聊天历史记录作为上下文传递给代理
        messages = await database.get_messages()
        new_message_index = len(messages)
        async with agent.run_mcp_servers():
            # 使用 agent.iter 初始化迭代器，支持节点级别的迭代
            async with agent.iter(prompt, message_history=messages) as run:
                # 使用 async for 循环自动处理节点迭代
                model_timestamp = datetime.now(tz=timezone.utc).isoformat()
                
                async for node in run:
                    # 结束节点 - 代理执行完成
                    if Agent.is_end_node(node):
                        # 发送[DONE]标记，表示流式响应结束
                        done_message = {
                            'role': 'model',
                            'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                            'content': '[DONE]',
                            'uid': conversation_uid
                        }
                        yield json.dumps(done_message, ensure_ascii=False).encode('utf-8') + b'\n'
                        # 不再重复发送之前已发送的内容
                    
                    # 模型请求节点 - 处理模型生成的流式响应
                    elif Agent.is_model_request_node(node):
                        async with node.stream(run.ctx) as request_stream:
                            async for event in request_stream:
                                if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                    # 直接使用增量内容，不累加
                                    delta_text = event.delta.content_delta
                                    # 创建模型响应并发送增量更新
                                    m = ModelResponse(parts=[TextPart(delta_text)], timestamp=datetime.fromtimestamp(0, tz=timezone.utc))
                                    response_msg = to_chat_message(m, conversation_uid)
                                    # 使用相同的时间戳，以便前端将增量更新应用到同一条消息
                                    response_msg['timestamp'] = model_timestamp
                                    yield json.dumps(response_msg, ensure_ascii=False).encode('utf-8') + b'\n'
                                elif isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                                    # 输出新文本部分的开始
                                    text = event.part.content
                                    m = ModelResponse(parts=[TextPart(text)], timestamp=datetime.now(tz=timezone.utc))
                                    response_msg = to_chat_message(m, conversation_uid)
                                    model_timestamp = response_msg['timestamp']  # 更新时间戳
                                    yield json.dumps(response_msg, ensure_ascii=False).encode('utf-8') + b'\n'
                    
                    # 工具调用节点 - 处理工具调用和结果
                    elif Agent.is_call_tools_node(node):
                        async with node.stream(run.ctx) as handle_stream:
                            async for event in handle_stream:
                                if isinstance(event, FunctionToolCallEvent):
                                    # 工具调用事件
                                    # 检测是否是Playwright MCP工具调用
                                    tool_name = getattr(event.part, 'tool_name', "未知工具")
                                    tool_info = f"工具: {tool_name}"
                                    m = ModelResponse(parts=[TextPart(tool_info)], timestamp=datetime.now(tz=timezone.utc))
                                    response_msg = to_chat_message(m, conversation_uid)
                                    yield json.dumps(response_msg, ensure_ascii=False).encode('utf-8') + b'\n'
                                elif isinstance(event, FunctionToolResultEvent):
                                    # 工具结果事件
                                    # 检测是否是Playwright MCP工具结果
                                    tool_call_id = getattr(event, 'tool_call_id', "未知ID")
                                    result_info = f"结果: {tool_call_id}"
                                    m = ModelResponse(parts=[TextPart(result_info)], timestamp=datetime.now(tz=timezone.utc))
                                    response_msg = to_chat_message(m, conversation_uid)
                                    yield json.dumps(response_msg, ensure_ascii=False).encode('utf-8') + b'\n'
                    else:
                        # 其他节点类型
                        other_info = f"\n其他节点: {type(node).__name__}"
                        m = ModelResponse(parts=[TextPart(other_info)], timestamp=datetime.now(tz=timezone.utc))
                        response_msg = to_chat_message(m, conversation_uid)
                        yield json.dumps(response_msg, ensure_ascii=False).encode('utf-8') + b'\n'

            # 将消息列表转换为JSON字节
            messages_json = ModelMessagesTypeAdapter.dump_json(run.ctx.state.message_history[new_message_index:])
            await database.add_messages(messages_json)

    return StreamingResponse(stream_messages(), media_type='text/plain')


P = ParamSpec('P')
R = TypeVar('R')


@dataclass
class Database:
    """Rudimentary database to store chat messages in SQLite.

    The SQLite standard library package is synchronous, so we
    use a thread pool executor to run queries asynchronously.
    """

    con: sqlite3.Connection
    _loop: asyncio.AbstractEventLoop
    _executor: ThreadPoolExecutor

    @classmethod
    @asynccontextmanager
    async def connect(
        cls, file: Path = THIS_DIR / '.chat_app_messages.sqlite'
    ) -> AsyncIterator[Database]:
        with logfire.span('connect to DB'):
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            con = await loop.run_in_executor(executor, cls._connect, file)
            slf = cls(con, loop, executor)
        try:
            yield slf
        finally:
            await slf._asyncify(con.close)

    @staticmethod
    def _connect(file: Path) -> sqlite3.Connection:
        con = sqlite3.connect(str(file))
        con = logfire.instrument_sqlite3(con)
        cur = con.cursor()
        cur.execute(
            'CREATE TABLE IF NOT EXISTS messages (id INT PRIMARY KEY, message_list TEXT);'
        )
        con.commit()
        return con

    async def add_messages(self, messages: bytes):
        await self._asyncify(
            self._execute,
            'INSERT INTO messages (message_list) VALUES (?);',
            messages,
            commit=True,
        )
        await self._asyncify(self.con.commit)

    async def get_messages(self) -> list[ModelMessage]:
        c = await self._asyncify(
            self._execute, 'SELECT message_list FROM messages ORDER BY id DESC LIMIT 4'
        )
        rows = await self._asyncify(c.fetchall)
        print(len(rows))
        messages: list[ModelMessage] = []
        for row in reversed(rows):
            messages.extend(ModelMessagesTypeAdapter.validate_json(row[0]))
        return messages

    def _execute(
        self, sql: LiteralString, *args: Any, commit: bool = False
    ) -> sqlite3.Cursor:
        cur = self.con.cursor()
        cur.execute(sql, args)
        if commit:
            self.con.commit()
        return cur

    async def _asyncify(
        self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        return await self._loop.run_in_executor(  # type: ignore
            self._executor,
            partial(func, **kwargs),
            *args,  # type: ignore
        )


if __name__ == '__main__':

    import uvicorn
    
    uvicorn.run(
        'pydantic_ai_examples.chat_app:app', reload=True, reload_dirs=[str(THIS_DIR)]
    )
