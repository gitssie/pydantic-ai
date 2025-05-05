"""Example of PydanticAI with multiple tools which the LLM needs to call in turn to answer a question.

In this case the idea is a "weather" agent — the user can ask for the weather in multiple cities,
the agent will use the `get_lat_lng` tool to get the latitude and longitude of the locations, then use
the `get_weather` tool to get the weather.

Run with:

    uv run -m pydantic_ai_examples.weather_agent
    
Note: Set the `GEMINI_API_KEY` environment variable to use the Gemini model.
"""

from __future__ import annotations as _annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import logfire
from httpx import AsyncClient, AsyncHTTPTransport

from pydantic_ai.agent import Agent
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent, PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai_examples.code_agent import CodeAgent

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')


@dataclass
class Deps:
    client: AsyncClient
    weather_api_key: str | None
    geo_api_key: str | None



# 使用指定的模型名称
model_name = 'gemini-2.0-flash'
print(f'Using model: {model_name}')
# 设置代理
proxy = "http://192.168.31.119:3213"
# 使用正确的 Gemini 模型格式和代理设置
transport = AsyncHTTPTransport(proxy=proxy)
custom_http_client = AsyncClient(transport=transport, timeout=30)
# 设置Gemini API密钥
gemini_api_key = ""
gemini_model = GeminiModel( 
    model_name,
    provider=GoogleGLAProvider(api_key=gemini_api_key, http_client=custom_http_client),
)

gemini_model = OpenAIModel(
    model_name='gemini-2.0-flash',  
    provider=OpenAIProvider(base_url='https://generativelanguage.googleapis.com/v1beta/openai/',api_key=gemini_api_key, http_client=custom_http_client),  
)


weather_agent = CodeAgent[Deps, str](
    model=gemini_model,
    deps_type=Deps,
    output_type=str,
    retries=2,
    instrument=False,
)

@weather_agent.tool_plain
def get_lat_lng(location_description: str
) -> dict[str, float]:
    """Get the latitude and longitude of a location.

    Args:
        ctx: The context.
        location_description: A description of a location.
    """
    return {'lat': 39.9, 'lng': 116.4074} 




@weather_agent.tool_plain
def get_weather(lat: float, lng: float) -> dict[str, Any]:
    """Get the weather at a location.

    Args:
        ctx: The context.
        lat: Latitude of the location.
        lng: Longitude of the location.
    """
    return {'temperature': '21 °C', 'description': 'Sunny'}


async def main():
    async with AsyncClient() as client:
        # create a free API key at https://www.tomorrow.io/weather-api/
        weather_api_key = os.getenv('WEATHER_API_KEY')
        # create a free API key at https://geocode.maps.co/
        geo_api_key = os.getenv('GEO_API_KEY')
        deps = Deps(
            client=client, weather_api_key=weather_api_key, geo_api_key=geo_api_key
        )
        
        # print("示例1: 使用 stream_text() 流式输出完整文本")
        # async with weather_agent.run_stream(
        #     '查询本月销售订单', deps=deps
        # ) as result:
        #     async for message in result.stream_text():
        #         print(f"收到: {message}")
        #
        #     # 完成后获取最终输出
        #     print("最终输出:", await result.get_output())
        
        # print("\n示例2: 使用 stream_text(delta=True) 流式输出增量文本")
        # async with weather_agent.run_stream(
        #     '查询北京天气', deps=deps
        # ) as result:
        #     async for message in result.stream():
        #         print(message,end="", flush=True)
            
        #     print("最终输出:", await result.get_output())

        print("\n示例2: 使用 stream() 方法流式输出")
        async with weather_agent.run_stream("现在日期", deps=deps) as result:
            async def stream_output():
                async for text in result.stream(debounce_by=0.01):
                    print(text, end="", flush=True)
            
            await stream_output()

        # 添加新消息（例如用户提示和代理响应）到数据库
        print("\n新消息JSON:", result.new_messages_json())

        print("\n示例3: 监听工具调用和执行过程并支持增量文本输出")
        # 通过 Agent.iter() 方法直接获取底层的执行流程
        async with weather_agent.iter(
            '现在日期', deps=deps
        ) as run:
            # 使用 async for 循环自动处理节点迭代
            #current_text = ""
            
            async for node in run:
                if Agent.is_end_node(node):
                    if run.result is not None:
                        print(f"\n✅ 最终结果: {run.result.output}")
                    else:
                        print("\n✅ 执行完成，但没有最终结果")
                    break
                    
                elif Agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as request_stream:
                        async for event in request_stream:
                            if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                # 增量输出文本
                                print(event.delta.content_delta, end="", flush=True)
                            elif isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                                # 输出新文本部分的开始
                                print(event.part.content, end="", flush=True)
                                
                elif Agent.is_call_tools_node(node):
                    async with node.stream(run.ctx) as handle_stream:
                        async for event in handle_stream:
                            if isinstance(event, FunctionToolCallEvent):
                                print(f"⚙️ 调用工具: {event.part.tool_name}")
                                print(f"  参数: {event.part.args_as_dict()}")
                                print(f"  调用ID: {event.call_id}")
                            elif isinstance(event, FunctionToolResultEvent):
                                print(f"📊 工具结果: {event.tool_call_id}")
                                print(f"  返回: {event.result.content}")
                else:
                    print(f"\n其他节点: {type(node).__name__}")


if __name__ == '__main__':
    asyncio.run(main())
