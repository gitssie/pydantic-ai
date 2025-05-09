import asyncio

from pydantic_ai_examples.xml_parser_model import XMLNode
import pytest
from typing import Dict, List, TypedDict



# 定义块类型，用于类型注解
class XMLBlock(TypedDict):
    type: str
    content: str
    tag: str


class TextBlock(TypedDict):
    type: str
    content: str


# 块可以是XML块或文本块
Block = Dict[str, str]


@pytest.mark.asyncio
async def test_xml_node():
    # 测试数据
    test_data = """切换标签页仍然失败。我将尝试另一种方法来获取黄金价格。我将使用AI搜索DeepSeek来查询今天的黄金价格。</thought>
<use_mcp_tool>
<tool_name>browser_click</tool_name>
<arguments>
{
  "element": "AI搜 DeepSeek-R1 帮你解答",
  "ref": "s2e135"
}</arguments>
</use_mcp_tool>"""
    # 测试1：分块处理数据
    print("\n1. 测试分块流式解析:")
    chunk_size = 20  # 每个块的大小
    chunks: List[str] = [test_data[i:i+chunk_size] for i in range(0, len(test_data), chunk_size)]
    node = XMLNode(buffer="", pos=0)
    #arr:List[XMLNode] = [node]
    for chunk in chunks:
        next = node.feed(chunk)
        if next:
            node = next
            if node.delta:
                print(node.delta, end="")


    node.complete()

    #print(arr)



if __name__ == "__main__":
    asyncio.run(test_xml_node()) 