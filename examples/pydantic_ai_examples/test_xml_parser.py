import asyncio
import pytest
from typing import Dict, List, TypedDict

from xml_parser_model import XMLNode


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
    test_data = """<thought>
我需要查询本月的销售订单。我可以使用 `sale_chart` 工具来完成这个任务。为了使用这个工具，我需要确定本月的开始日期和结束日期。我可以使用 python 的 `datetime` 模块来计算这些日期。
</thought>
<py>
import datetime

today = datetime.date.today()
first_day_of_month = datetime.date(today.year, today.month, 1)
last_day_of_month = datetime.date(today.year, today.month, calendar.monthrange(today.year, today.month)[1])

print(first_day_of_month)
print(last_day_of_month)
</py>
<observation>
2024-07-01
2024-07-31
</observation>
<py>
print(sale_chart(date_start=datetime.date(2024, 7, 1), date_end=datetime.date(2024, 7, 31)))
</py>"""
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