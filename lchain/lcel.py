import os
from operator import itemgetter
from typing import Optional

from langchain_classic.chains.base import Chain
from langchain_classic.chains.structured_output import create_openai_fn_runnable
from langchain_core.runnables import RunnableLambda
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputKeyToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
import requests


@tool
def multiply(a: int, b: int) -> Optional[int]:
    """将2个数相乘"""
    return a * b


@tool
def get_weather(location: str) -> dict:
    """查询实时天气 (和风天气代理)。

    参数:
      location: 地点编码 (例如 101010100)
    返回: 含原始响应/状态的字典。
    """
    # 当前为模拟实现：直接返回固定格式描述字符串。
    # 如需真实天气，可替换为实际 HTTP 请求逻辑。
    return f"地点编码为{location}的城市，今日晴天，温度30度，微风，适合出行。"


class RecordPerson(BaseModel):
    '''记录一个人的一些识别信息'''
    name: str = Field(..., description="名字")
    age: int = Field(..., description="年龄")
    fav_food: Optional[str] = Field(None, description="这个人喜欢的食物")


class RecordDog(BaseModel):
    '''记录关于一只狗的一些识别信息'''
    name: str = Field(..., description="这只狗的名字")
    color: str = Field(..., description="这只狗的颜色")
    fav_food: Optional[str] = Field(None, description="这只狗最喜欢的食物")


def call():
    os.environ['OPENAI_API_KEY'] = os.getenv("DEEPSEEK_API_KEY")
    chat_model = ChatDeepSeek(model="deepseek-chat", temperature=0.7)

    parser = PydanticOutputParser(pydantic_object=RecordDog)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个信息提取助手。请从给定的文本中提取信息。
            {format_instructions}"""),
        ("human", "请从以下文本中提取关于狗的信息：{input}")
    ])

    partial_prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    chain = partial_prompt | chat_model | parser

    input = "哈里是一只胖乎乎的棕色比格犬，非常喜欢吃鸡肉"

    try:
        result = chain.invoke({"input": input})
        print("提取成功！")
        print(f"结构化结果: {result}")

    except Exception as e:
        print(f"解析过程中出现错误: {e}")


def final_response(ai_message: AIMessage) -> str:
    tools = [get_weather, multiply]
    tool_map = {tool.name: tool for tool in tools}
    chosen_tool = tool_map[ai_message["name"]]

    return itemgetter("arguments") | chosen_tool


def call_two():
    os.environ['OPENAI_API_KEY'] = os.getenv("DEEPSEEK_API_KEY")
    chat_model = ChatDeepSeek(model="deepseek-chat", temperature=0.7)
    functions = [get_weather, multiply]

    openai_functions = [convert_to_openai_function(f) for f in functions]
    print(openai_functions)
    print()
    tools = [get_weather, multiply]
    print('tools', tools)

    # structured_llm = create_openai_fn_runnable(functions, chat_model) | itemgetter("arguments") | get_weather
    structured_llm = create_openai_fn_runnable(functions, chat_model) | final_response
    str_resp = structured_llm.invoke("请帮我查询地区编码为:101010100的天气")

    tool_map = {tool.name: tool for tool in tools}
    print(str_resp)


# 创建工具调用链
def create_tool_chain():
    tools = [get_weather, multiply]
    tool_map = {tool.name: tool for tool in tools}
    print('tool_map', tool_map)
    # 将工具描述转换为 DeepSeek 格式
    tool_descriptions = []
    for tool_func in tools:
        tool_descriptions.append({
            "type": "function",
            "function": {
                "name": tool_func.name,
                "description": tool_func.description,
                "parameters": {
                    "type": "object",
                    "properties": tool_func.args_schema.schema()["properties"],
                    "required": list(tool_func.args_schema.schema()["properties"].keys())
                }
            }
        })

    return tool_map, tool_descriptions


# 执行工具调用
def execute_with_tools(user_input):
    tools_map, tool_descriptions = create_tool_chain()
    chat_model = ChatDeepSeek(model="deepseek-chat")
    response = chat_model.invoke(
        [HumanMessage(content=user_input)],
        tools=tool_descriptions
    )
    print('工具响应',response)
    # 解析响应并执行工具
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            # 找到对应的工具函数
            for name, tool_func in tools_map.items():
                if name == tool_call['name']:
                    result = tool_func.run(tool_call['args'])
                    print(f"工具执行结果: {result}")
                    return result

    return response.content


class GetWeatherToolChain(Chain):
    """将当前文件中的工具调用流程包装成单输入单输出的 Chain。
    输入: "input" (自然语言，包含地点或编码)
    输出: "text" (工具执行后的结果字符串)
    可直接放入 SimpleSequentialChain 第一环。"""

    @property
    def input_keys(self) -> List[str]:
        return ["input"]

    @property
    def output_keys(self) -> List[str]:
        return ["text"]

    def _call(self, inputs: Dict, run_manager=None) -> Dict:
        query = inputs.get("input", "")
        # 直接复用 execute_with_tools 逻辑
        result = execute_with_tools(query)
        # 统一转成字符串
        return {"text": result if isinstance(result, str) else str(result)}


def create_get_weather_chain() -> Chain:
    """工厂方法：返回一个可嵌入 SimpleSequentialChain 的天气获取 Chain。"""
    return GetWeatherToolChain()




if __name__ == '__main__':
    # call_two()
    chain = create_get_weather_chain()
    response = chain.invoke("请帮我查询地区编码为:101010100的天气")
    print('最后',response)
    # execute_with_tools("请帮我查询地区编码为:101010100的天气")
    # print(get_weather("101010100"))
    # chain = create_advanced_llm_chain()
    # result = chain.invoke("今天北京天气怎么样,北京编码为:101010100")
    # print('result',result)