import json
import os

from langchain_core.output_parsers import JsonOutputKeyToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from utils.weather import get_weather


def final_response(ai_message: str) -> str:
    print('ai_message:', ai_message)
    # data = json.loads(ai_message)
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system","这是实时的天气数据，详细信息是:{detail}"),
            ("system","请你解析该数据，以自然语言的形式回复")
        ]
    )
    messages = chat_template.format_messages(detail=ai_message)

    # 配置 API Key 和 Base URL
    os.environ['OPENAI_API_KEY'] = os.getenv("DEEPSEEK_API_KEY")  # 此处填写你的 DeepSeek API Key

    # 初始化模型，指向 DeepSeek 的 API 端点
    llm = ChatOpenAI(
        model="deepseek-chat",        # 或其他 DeepSeek 模型，如 'deepseek-reasoner'
        base_url="https://api.deepseek.com", # DeepSeek 的 API 地址
        temperature=0.7,
    )

    response = llm.invoke(messages)
    return response.content

def call():
    # 配置 API Key 和 Base URL
    os.environ['OPENAI_API_KEY'] = os.getenv("DEEPSEEK_API_KEY")  # 此处填写你的 DeepSeek API Key

    # 初始化模型，指向 DeepSeek 的 API 端点
    llm = ChatOpenAI(
        model="deepseek-chat",        # 或其他 DeepSeek 模型，如 'deepseek-reasoner'
        base_url="https://api.deepseek.com", # DeepSeek 的 API 地址
        temperature=0.7,
    )
    llm_with_tools = llm.bind_tools([get_weather])
    response = llm_with_tools.invoke("北京天气怎么样")
    print(response)

    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system","这是实时的天气数据：{weather_data}"),
            ("human","{user_input}")
        ]
    )

    chain = llm_with_tools | JsonOutputKeyToolsParser(key_name="get_weather",first_tool_only=True) | get_weather | final_response
    weather_data = chain.invoke("今天北京天气怎么样,北京编码为:101010100")

    messages = chat_template.format_messages(weather_data=weather_data,user_input="今天北京天气怎么样,北京编码为:101010100")

    response = llm.invoke(messages)
    print(response.content)



if __name__ == '__main__':
    call()