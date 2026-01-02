import asyncio
import operator
from typing import Optional, Union, TypedDict, Annotated


from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langgraph.constants import END, START
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode, create_react_agent
from pydantic import BaseModel
from pydantic.v1 import Field

from lgraph.tools.weather import get_weather


class WeatherInfo(BaseModel):
    """Extracted weather info from a specific city"""
    text: str = Field(description="The weather in the city")
    windDir:str = Field(description="The wind direction of the city")
    windSpeed:str = Field(description="The wind speed of the city")

@tool(args_schema=WeatherInfo)
def insert_db(text,windDir,windSpeed):
    """Insert weather information into the database."""
    print("insert db：",text,windDir,windSpeed)

class QueryWeatherSchema(BaseModel):
    """Schema for query weather info by city code"""
    city_code: str = Field(description="The location code of the city")

@tool(args_schema=QueryWeatherSchema)
def query_weather_from_db(city_code:str):
    """Query weather info by city code from db"""
    print("query db：",city_code)
    return f"{city_code},text:大雨，windDir:西北风"

chat_model = ChatDeepSeek(model="deepseek-chat", temperature=0.7)
tools = [get_weather,insert_db,query_weather_from_db]
tool_node = ToolNode(tools)

class State(TypedDict):
    messages: Annotated[list,add_messages]

def should_continue(state:State):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    else:
        return "tools"

async def call_model(state:State,config:RunnableConfig):
    messages = state["messages"]
    response = await chat_model.ainvoke(messages,config)
    return {"messages": [response]}

def call():
    graph = create_react_agent(chat_model,tools)
    resp = graph.invoke({"messages":["请帮我查询地区编码为:101010100的天气"]})
    print(resp)
    # image_data = graph.get_graph(xray=True).draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(image_data)
    # print("Graph saved as graph.png")
    """ create_react_agent 相当于做了如下操作
        work_flow = StateGraph(State)
        work_flow.add_node("agent",call_model)
        work_flow.add_node("tools",tool_node)
    
        work_flow.add_edge(START,"agent")
    
        work_flow.add_conditional_edges("agent",should_continue,["tools",END])
    
        work_flow.add_edge("tools","agent")
        app = work_flow.compile()
    """

def print_stream(stream):
    for sub_stream in stream:
        print(sub_stream)

async def print_astream():
    input_messages = {"messages": ["北京天气怎么样，北京的城市编码为:101010100"]}
    graph = create_react_agent(chat_model, tools)
    async for chunk in graph.astream(input=input_messages,stream_mode="values"):
        chunk["messages"][-1].pretty_print()

def call2():
    input_messages = {"messages": ["给我讲个笑话"]}
    graph = create_react_agent(chat_model, tools)
    print_stream(graph.stream(input_messages,stream_mode="values"))
if __name__ == '__main__':
    asyncio.run(print_astream())