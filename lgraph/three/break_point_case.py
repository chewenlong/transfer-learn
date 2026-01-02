import asyncio
from typing import TypedDict

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

from lgraph.tools.weather import get_weather, query_weather_from_db, insert_weather_to_db, delete_weather_from_db, \
    update_weather_to_db


class State(TypedDict):
    user_input: str
    model_response: str
    user_approval:str


llm = ChatDeepSeek(model="deepseek-chat", temperature=0.7)
tools = [get_weather,query_weather_from_db,insert_weather_to_db,delete_weather_from_db,update_weather_to_db]
tool_node = ToolNode(tools)
llm = llm.bind_tools(tools)

def call_model(state):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    elif last_message.tool_calls[0]["name"]=="delete_weather_from_db":
        return "run_tool"
    else :
        return "continue"

def run_tool(state):
    new_messages = []
    tool_calls = state["messages"][-1].tool_calls
    tools = [delete_weather_from_db]
    tools = {t.name:t  for t in tools }

    for tool_call in tool_calls:
        tool = tools[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        new_messages.append({
            "role":"tool",
            "name":tool_call["name"],
            "content":result,
            "tool_call_id":tool_call["id"],
        })
    return {"messages":new_messages}



async def call():
    work_flow = StateGraph(MessagesState)

    work_flow.add_node("agent",call_model)
    work_flow.add_node("action",tool_node)
    work_flow.add_node("run_tool",run_tool)

    work_flow.add_edge(START,"agent")
    work_flow.add_conditional_edges("agent",should_continue,{
        "continue":"action",
        "run_tool":"run_tool",
        "end":END
    })

    work_flow.add_edge("action","agent")
    work_flow.add_edge("run_tool","agent")

    memory = MemorySaver()
    graph = work_flow.compile(checkpointer=memory,interrupt_before=["run_tool"])

    # image_data = graph.get_graph(xray=True).draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(image_data)
    # print("Graph saved as graph.png")

    config = {"configurable":{"thread_id": "1"}}
    # async for chunk in graph.astream({"messages":"北京天气怎么样，北京的城市编码为:101010100，如果查到则保存数据"},config, stream_mode="values"):
    #     chunk["messages"][-1].pretty_print()
    async for chunk in graph.astream({"messages":"删除北京天气数据，北京的城市编码为:101010100"},config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()

    async for chunk in graph.astream(None,config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
if __name__ == '__main__':
    asyncio.run(call())