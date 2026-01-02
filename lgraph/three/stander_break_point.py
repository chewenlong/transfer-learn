import asyncio
from typing import TypedDict

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

from lgraph.tools.weather import *


class State(TypedDict):
    user_input: str
    model_response: str
    user_approval:str


llm = ChatDeepSeek(model="deepseek-chat", temperature=0.7)

def call_model(state):
    messages = state["user_input"]
    if '删除' in messages:
        state['user_approval']=f"用户输入的指令是{messages}，请人工确认是否执行"
    else:
        response = llm.invoke(messages)
        state['user_approval']="直接运行"
        state['model_response']=response
    return state

def execute_users(state):
    user_msg = state["user_approval"]
    if user_msg == '是':
        response = "您的删除请求已经获得管理员的批准并成功执行。如果您有其他问题或需要进一步的帮助，请随时联系我们"
        return {"model_response":AIMessage(response)}
    elif user_msg == '否':
        response = "对不起，您当前的请求是高风险操作，管理员不允许执行！"
        return {"model_response": AIMessage(response)}
    else:
        return state

def translate_message(state):
    system_prompt = """翻译为法语"""
    messages = state["model_response"]
    messages = [SystemMessage(content=system_prompt)] + [HumanMessage(content=messages.content)]

    response = llm.invoke(messages)
    return {"model_response": response}

async def call():
    builder = StateGraph(State)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_users", execute_users)
    builder.add_node("translate_message", translate_message)

    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", "execute_users")
    builder.add_edge("execute_users", "translate_message")
    builder.add_edge("translate_message", END)

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory,interrupt_before=["execute_users"])

    # image_data = graph.get_graph(xray=True).draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(image_data)
    # print("Graph saved as graph.png")

    config = {"configurable":{"thread_id": "1"}}
    async for chunk in graph.astream({"user_input": "我将在数据库中删除 id 为 muyu 的所有信息"}, config, stream_mode="values"):
        print(chunk)

    snapshot = graph.get_state(config)
    snapshot.values['user_approval']='是'
    graph.update_state(config, snapshot.values)
    async for chunk in graph.astream(None, config, stream_mode="values"):
        print(chunk)

    async for chunk in graph.astream({"user_input": "我将在数据库中删除 id 为 muyu 2222222的所有信息"}, config, stream_mode="values"):
        print(chunk)

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

tools = [get_weather,multiply]

def call_model(state):
    model_with_tools = llm.bind_tools(tools)
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

async def call2():
    tools_node = ToolNode(tools)

    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", call_model)
    workflow.add_node("action", tools_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent",should_continue,{
        "continue":"action",
        "end":END,
    })
    workflow.add_edge("action","agent")

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory,interrupt_before=["action"])
    config = {"configurable":{"thread_id": "1"}}
    async for chunk in graph.astream({"messages":"北京天气怎么样，北京的城市编码为:101010100"},config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
    async for chunk in graph.astream(None, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()

tools1 = [get_weather,query_weather_from_db,insert_weather_to_db,delete_weather_from_db,update_weather_to_db]
tool_node = ToolNode(tools1)

if __name__ == '__main__':
    asyncio.run(call2())