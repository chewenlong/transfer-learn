import asyncio
import os
import uuid
from typing import Optional, Annotated

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, add_messages, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict

from lgraph.tools.weather import get_weather


class State(TypedDict):
    messages: Annotated[list,add_messages]

chat_model = ChatDeepSeek(model="deepseek-chat", temperature=0.7)



def llm_node(state:State):
    response = chat_model.invoke(state["messages"])
    return {"messages": response}

def action_node(state:State):
    system_prompt = """翻译为法语"""
    messages = state["messages"][-1]
    messages = [SystemMessage(content=system_prompt)] + [HumanMessage(content=messages.content)]

    response = chat_model.invoke(messages)
    return {"messages": response}



async def call1():
    builder = StateGraph(State)

    builder.add_node("llm_node", llm_node)
    builder.add_node("action_node", action_node)

    builder.add_edge(START, "llm_node")
    builder.add_edge("llm_node", "action_node")
    builder.add_edge("action_node", END)
    config = {"configurable":{"thread_id":1}}
    memory_server = MemorySaver()
    graph = builder.compile(checkpointer=memory_server)
    for chunk in graph.stream({"messages": ["你好，我叫木雨"]},config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()

    for chunk in graph.stream({"messages": ["我叫什么名字"]},config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()

async def call2():
    tools = [get_weather]
    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        graph = create_react_agent(chat_model,tools=tools,checkpointer=checkpointer)
        config = {"configurable": {"thread_id": 1}}
        for chunk in graph.stream({"messages": ["你好，我叫木雨"]}, config, stream_mode="values"):
            chunk["messages"][-1].pretty_print()

        for chunk in graph.stream({"messages": ["我叫什么名字"]}, config, stream_mode="values"):
            chunk["messages"][-1].pretty_print()

def call_model(state:MessagesState,config:RunnableConfig,*,store:BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("memories",user_id)
    memories = store.search(namespace)
    info = "\n".join([d.value["data"] for d in memories])
    last_message = state["messages"][-1]
    store.put(namespace,str(uuid.uuid4()),{"data":last_message.content})

    system_msg = f"使用如下上下文回答用户问题，上下文{info}"
    response = chat_model.invoke(
        [{"type":"system","content":system_msg}] +state["messages"]
    )
    store.put(namespace,str(uuid.uuid4()),{"data":response.content})
    return {"messages": response}

async def call3():
    #存储业务数据 保存用户消息和模型输出（记忆）
    in_memory_store = InMemoryStore()
    #存“执行状态” 保存 Graph 执行状态（节点输入输出、运行日志、断点恢复）
    memory = MemorySaver()
    builder = StateGraph(State)
    builder.add_node("llm_node", call_model)
    builder.add_edge(START, "llm_node")
    builder.add_edge("llm_node", END)

    graph = builder.compile(checkpointer=memory, store=in_memory_store)
    config = {"configurable":{"thread_id": "1"},"user_id":"1"}
    async for chunk in graph.astream({"messages": ["你好，我叫木雨"]}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()

    config = {"configurable":{"thread_id": "2"},"user_id":"1"}
    async for chunk in graph.astream({"messages": ["我叫什么名字"]}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()

if __name__ == '__main__':
    asyncio.run(call3())
