import operator
import os
from typing import TypedDict, List, Annotated

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessageGraph
from sympy.abc import lamda

llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DPURL"),
    model="deepseek-chat",
)
class State(TypedDict):
    messages: Annotated[List[str], operator.add]


def chat_with_model(state):
    print(state)
    messages = state["messages"]
    print("------------")
    response = llm.invoke(messages)
    return {"messages": [response]}


def call():

    builder = MessageGraph()
    builder.add_node("chatbot", lambda state:[("assistant","你好最帅气的人")])

    builder.set_entry_point("chatbot")
    builder.set_finish_point("chatbot")
    graph = builder.compile()

    resp = graph.invoke([("user","你好请介绍下自己")])
    print(resp)


if __name__ == '__main__':
    call()
