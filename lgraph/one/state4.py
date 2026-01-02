import operator
import os
from typing import TypedDict, List, Annotated

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph

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

def convert_message(state):
    PROMPT="""
        You are a data extraction specialist tasked with retrieving key information from a text.
        Extract such information for the provided text and output it in JSON format. Outline the key data points extracted
    """
    print(state)
    print("------------")
    messages = state["messages"]
    messages = messages[-1]
    messages = [
        SystemMessage(content=PROMPT),
        HumanMessage(content=messages.content),
    ]
    resp = llm.invoke(messages)
    return {'messages': [resp]}
def call():

    builder = StateGraph(State)
    builder.add_node("chat", chat_with_model)
    builder.add_node("convert", convert_message)

    builder.set_entry_point("chat")
    builder.add_edge("chat", "convert")
    builder.add_edge("convert", END)

    graph = builder.compile()
    query = "请你好好介绍自己"
    initial_state = {"messages": [HumanMessage(content=query)]}
    resp = graph.invoke(initial_state)
    print(resp)


if __name__ == '__main__':
    call()
