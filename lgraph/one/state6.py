import os
from typing import TypedDict, List, Annotated

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, add_messages

llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DPURL"),
    model="deepseek-chat",
)


class State(TypedDict):
    messages: Annotated[List, add_messages]


def chat_with_model(state):
    print(state)
    messages = state["messages"]
    print("------------")
    response = llm.invoke(messages)
    return {"messages": [response]}


def call():
    print(os.getenv("LANGSMITH_TRACING"))
    print(os.getenv("LANGSMITH_API_KEY"))
    builder = StateGraph(State)
    builder.add_node("chatbot", chat_with_model)

    builder.set_entry_point("chatbot")
    builder.set_finish_point("chatbot")
    graph = builder.compile()

    resp = graph.invoke({"messages": [{"role": "user", "content": "你好请介绍下自己"}]})

    print(resp)


if __name__ == '__main__':
    call()
