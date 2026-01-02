import os
from typing import Optional

from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from typing_extensions import TypedDict


class InputState(TypedDict):
    question: str
    llm_answer: Optional[str]



class OutputState(TypedDict):
    answer: str

class OverallState(InputState, OutputState):
    pass

def llm_node(state:InputState):
    messages = [
        ("system","你是一位乐于助人的智能小助理"),
        ("human",state["question"])
    ]
    llm = ChatOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DPURL"),
        model="deepseek-chat",
    )
    response = llm.invoke(messages)
    return {"llm_answer": response.content}

def action_node(state:InputState):
    messages = [
        ("system","翻译为法语"),
        ("human",state["llm_answer"])
    ]
    llm = ChatOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DPURL"),
        model="deepseek-chat",
    )
    response = llm.invoke(messages)
    return {"answer": response.content}

def call():
    builder = StateGraph(OverallState, input=InputState, output=OutputState)

    builder.add_node("llm_node", llm_node)
    builder.add_node("action_node", action_node)

    builder.add_edge(START, "llm_node")
    builder.add_edge("llm_node", "action_node")
    builder.add_edge("action_node", END)

    graph = builder.compile()
    final_answer = graph.invoke({"question": "你好"})
    print(final_answer)


if __name__ == '__main__':
    call()
