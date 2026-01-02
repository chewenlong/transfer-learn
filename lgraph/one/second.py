import os

from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from typing_extensions import TypedDict


class InputState(TypedDict):
    question: str


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
    return {"answer": response.content}

def call():
    builder = StateGraph(OverallState, input=InputState, output=OutputState)

    builder.add_node("agent_node", llm_node)

    builder.add_edge(START, "agent_node")
    builder.add_edge("agent_node", END)

    graph = builder.compile()
    final_answer = graph.invoke({"question": "你好"})
    print(final_answer)


if __name__ == '__main__':
    call()
