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


def call():
    builder = StateGraph(OverallState, input=InputState, output=OutputState)

    def agent_node(state: InputState):
        print("我是一个AI Agent")
        return {"question": state["question"]}

    def action_node(state: InputState):
        print("我现在是个执行者")
        step = state["question"]
        return {"answer": f"我现在执行成功了，我接收到的问题是{step}"}

    builder.add_node("agent_node", agent_node)
    builder.add_node("action_node", action_node)

    builder.add_edge(START, "agent_node")
    builder.add_edge("agent_node", "action_node")
    builder.add_edge("action_node", END)

    graph = builder.compile()
    final_answer = graph.invoke({"question": "你好"})
    print(final_answer)


if __name__ == '__main__':
    call()
