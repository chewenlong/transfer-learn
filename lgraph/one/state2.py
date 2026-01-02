from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph


class State(TypedDict):
    x: int
    y: int

def add(state) :
    print(state)
    # 返回完整的状态，只更新需要修改的字段
    return {"x": state['x'] + 1}

def subtraction(state) :
    print(state)
    return {"y": state['x']}


def call():
    builder = StateGraph(State)
    builder.add_node("add", add)
    builder.add_node("sub", subtraction)

    builder.add_edge(START, "add")
    builder.add_edge("add", "sub")
    builder.add_edge("sub", END)

    graph = builder.compile()
    initial_state = {"x": 10}
    resp = graph.invoke(initial_state)
    print(resp)

if __name__ == '__main__':
    call()
