import operator
from typing import TypedDict, List, Annotated

from langgraph.constants import START, END
from langgraph.graph import StateGraph


class State(TypedDict):
    messages: Annotated[List[str], operator.add]


def add(state):
    print(state)
    msg = state['messages'][-1]
    resp = {"x": msg['x'] + 1}
    return {"messages": [resp]}


def subtraction(state):
    print(state)
    msg = state['messages'][-1]
    resp = {"x": msg['x'] - 2}
    return {"messages": [resp]}


def call():
    builder = StateGraph(State)
    builder.add_node("add", add)
    builder.add_node("sub", subtraction)

    builder.add_edge(START, "add")
    builder.add_edge("add", "sub")
    builder.add_edge("sub", END)

    graph = builder.compile()
    initial_state = {'messages':[{"x": 10}]}
    resp = graph.invoke(initial_state)
    print(resp)


if __name__ == '__main__':
    call()
