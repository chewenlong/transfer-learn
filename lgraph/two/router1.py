from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph


class State(TypedDict):
    x: int
    y: int

def node_a(state) :
    print(state)
    # 返回完整的状态，只更新需要修改的字段
    return {"x": state['x'] + 1}

def node_b(state) :
    print(state)
    return {"x": state['x']-2}

def node_c(state) :
    print(state)
    return {"x": state['x']-1}

def routing_func(state):
    if state["x"] == 11:
        return "node_b"
    else:
        return "node_c"

def call():
    builder = StateGraph(dict)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_node("node_c", node_c)
    builder.set_entry_point("node_a")

    builder.add_conditional_edges("node_a", routing_func)

    graph = builder.compile()

    # image_data = graph.get_graph(xray=True).draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(image_data)
    # print("Graph saved as graph.png")

    initial_state = {"x": 10}
    resp = graph.invoke(initial_state)
    print(resp)

if __name__ == '__main__':
    call()
