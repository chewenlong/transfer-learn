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
    if state["x"] == 10:
        return True
    else:
        return False

def call():
    builder = StateGraph(State)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_node("node_c", node_c)
    builder.set_entry_point("node_a")

    builder.add_conditional_edges("node_a", routing_func,{True:"node_b", False:"node_c"})
    builder.add_edge("node_b",END)
    builder.add_edge("node_c",END)
    graph = builder.compile()

    generate_image(graph)

    initial_state = {"x": 10}
    resp = graph.invoke(initial_state)
    print(resp)


def generate_image(graph):
    image_data = graph.get_graph(xray=True).draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(image_data)
    print("Graph saved as graph.png")


if __name__ == '__main__':
    call()
