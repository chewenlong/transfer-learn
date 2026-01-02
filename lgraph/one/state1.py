from langgraph.constants import START, END
from langgraph.graph import StateGraph


def add(state):
    print(state)
    return {"x": state['x'] + 1}


def subtraction(state):
    print(state)
    return {"y": state['x'] - 2}


def call():
    builder = StateGraph(dict)
    builder.add_node("add", add)
    builder.add_node("sub", subtraction)

    builder.add_edge(START, "add")
    builder.add_edge("add", "sub")
    builder.add_edge("sub", END)

    graph = builder.compile()
    # 可视化图
    # image_data = graph.get_graph(xray=True).draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(image_data)
    # print("Graph saved as graph.png")
    initial_state = {"x": 10}
    resp = graph.invoke(initial_state)
    print(resp)

if __name__ == '__main__':
    call()
