import os
from typing import TypedDict, Literal

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.message import MessagesState

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')

llm = ChatTongyi(
    dashscope_api_key=DASHSCOPE_API_KEY,  # 如果未设置环境变量，在此处直接填写
    model="qwen3-max",  # 指定模型，例如 qwen-max, qwen-plus, qwen-vl-plus等:cite[1]
    temperature=0.5,
    streaming=True  # 如需流式输出，可设置为True
)

coder_llm = ChatTongyi(
    dashscope_api_key=DASHSCOPE_API_KEY,  # 如果未设置环境变量，在此处直接填写
    model="qwen3-coder-plus",  # 指定模型，例如 qwen-max, qwen-plus, qwen-vl-plus等:cite[1]
    temperature=0.5,
    streaming=True  # 如需流式输出，可设置为True
)


class AgentState(MessagesState):
    next: str


members = ["chat", "coder", "sqler"]
options = members + ["FINISH"]


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH"""
    next: Literal[*options]


def supervisor(state: AgentState):
    system_prompt = (
        f"You are a supervisor tasked with managing a conversation between the following workers: {members}.\n\n"
        "Each worker has a specific role:\n"
        "- chat: Responds directly to user inputs using natural language.\n"
        "- coder: Activated for tasks that require mathematical calculations or specific coding needs.\n"
        "- sqler: Used when database queries or explicit SQL generation is needed.\n\n"
        "Given the following user request, respond with the worker to act next. "
        "Each worker will perform a task and respond with their results and status. "
        "When finished, respond with FINISH."
    )

    messages = [
                   {"role": "system", "content": system_prompt},
               ] + state["messages"]

    response = llm.with_structured_output(Router).invoke(messages)

    next_ = response["next"]

    if next_ == "FINISH":
        next_ = END

    return {"next": next_}


def chat(state: AgentState):
    messages = state["messages"][-1]
    model_response = llm.invoke(messages.content)
    final_response = [HumanMessage(content=model_response.content, name="chat")]  # 添加名称
    return {"messages": final_response}


def coder(state: AgentState):
    messages = state["messages"][-1]
    model_response = llm.invoke(messages.content)
    final_response = [HumanMessage(content=model_response.content, name="coder")]  # 添加名称
    return {"messages": final_response}


def sqler(state: AgentState):
    messages = state["messages"][-1]
    model_response = llm.invoke(messages.content)
    final_response = [HumanMessage(content=model_response.content, name="sqler")]  # 添加名称
    return {"messages": final_response}


def call():
    builder = StateGraph(AgentState)
    builder.add_node("supervisor",supervisor)
    builder.add_node("chat",chat)
    builder.add_node("coder",coder)
    builder.add_node("sqler",sqler)

    for member in members:
        builder.add_edge(member,"supervisor")

    builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
    {
        "chat": "chat",
        "coder": "coder",
        "sqler": "sqler",
        END: END,
    }
    )
    builder.add_edge(START,"supervisor")
    graph = builder.compile()

    for chunk in graph.stream(
            {"messages": "你好请介绍下你自己"},
            stream_mode='values'
    ):
        print(chunk)

    image_data = graph.get_graph(xray=True).draw_mermaid_png()
    with open("supervisor_agent_graph.png", "wb") as f:
        f.write(image_data)
    print("Graph saved as supervisor_agent_graph.png")

if __name__ == '__main__':
    call()
