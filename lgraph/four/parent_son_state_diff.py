import asyncio
from typing import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.constants import START
from langgraph.graph import StateGraph

llm = ChatDeepSeek(model="deepseek-chat", temperature=0.7)

class ParentState(TypedDict):
    user_input:str
    final_answer:str

def parent_node1(state: ParentState):
    response = llm.invoke(state["user_input"])
    return {"final_answer":response.content}

class SubgraphState(TypedDict):
    response_answer:str
    summary_answer:str
    score: str

def subgraph_node1(state: SubgraphState):
    system_prompt = """
        请总结内容为50个字或更少的字数
    """
    messages = state["response_answer"]
    messages = [SystemMessage(content = system_prompt)] + [HumanMessage(content = messages)]
    response = llm.invoke(messages)
    return {"summary_answer":response}

def subgraph_node2(state: SubgraphState):
    messages = f""",
        This is the full content of what you received：{state["response_answer"]} ",
        This information is summarized for the full content:{state["summary_answer"]} ",
        Please rate the text and summary information, returning a scale of 1 to 10. Note: Only the score value needs to be returned.",
    """
    response = llm.invoke(messages)
    return {"score":response.content}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node("subgraph_node1",subgraph_node1)
subgraph_builder.add_node("subgraph_node2",subgraph_node2)
subgraph_builder.add_edge(START,"subgraph_node1")
subgraph_builder.add_edge("subgraph_node1", "subgraph_node2")
subgraph = subgraph_builder.compile()

def parent_node2(state:ParentState):
    response = subgraph.invoke({"response_answer":state["final_answer"]})
    return {"final_answer":response["score"]}

async def call():
    builder = StateGraph(ParentState)
    builder.add_node("node1",parent_node1)
    builder.add_node("node2",parent_node2)
    builder.add_edge(START, "node1")
    builder.add_edge("node1", "node2")
    graph = builder.compile()

    async for chunk in graph.astream({"user_input": "我现在想学习大模型，应该关注哪些技术"}, stream_mode="values"):
        print(chunk)







if __name__ == '__main__':
    asyncio.run(call())