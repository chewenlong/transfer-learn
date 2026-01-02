import operator
from typing import Optional, Union, TypedDict, Annotated

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field


# 结构化输出

class UserInfo(BaseModel):
    """Extracted user information, such as name, age, email, and phone number, if relevant."""
    name: str = Field(description="The name of the user")
    age: Optional[int] = Field(description="The age of the user")
    email: str = Field(description="The email of the user")
    phone: Optional[str] = Field(description="The phone number of the user")


class ConversationalResponse(BaseModel):
    """Respond to the user's query in a conversational manner. Be kind and helpful."""
    response: str = Field(description="A conversational response to the user's query")


class FinalResponse(BaseModel):
    final_output: Union[ConversationalResponse, UserInfo]


chat_model = ChatDeepSeek(model="deepseek-chat", temperature=0.7)


def chat_with_model(state):
    print(state)
    print("==============")
    messages = state["messages"]
    struct_llm = chat_model.with_structured_output(FinalResponse)
    resp = struct_llm.invoke(messages)
    return {"messages": [resp]}


def final_answer(state):
    print(state)
    messages = state["messages"][-1]
    resp = messages.final_output.response
    return {"messages": [resp]}


def insertdb(state):
    print("insertdb：", state)
    print("insert db")


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


def generate_branch(state: AgentState):
    messages = state["messages"][-1]
    output = messages.final_output
    if isinstance(output, UserInfo):
        return True
    elif isinstance(output, ConversationalResponse):
        return False


def call():
    graph = StateGraph(AgentState)
    graph.add_node("chat_with_model", chat_with_model)
    graph.add_node("final_answer", final_answer)
    graph.add_node("insertdb", insertdb)

    graph.set_entry_point("chat_with_model")

    graph.add_conditional_edges("chat_with_model", generate_branch,
                                {True: "insertdb", False: "final_answer"})
    graph.set_finish_point("final_answer")
    graph.set_finish_point("insertdb")
    graph = graph.compile()

    image_data = graph.get_graph(xray=True).draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(image_data)
    print("Graph saved as graph.png")

    # input_query = "我叫che，今年28岁,邮箱che@qq.com，电话13000000000"
    input_query = "请介绍下你自己"
    input_msgs = {"messages": [HumanMessage(content=input_query)]}
    result = graph.invoke(input_msgs)

    print(result)


if __name__ == '__main__':
    call()
