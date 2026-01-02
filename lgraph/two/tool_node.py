import operator
from typing import Optional, Union, TypedDict, Annotated

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from pydantic.v1 import Field

from lgraph.tools.weather import get_weather, WeatherCode


class MultiplyParams(BaseModel):
    a: int = Field(description="One of the parameters for multiplication")
    b: int = Field(description="Another one of the parameters for multiplication")

@tool(args_schema=MultiplyParams)
def multiply(a: int, b: int) -> Optional[int]:
    """将2个数相乘"""
    return a * b

class UserInfo(BaseModel):
    """Extracted user information, such as name, age, email, and phone number, if relevant."""
    name: str = Field(description="The name of the user")
    age: Optional[int] = Field(description="The age of the user")
    email: str = Field(description="The email of the user")
    phone: Optional[str] = Field(description="The phone number of the user")

@tool(args_schema=UserInfo)
def insert_db(name,age,email,phone):
    """Insert user information into the database."""
    print("insert db：",name,age,email,phone)

class ConversationalResponse(BaseModel):
    """Respond to the user's query in a conversational manner. Be kind and helpful."""
    response: str = Field(description="A conversational response to the user's query")

class FinalResponse(BaseModel):
    final_output: Union[ConversationalResponse, UserInfo,MultiplyParams,WeatherCode]

def chat_with_model(state):
    print(state)
    print("==============")
    messages = state["messages"]
    struct_llm = chat_model.with_structured_output(FinalResponse)
    resp = struct_llm.invoke(messages)
    return {"messages": [resp]}

def final_answer(state):
    messages = state["messages"][-1]
    resp = messages.final_output.response
    return {"messages": [resp]}

chat_model = ChatDeepSeek(model="deepseek-chat", temperature=0.7)
tools = [get_weather, multiply, insert_db]
tool_node = ToolNode(tools)
model_with_tools = chat_model.bind_tools(tools)

def execute_function(state):
    print(state)
    messages = state["messages"][-1].final_output
    resp = tool_node.invoke({"messages": [model_with_tools.invoke(str(messages))]})
    print(f"response: {resp}")
    resp = resp["messages"][0].content
    return {"messages": [resp]}

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

def generate_branch(state: AgentState):
    messages = state["messages"][-1]
    output = messages.final_output
    if isinstance(output, ConversationalResponse):
        return False
    else:
        return True



def call():
    graph = StateGraph(AgentState)
    graph.add_node("chat_with_model", chat_with_model)
    graph.add_node("final_answer", final_answer)
    graph.add_node("execute_function", execute_function)

    graph.set_entry_point("chat_with_model")
    graph.add_conditional_edges("chat_with_model", generate_branch,
                                {True: "execute_function", False: "final_answer"})
    graph.set_finish_point("final_answer")
    graph.set_finish_point("execute_function")

    graph = graph.compile()
    input_query = "我叫che，今年28岁,邮箱che@qq.com，电话13000000000"
    input_msgs = {"messages": [HumanMessage(content=input_query)]}
    result = graph.invoke(input_msgs)
    print(result["messages"][-1])
    image_data = graph.get_graph(xray=True).draw_mermaid_png()
    with open("tool_graph.png", "wb") as f:
        f.write(image_data)
    print("Graph saved as tool_graph.png")

if __name__ == '__main__':
    call()