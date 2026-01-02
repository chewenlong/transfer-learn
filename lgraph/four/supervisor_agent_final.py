
import operator
import os
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import ToolMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.constants import END, START
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from openai import OpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_experimental.utilities import PythonREPL
from pydantic import BaseModel
from sqlalchemy.orm import Session

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')

repl = PythonREPL()

llm  = ChatTongyi(
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

# 数据模型定义
class AddSaleSchema(BaseModel):
    product_id: int
    employee_id: int
    customer_id: int
    sale_date: str
    quantity: int
    amount: float
    discount: float


class DeleteSaleSchema(BaseModel):
    sales_id: int


class UpdateSaleSchema(BaseModel):
    sales_id: int
    quantity: int
    amount: float


class QuerySalesSchema(BaseModel):
    count: int


# 1. 添加销售数据
@tool(args_schema=AddSaleSchema)
def add_sale(product_id, employee_id, customer_id, sale_date, quantity, amount, discount):
    """Add sale record to the database."""
    print("add sale:", product_id, employee_id, customer_id, sale_date, quantity, amount, discount)


# 2. 删除销售数据
@tool(args_schema=DeleteSaleSchema)
def delete_sale(sales_id):
    """Delete sale record from the database."""
    print("delete sale:", sales_id)


# 3. 修改销售数据
@tool(args_schema=UpdateSaleSchema)
def update_sale(sales_id, quantity, amount):
    """Update sale record in the database."""
    print("update sale:", sales_id, quantity, amount)


# 4. 查询销售数据
@tool(args_schema=QuerySalesSchema)
def query_sales(count):
    """Retrieve specified data in reverse chronological order based on quantity"""
    print("query_sale:")
    return [
        {"sales_id": 5, "quantity": 5, "amount": 5},
        {"sales_id": 4, "quantity": 4, "amount": 4},
        {"sales_id": 3, "quantity": 3, "amount": 3},
        {"sales_id": 2, "quantity": 2, "amount": 2},
        {"sales_id":1,"quantity":1,"amount":1},
    ]


from typing import Annotated, TypedDict, Literal


@tool
def python_repl(
        code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code.
    If you want to see the output of a value,
    you should print it out with `print(...)`.
    This is visible to the user.
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"

    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str

# 创建数据库 Agent
db_agent = create_react_agent(
    llm,
    tools = [add_sale, delete_sale, update_sale, query_sales],
    prompt=SystemMessage(content="You should provide accurate data for the code_generator to use.")
)

# 创建程序员 Agent
code_agent = create_react_agent(
    coder_llm,
    [python_repl],
    prompt=SystemMessage(content="Run python code to display diagrams or output execution results.")
)

class AgentState(MessagesState):
    next: "str"

def db_node(state: AgentState):
    result = db_agent.invoke(state)
    return {
        "messages": [
            HumanMessage(content=result["messages"][-1].content, name="sqler")
        ]
    }

def code_node(state: AgentState):
    result = code_agent.invoke(state)
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name="coder")]
    }

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

def call():
    builder = StateGraph(AgentState)
    builder.add_node("supervisor",supervisor)
    builder.add_node("chat",chat)
    builder.add_node("coder",code_node)
    builder.add_node("sqler",db_node)

    for member in members:
        builder.add_edge(member,"supervisor")

    builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "chat": "chat",
            "coder": "coder",
            "sqler": "sqler",
            "FINISH": END,
            END: END,
        },
    )

    builder.add_edge(START,"supervisor")
    graph = builder.compile()

    for chunk in graph.stream(
            {"messages": "根据前5名的销售记录id，生成对应的销售额柱状图"},
            stream_mode='values'
    ):
        print(chunk)

    # image_data = graph.get_graph(xray=True).draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(image_data)
    # print("Graph saved as graph.png")

if __name__ == '__main__':
    call()