import operator
import os

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import ToolMessage, BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from openai import OpenAI
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel
from sqlalchemy.orm import Session

#执行顺序
# db_manager
#  → tool(query_sales)
#  → db_manager
#  → code_generator
#  → tool(python_repl)
#  → code_generator
#  → END
# 判断“下一步是谁”的速查表
# | 当前节点      | 最后一条消息                | 下一步            |
# | --------- | --------------------- | -------------- |
# | 任意 Agent  | 有 tool_calls          | call_tool      |
# | call_tool | sender=db_manager     | db_manager     |
# | call_tool | sender=code_generator | code_generator |
# | Agent     | 含 FINAL ANSWER        | END            |
# | Agent     | 普通消息                  | continue 分支    |


DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')

repl = PythonREPL()

db_llm  = ChatTongyi(
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


from typing import Annotated


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
    return result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."


def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful AI assistant, collaborating with other assistants. "
                    "Use the provided tools to progress towards answering the question. "
                    "If you are unable to fully answer, that's OK, another assistant with different tools "
                    "will help where you left off. Execute what you can to make progress. "
                    "If you or any of the other assistants have the final answer or deliverable, "
                    "prefix your response with FINAL ANSWER so the team knows to stop. "
                    "You have access to the following tools: {tool_names}.\n{system_message}"
                ),
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)

# 创建数据库 Agent
db_agent = create_agent(
    db_llm,
    [add_sale, delete_sale, update_sale, query_sales],
    system_message="You should provide accurate data for the code_generator to use, and source code shouldn't be the final answer.",
)

# 创建程序员 Agent
code_agent = create_agent(
    coder_llm,
    [python_repl],
    system_message="Run python code to display diagrams or output execution results.",
)

import functools

def agent_node(state, agent, name):
    """Invoke an agent and wrap the result into a standardized message format."""
    result = agent.invoke(state)

    # 将代理输出转换为适合附加到全局状态的格式
    if isinstance(result, ToolMessage):
        # 如果是 ToolMessage，直接保留
        pass
    else:
        # 创建一个 AIMessage 类的新实例
        # 包含 result 对象的所有数据（除了 type 和 name）
        # 并设置新实例的 name 属性为特定的值 name
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)

    return {
        "messages": [result],
        # 跟踪发件人，这样我们就知道下一个要传给谁
        "sender": name,
    }

# 使用 functools.partial 固定 agent 和 name 参数，方便调用
db_node = functools.partial(agent_node, agent=db_agent, name="db_manager")
code_node = functools.partial(agent_node, agent=code_agent, name="code_generator")

from typing import Literal

def router(state):
    # 这是一个路由
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # 前一个代理正在调用一个工具
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # 任何Agent都决定工作完成
        return END
    return "continue"

from typing import Annotated, Sequence
from typing_extensions import TypedDict

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


def call():
    tools = [add_sale, delete_sale, update_sale, query_sales, python_repl]
    tool_executor = ToolNode(tools)

    work_flow = StateGraph(AgentState)
    work_flow.add_node("db_manager",db_node)
    work_flow.add_node("code_generator",code_node)
    work_flow.add_node("call_tool",tool_executor)

    work_flow.add_conditional_edges("db_manager", router,
                                    {"continue":"code_generator","call_tool":"call_tool",END:END}
                                    )

    work_flow.add_conditional_edges("code_generator", router,
                                    {"continue":"db_manager","call_tool":"call_tool",END:END}
                                    )

    work_flow.add_conditional_edges("call_tool", lambda x:x["sender"],
                                    {"db_manager":"db_manager","code_generator":"code_generator"}
                                    )

    work_flow.set_entry_point("db_manager")
    graph = work_flow.compile()

    for chunk in graph.stream(
            {"messages": [HumanMessage(content="根据sales_id使用折线图显示前5名销售的销售总额")]},
            {"recursion_limit": 10},
            stream_mode='values'
    ):
        print(chunk)


    # image_data = graph.get_graph(xray=True).draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(image_data)
    # print("Graph saved as graph.png")
if __name__ == '__main__':
    call()