import os
from typing import Optional

from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
#结构化输出

class UserInfo(BaseModel):
    """Extracted user information, such as name, age, email, and phone number, if relevant."""
    name: str = Field(description="The name of the user")
    age: Optional[int] = Field(description="The age of the user")
    email: str = Field(description="The email of the user")
    phone: Optional[str] = Field(description="The phone number of the user")


chat_model = ChatDeepSeek(model="deepseek-chat", temperature=0.7)

struct_llm = chat_model.with_structured_output(UserInfo)
def call():
    resp = struct_llm.invoke("我叫che，今年28岁,邮箱che@qq.com，电话13000000000")
    print(resp)


if __name__ == '__main__':
    call()
