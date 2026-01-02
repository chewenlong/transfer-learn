import os

from langchain_classic.chains.llm import LLMChain
from langchain_classic.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationEntityMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def call():

    momery = ConversationBufferMemory(return_messages=True)
    momery.save_context({"input":"你好，请介绍一下你自己"},{"output":"我是一个全能小助手"})
    print(momery.load_memory_variables({}))

def call2():
    DEFAULT_TEMPLATE="""
        以下是人类与AI之间友好的对话，AI表现的很健谈，并提供了大量当前对话：
        {history}
        Human：{input}
        AI:
    """
    prompt = PromptTemplate(input_variables=["history", "input"],template=DEFAULT_TEMPLATE)
    deep_seek = ChatOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DPURL"),
        model="deepseek-chat",
    )

    conversaton = LLMChain(
        llm = deep_seek,
        prompt = prompt,
        memory = ConversationBufferMemory(memory_key="history"),
    )
    resp1 = conversaton.invoke(input="你好，请你介绍一下你自己")
    print(resp1)
    resp2 = conversaton.invoke(input="我是木羽，正在学习大模型开发知识")
    print(resp2)
    resp3 = conversaton.invoke(input="木羽在干什么")
    print(resp3)

def call3():
    memory = ConversationBufferWindowMemory(k=1,return_messages=True)
    memory.save_context({"input":"你好"},{"output":"我在请问有什么事情"})
    memory.save_context({"input":"初次对话，你能介绍下你自己吗"},{"output":"当然可以。我是一个无所不能的AI小助手"})

    DEFAULT_TEMPLATE="""
        以下是人类与AI之间友好的对话，AI表现的很健谈，并提供了大量当前对话：
        {history}
        Human：{input}
        AI:
    """
    prompt = PromptTemplate(input_variables=["history", "input"],template=DEFAULT_TEMPLATE)
    deep_seek = ChatOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DPURL"),
        model="deepseek-chat",
    )

    conversaton = LLMChain(
        llm = deep_seek,
        prompt = prompt,
        memory = ConversationBufferWindowMemory(memory_key="history",k=1),
    )
    resp1 = conversaton.invoke(input="你好，请你介绍一下你自己")
    print(resp1)
    resp2 = conversaton.invoke(input="我是木羽，正在学习大模型开发知识")
    print(resp2)
    resp3 = conversaton.invoke(input="木羽在干什么")
    print(resp3)
    # print(memory.load_memory_variables({}))

def call4():
    deep_seek = ChatOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DPURL"),
        model="deepseek-chat",
    )

    memory = ConversationEntityMemory(llm=deep_seek,return_messages=True)
    _input = {"input":"steven和Sam正在做一个黑客马拉松项目"}
    memory.load_memory_variables(_input)
    memory.save_context(_input,
                        {"output":"听起来是个很棒的项目，他们在做什么样的项目"}
                        )
    # resp = memory.load_memory_variables({"input":"谁是Sam"})
    # print(resp)
    template = """
        你是一个服务于人类的助手，由OpenAI训练的大型语言模型驱动。
        你被设计用于协助完成各种任务，从回答简单问题到提供广泛主题的深入解释与讨论。作为语言模型，你能够根据接收到的输入生成类人文本，从而进行自然流畅的对话，并提供连贯且切题的回应。
        你持续学习进步，能力不断进化。可以处理和理解海量文本，并运用这些知识为各种问题提供准确且信息丰富的解答。你可以获取人类在下文"背景信息"板块提供的个性化信息，同时能基于接收的输入自主生成文本，从而参与多元话题的讨论并提供解释说明。
        总体而言，你是一个功能强大的工具，能协助完成多样化任务，并就各类主题提供有价值的见解和信息。无论人类需要具体问题的帮助，还是想就某个话题展开对话，你都能随时提供支持。
        上下文：
        {entities}
        
        当前对话：
        {history}
        最后一行：
        人类：{input}
        你：
    """
    prompt_template = PromptTemplate(
        input_variables=["entities","history","input"],
        template=template,
    )

    conversation_with_entity = LLMChain(
        llm = deep_seek,
        prompt = prompt_template,
        memory = ConversationEntityMemory(llm=deep_seek),
        verbose=True,
    )
    result1 = conversation_with_entity.invoke(input="steven和Sam正在做一个黑客马拉松项目")
    print(result1)  # 因为 LLMChain 返回的字典中，"text" 键对应模型的输出
    #
    result2 = conversation_with_entity.invoke(input="他们正试图为LangChain添加更复杂的记忆结构")
    print(result2)
    result3 = conversation_with_entity.invoke(input="你对 Deven 和 Sam了解多少")
    print(result3)
    #
    # response =conversation_with_entity.invoke(input="Sam在做什么")
    # print('response',response)

if __name__ == '__main__':
    call4()