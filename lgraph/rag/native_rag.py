import os

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')

# 1. 初始化通义千问 Embedding 模型
embeddings_model = DashScopeEmbeddings(
    model="text-embedding-v3",  # 指定模型版本
    dashscope_api_key=DASHSCOPE_API_KEY
)

llm  = ChatTongyi(
    dashscope_api_key=DASHSCOPE_API_KEY,  # 如果未设置环境变量，在此处直接填写
    model="qwen3-max",  # 指定模型，例如 qwen-max, qwen-plus, qwen-vl-plus等:cite[1]
    temperature=0.5,
    streaming=True  # 如需流式输出，可设置为True
)

class EmbeddingGenerator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = embeddings_model

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            response = self.client.embed_documents(text)
            embeddings.append(response.data[0])
        return embeddings

    def embed_query(self, query):
        # 使用相同的处理逻辑，只是这次只为单个查询处理
        response = self.client.embed_query(query)
        if hasattr(response, 'data') and response.data:
            return response.data[0].embedding
        return [0] * 1024  # 如果获取嵌入失败，返回零向量

def call():
    embeddings = []
    loader = WebBaseLoader(
        web_path="https://lilianweng.github.io/posts/2023-06-23-agent/",
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_ = ("post-content","post-title","post-header")
            )
        ),
    )

    docs = loader.load()
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    splits = text_spliter.split_documents(docs)

    texts = [content for document in splits for split_type, content in document if split_type == 'page_content']
    chroma_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings_model,  # 使用定义的嵌入生成器实例
        create_collection_if_not_exists=True
    )
    IDs = chroma_store.add_texts(texts=texts)

    retriever = chroma_store.as_retriever()

    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
             {"context": retriever | format_docs, "question": RunnablePassthrough()}
             | prompt
             | llm
             | StrOutputParser()
         )

    rag_res = rag_chain.invoke("What is Task Decomposition?")
    print(rag_res)
    # for doc_splits in splits:
    #     for split_type, split_content in doc_splits:
    #         if split_type == 'page_content' and split_content.strip():
    #             try:
    #                 response = embeddings_model.embed_documents(split_content)
    #                 embeddings.append(response[0])
    #             except Exception as e:
    #                 print(f"请求失败，错误信息：{e}")
    #
    # for i,embedding in enumerate(embeddings):
    #     print(f"{i+1}: {embedding[:3]}")
    # embedded_query = embeddings.embed_query("请问车文龙喜欢吃什么")









if __name__ == '__main__':
    call()