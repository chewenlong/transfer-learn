import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma


def call():
    examples = [
        {"input": "内蒙古省会在哪里?", "output": "呼和浩特"},
        {"input": "世界最高的山峰是?", "output": "喜马拉雅山"},
        {"input": "世界最深的海沟是?", "output": "马里亚纳海沟"},
        {"input": "车文龙最喜欢什么食物?", "output": "车文龙最喜欢的食物是土豆炖牛肉"},
    ]

    # 拼接成文本列表
    to_vectorize = [" ".join(example.values()) for example in examples]

    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v3", dashscope_api_key=DASHSCOPE_API_KEY
    )

    # 初始化 Chroma 向量数据库
    vectorstore = Chroma.from_texts(
        texts=to_vectorize,
        embedding=embeddings,
        metadatas=examples,
        collection_name="my_example_collection",
    )

    print("=== 初始向量存储创建完成 ===")

    # -------- 查询(Query) --------
    query = "请问车文龙喜欢吃什么"
    result = vectorstore.similarity_search(query, k=1)
    print("查询结果:", result)

    # -------- 新增(Add) --------
    new_example = {"input": "地球上最大的沙漠是?", "output": "撒哈拉沙漠"}
    vectorstore.add_texts(
        texts=[" ".join(new_example.values())],
        metadatas=[new_example],
    )
    print("新增条目完成")

    # -------- 修改(Update) --------
    # Chroma 没有直接 update 方法，可以先删除再添加新的
    updated_example = {
        "input": "车文龙最喜欢什么食物?",
        "output": "车文龙最喜欢的食物是炸鸡",
    }
    # 删除原来的（这里依然用索引示例，实际项目建议也用 metadata 匹配）
    vectorstore.delete(ids=[vectorstore._collection.get()["ids"][3]])  # 假设它是第四条
    # 添加新的
    vectorstore.add_texts(
        texts=[" ".join(updated_example.values())],
        metadatas=[updated_example],
    )
    print("更新条目完成")

    # -------- 删除(Delete) --------
    delete_example = {"input": "世界最深的海沟是?", "output": "马里亚纳海沟"}
    # 根据 metadata 删除：先取出底层 collection 的 ids 和 metadatas，一一对应匹配
    collection_data = vectorstore._collection.get()
    all_ids = collection_data["ids"]
    all_metadatas = collection_data["metadatas"]

    ids_to_delete = [
        _id
        for _id, meta in zip(all_ids, all_metadatas)
        if meta == delete_example
    ]

    if ids_to_delete:
        vectorstore.delete(ids=ids_to_delete)
        print("删除条目完成, 删除的 ids:", ids_to_delete)
    else:
        print("未找到需要删除的条目")

    # 再查询一次看效果
    query2 = "世界最深的海沟是哪里？"
    result2 = vectorstore.similarity_search(query2, k=1)
    print("查询结果2:", result2)


if __name__ == "__main__":
    call()
