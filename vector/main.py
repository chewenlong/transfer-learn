#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/31 18:11
@Author  : chewl1
@File    : main.py
"""
from pymilvus import MilvusClient

if __name__ == '__main__':
    client = MilvusClient("milvus_demo.db")

    if client.has_collection(collection_name="demo_collection"):
        client.drop_collection(collection_name="demo_collection")
    client.create_collection(
        collection_name="demo_collection",
        dimension=768,  # The vectors we will use in this demo has 768 dimensions
    )
