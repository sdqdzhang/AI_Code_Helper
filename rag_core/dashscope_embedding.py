import os
from typing import List
from openai import OpenAI
from langchain_core.embeddings import Embeddings as BaseEmbeddings

# 定义阿里云 DashScope 兼容模式的 Base URL
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# DashScope 官方限制 BATCH_SIZE 最大为 10
BATCH_SIZE = 10

class CustomDashScopeEmbeddings(BaseEmbeddings):
    """
    自定义的 Embedding 包装器，用于使用原生的 openai.OpenAI 客户端
    调用阿里云 DashScope 的兼容 API。
    """

    def __init__(self, model: str, api_key: str, base_url: str = DASHSCOPE_BASE_URL):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        实现 LangChain 要求的文档 Embedding 方法，并强制进行分批处理。
        """
        all_embeddings: List[List[float]] = []

        # 在应用运行时，这个方法通常只用于 build_index.py，但在 db_manager
        # 中用于检索（单查询）时也会间接调用，所以我们保持其稳健性。

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"❌ DashScope Embedding API 调用失败。错误信息: {e}")
                # 重新抛出异常，让上层捕获处理
                raise e

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        实现 LangChain 要求的查询 Embedding 方法 (db_manager 依赖此方法)。
        """
        # 查询通常只有一条，不需要分批
        return self.embed_documents([text])[0]