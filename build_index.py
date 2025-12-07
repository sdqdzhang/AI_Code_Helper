import os
import sys
from dotenv import load_dotenv
from typing import List
from tqdm import tqdm  # 导入 tqdm 库

# 核心依赖：LangChain的向量存储和Embedding基类
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings as BaseEmbeddings

# 严格按照用户要求，使用原生的 openai 客户端和 tiktoken
from openai import OpenAI
import tiktoken

# 导入自定义模块。
from rag_core.config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME
from rag_core.indexing_utils import get_processed_chunks

# 定义阿里云 DashScope 兼容模式的 Base URL
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 关键修复：定义分批处理大小，根据 DashScope 错误信息，最大值不能超过 10
BATCH_SIZE = 10


class CustomDashScopeEmbeddings(BaseEmbeddings):
    """
    自定义的 Embedding 包装器，用于使用原生的 openai.OpenAI 客户端
    调用阿里云 DashScope 的兼容 API，同时满足 LangChain Embeddings 接口要求。
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        实现 LangChain 要求的文档 Embedding 方法，并强制进行分批处理。
        """
        all_embeddings: List[List[float]] = []

        # 计算总批次数量
        num_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

        # --- 关键修复：手动分批处理 (Batching)，并使用 tqdm 显示进度 ---
        for i in tqdm(range(0, len(texts), BATCH_SIZE),
                      total=num_batches,
                      desc="向量化批次进度"):
            batch = texts[i:i + BATCH_SIZE]

            # 调用原生的 client.embeddings.create API
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                # 从响应中提取 embedding 向量并添加到总列表
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                # 如果某一批次失败，打印错误，并重新抛出以终止流程
                tqdm.write(f"❌ 警告：Embedding 过程中，批次 {i // BATCH_SIZE} 失败。错误信息: {e}")
                raise e

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        实现 LangChain 要求的查询 Embedding 方法。
        """
        # 查询通常只有一条，不需要分批
        return self.embed_documents([text])[0]


def build_index():
    """
    执行 RAG 索引构建的主流程：
    1. 加载和处理文档。
    2. 估算 Token 成本并请求用户确认。
    3. 初始化 Embedding 模型（使用自定义 Wrapper）。
    4. 创建 Chroma 向量数据库并存储数据。
    """
    print("=" * 60)
    print("🚀 悬浮 RAG 编程助手 - 知识索引构建工具")
    print("=" * 60)

    # 1. 加载环境变量 (确保 DASHSCOPE_API_KEY 已在 .env 文件中配置)
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        print("❌ 错误：DASHSCOPE_API_KEY 未在 .env 文件中配置！请检查您的密钥。")
        return

    # 2. 获取处理后的文档块
    chunks = get_processed_chunks()
    if not chunks:
        print("索引构建中止。")
        return

    # 3. Token 估算和确认
    print("-> 正在估算 Token 数量...")
    try:
        # 使用 tiktoken 估算 Token
        tokenizer = tiktoken.get_encoding("cl100k_base")

        all_text = [chunk.page_content for chunk in chunks]
        token_counts = [len(tokenizer.encode(text)) for text in all_text]
        total_tokens = sum(token_counts)

        print("-" * 50)
        print(f"📝 索引任务总结:")
        print(f"   总知识块数量: {len(chunks)} 个")
        print(f"   预计总 Token 数量 (用于 Embedding): {total_tokens:,} Tokens")
        print(f"   使用的模型: {EMBEDDING_MODEL_NAME}")
        print("-" * 50)

        # 请求用户确认
        user_input = input("❓ 确认开始向量化 (这会产生 API 费用)？(输入 'Y' 或 'N'): ").strip().upper()

        if user_input != 'Y':
            print("🛑 用户取消了索引构建。")
            return

    except ImportError:
        print("⚠️ 警告：未安装 tiktoken 库，无法估算 Token。继续构建...")
    except Exception as e:
        print(f"❌ 错误：Token 估算或用户确认失败。错误信息: {e}")
        return

    # 4. 初始化 Embedding 模型
    try:
        print(f"-> 正在初始化 OpenAI 兼容 Embedding 模型 (DashScope): {EMBEDDING_MODEL_NAME}...")

        embeddings = CustomDashScopeEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            api_key=api_key,
            base_url=DASHSCOPE_BASE_URL
        )
        print("-> Embedding 模型初始化成功。")
    except Exception as e:
        print(f"❌ 错误：Embedding 模型初始化失败。请检查 API Key 和依赖库。错误信息: {e}")
        return

    # 5. 创建 Chroma 向量数据库并存储
    # 注意：Chroma.from_documents 在内部调用了 CustomDashScopeEmbeddings.embed_documents，
    # 进度条已经在 embed_documents 中实现，无需在此处重复包装。
    print(f"-> 正在创建和填充 Chroma 数据库到: {CHROMA_DB_PATH}...")
    try:
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME
        )

        print("✅ 索引构建成功！")
        print(f"共计 {len(chunks)} 个知识块已存储到 ChromaDB 的 '{COLLECTION_NAME}' 集合中。")
    except Exception as e:
        print(f"❌ 错误：存储到 ChromaDB 失败。错误信息: {e}")
        # 打印原始错误信息，帮助调试
        print(f"原始错误详情: {e}")


if __name__ == "__main__":
    build_index()