import os
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

# 导入配置和自定义 Embedding
from rag_core.config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME
from rag_core.dashscope_embedding import CustomDashScopeEmbeddings

class DBManager:
    """
    封装 ChromaDB 的加载和查询（检索）逻辑。
    负责将用户查询向量化，并从持久化的向量库中检索相关文档块。
    """

    def __init__(self):
        # 1. 加载环境变量
        load_dotenv()
        dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")

        if not dashscope_api_key:
            print("❌ 错误：DASHSCOPE_API_KEY 未设置。请检查 .env 文件。")
            self.db = None
            return

        # 2. 初始化 DashScope Embedding 模型 (用于查询向量化)
        try:
            print("-> 正在初始化 DashScope Embedding 模型...")
            self.embeddings = CustomDashScopeEmbeddings(
                model=EMBEDDING_MODEL_NAME,
                api_key=dashscope_api_key
            )
            print("✅ DBManager: DashScope Embedding 模型初始化成功。")
        except Exception as e:
            print(f"❌ DBManager: Embedding 模型初始化失败。错误: {e}")
            self.db = None
            return

        # 3. 加载 ChromaDB 数据库
        try:
            # LangChainDeprecationWarning: Chroma is deprecated. Use langchain-chroma package instead.
            self.db = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=self.embeddings,
                collection_name=COLLECTION_NAME
            )
            print(f"✅ DBManager: ChromaDB 从 {CHROMA_DB_PATH} 加载成功。")

        except Exception as e:
            print(f"❌ DBManager: ChromaDB 加载或初始化失败。请先运行 build_index.py。错误: {e}")
            self.db = None

    def retrieve_documents(self, query: str, k: int) -> List[Document]:
        """
        从向量数据库中检索最相关的文档块，k 值由调用者 (RAGEngine) 传入。
        """
        if not self.db:
            print("❌ 数据库未初始化，无法执行检索。")
            return []

        try:
            # 使用 similarity_search 替代 retriever.invoke，以支持动态 K
            documents = self.db.similarity_search(query, k=k)
            print(f"🔎 检索到 {len(documents)} 个相关文档块 (K={k})。")
            return documents
        except Exception as e:
            print(f"❌ 检索过程中发生错误。错误: {e}")
            return []