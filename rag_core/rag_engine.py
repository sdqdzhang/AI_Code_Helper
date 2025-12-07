import os
import asyncio
from dotenv import load_dotenv
# 修复导入：使用新的 ChatOllama 路径 (尽管它在新版本中也被弃用，但目前可用)
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import Runnable

# 导入配置和数据库管理器
from rag_core.db_manager import DBManager


class RAGEngine:
    """
    核心推理引擎类：接收用户查询，检索上下文，并调用 Ollama LLM 异步生成答案。
    """

    # RAG 提示词模板
    RAG_PROMPT = """
    您是一位专业的 Pandas 编程助手。您的任务是使用提供的上下文（代码片段和文档）来回答用户的问题。
    请以清晰、简洁、专业的中文回答。
    如果上下文不足以回答问题，请说明您找不到相关信息。
    请务必在回答中尽可能提供代码示例。

    --- 上下文 ---
    {context}
    ---
    用户问题: {question}
    """

    def __init__(self, db_manager: DBManager):
        # 1. 加载环境变量
        load_dotenv()

        self.db_manager = db_manager
        self.llm_model_name = None
        self.llm_base_url = None
        self.k_value = None
        self.chain: Runnable = None

    def configure(self, llm_model_name: str, llm_base_url: str, k_value: int):
        """
        根据设置页面动态配置 LLM 和检索 K 值。
        """
        # 优化：如果配置未变，无需重新初始化
        if (self.llm_model_name == llm_model_name and
                self.llm_base_url == llm_base_url and
                self.k_value == k_value and
                self.chain is not None):
            return

        self.llm_model_name = llm_model_name
        self.llm_base_url = llm_base_url
        self.k_value = k_value

        # 2. 重新初始化 Ollama 客户端
        try:
            # 使用 ChatOllama 客户端连接本地 LLM
            self.llm = ChatOllama(
                model=self.llm_model_name,
                base_url=self.llm_base_url,
                temperature=0.0
            )
            self.prompt = ChatPromptTemplate.from_template(self.RAG_PROMPT)
            # 创建 Runnable Chain
            self.chain = self.prompt | self.llm
            print(f"✅ RAGEngine: 配置更新成功。LLM: {self.llm_model_name}, K: {self.k_value}")
        except Exception as e:
            print(f"❌ RAGEngine: Ollama LLM 配置或连接失败。错误: {e}")
            self.llm = None
            self.chain = None
            raise RuntimeError("LLM 配置失败，请检查 Ollama 服务。")

    def _format_context(self, documents: list[Document]) -> str:
        """将检索到的文档格式化为 LLM 可用的字符串。"""
        formatted_context = []
        for doc in documents:
            source = doc.metadata.get('source', '未知来源')
            api_name = doc.metadata.get('api_name', '通用')

            # 使用 Markdown 格式化，强调来源
            formatted_context.append(
                f"### 来源: {source} (API: {api_name})\n"
                f"```text\n{doc.page_content}\n```"
            )
        return "\n\n---\n\n".join(formatted_context)

    async def generate_answer(self, query: str) -> str:
        """
        执行 RAG 流程，异步生成答案。
        """
        if not self.chain:
            return "RAG 引擎未配置或 LLM 连接失败，请检查设置和 Ollama 服务状态。"

        try:
            # 1. 同步检索上下文 (DBManager 内部没有异步 API)
            retrieved_docs = self.db_manager.retrieve_documents(query, k=self.k_value)
            context = self._format_context(retrieved_docs)

            if not context:
                context = "未找到任何相关的知识文档。"

            # 2. 异步调用 LLM Chain
            # 这里的 ainvoke 是关键，它允许在 QThreadPool 线程中运行 asyncio 循环等待
            response = await self.chain.ainvoke({
                "context": context,
                "question": query
            })

            return response.content

        except Exception as e:
            print(f"❌ 异步 LLM 调用失败。错误: {e}")
            return f"❌ LLM 生成答案失败。请检查 Ollama 服务连接和模型状态。错误: {e}"