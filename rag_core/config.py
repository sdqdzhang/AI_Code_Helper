import os
import os.path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# --- 路径配置 ---
# 存放 Pandas .rst.txt 源文档的根目录。
DOCS_PATH = os.path.join(os.path.dirname(__file__), '..', 'pandas_docs')

# 本地 ChromaDB 存储路径
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), 'chroma_db')

# ChromaDB Collection 名称
COLLECTION_NAME = "pandas_api_reference"

# PyQt Settings 文件名称 (用于持久化保存设置)
SETTINGS_FILE = "panda_rag_helper_settings"

# --- 模型配置 (Embedding 和 LLM) ---
# 阿里云 DashScope Embedding 模型 (需要配置 DASHSCOPE_API_KEY)
EMBEDDING_MODEL_NAME = "text-embedding-v3"

# LLM 模型配置 (默认值，实际运行时从 QSettings 加载)
DEFAULT_LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3.1")
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")

# --- 分块策略配置 (Chunking) ---
RST_HEADERS_TO_SPLIT_BY = [
    (r"\n(={3,}|-{3,}|~{3,})\n", "Header"),
    (r"\n.. code-block:: python\n", "CodeBlock"),
]
CHUNK_SIZE = 1000  # 字符数
CHUNK_OVERLAP = 200 # 字符重叠

# --- RAG 检索配置 (Retrieval) ---
# K 值：默认值 (实际运行时从 QSettings 加载)
DEFAULT_RETRIEVAL_K = 3

# --- UI 配置 ---
DEFAULT_THEME = "Light" # Light or Dark