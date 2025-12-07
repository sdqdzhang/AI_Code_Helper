import os
from typing import List
from tqdm import tqdm  # 导入 tqdm 库

# 核心模块导入：LangChain 0.2.x 架构
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
# 最终修复：使用 LangChain 最新版本中最常用的导入路径。
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 从 config.py 导入配置
from rag_core.config import DOCS_PATH, RST_HEADERS_TO_SPLIT_BY, CHUNK_SIZE, CHUNK_OVERLAP


# --------------------
# 1. 定制加载器
# --------------------

def load_documents() -> List[Document]:
    """
    递归加载 DOCS_PATH 下所有 .rst.txt 文件，并附带原始元数据。
    """
    print(f"-> 正在从目录: {DOCS_PATH} 递归加载文档...")

    try:
        # 修复：通过 loader_kwargs 明确指定编码为 'utf-8'
        loader = DirectoryLoader(
            DOCS_PATH,
            glob="**/*.rst.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'},
            recursive=True
        )
        documents = loader.load()
        print(f"-> 文档加载完成，共计 {len(documents)} 个原始文档。")
        return documents
    except Exception as e:
        print(f"❌ 文档加载失败。请检查路径或文件权限。错误信息: {e}")
        return []


# --------------------
# 2. 元数据和分块函数
# --------------------

def _extract_metadata(doc: Document) -> dict:
    """
    根据文件路径和内容提取结构化元数据。
    """
    source_path = doc.metadata.get('source', '')

    metadata = {
        "source": os.path.basename(source_path),
        "doc_type": "GENERAL",
        "api_name": None
    }

    # 将路径标准化为斜杠，方便检查
    normalized_path = source_path.replace(os.sep, '/')

    if 'reference/api' in normalized_path:
        metadata["doc_type"] = "API_REFERENCE"
        # 尝试从文件名提取 API 名称 (e.g., pandas.DataFrame.agg.rst.txt)
        try:
            filename = os.path.basename(source_path)
            # 去除 .rst.txt 后缀
            api_name = filename.replace('.rst.txt', '')
            metadata["api_name"] = api_name
        except:
            pass
    elif 'getting_started' in normalized_path or 'user_guide' in normalized_path:
        metadata["doc_type"] = "TUTORIAL_GUIDE"
    elif 'development' in normalized_path:
        metadata["doc_type"] = "DEVELOPMENT_GUIDE"

    return metadata


def split_and_add_metadata(documents: List[Document]) -> List[Document]:
    """
    对文档进行结构化分块，并添加丰富的元数据。
    """
    all_chunks = []

    # 使用 RecursiveCharacterTextSplitter，关闭正则表达式模式

    # 分隔符：使用 ReST 标题的字符（作为纯字符串），加上通用的换行和空格。
    rst_separators = ["\n=======================================\n",
                      "\n---------------------------------------\n",
                      "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
                      "\n.. code-block:: python\n"]

    separators = rst_separators + ["\n\n", "\n", " "]

    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 默认 is_separator_regex=False，避免 list index out of range 错误
    )

    print("-> 正在进行结构化分块和元数据提取...")

    # 使用 tqdm 包装文档列表以显示进度
    for i, doc in enumerate(tqdm(documents, desc="分块处理进度")):
        # 提取基础元数据
        base_metadata = _extract_metadata(doc)

        # 1. 分割文档
        try:
            chunks = text_splitter.create_documents(
                texts=[doc.page_content],
                metadatas=[base_metadata]
            )

            # 检查分块结果是否有效
            if not chunks or len(chunks) == 0:
                # 使用 tqdm.write 避免进度条被干扰
                tqdm.write(f"⚠️ 警告：文件 '{base_metadata.get('source', 'Unknown File')}' 分块后为空，已跳过。")
                continue

        except Exception as e:
            # 如果分块失败，打印错误和文件源，跳过该文档
            tqdm.write(f"❌ 错误：文件 '{base_metadata.get('source', 'Unknown File')}' 分块失败，已跳过。错误: {e}")
            continue

        # 2. 进一步优化和添加元数据
        for chunk in chunks:
            # 确保元数据完整性
            final_metadata = base_metadata.copy()

            # 添加全局唯一的 chunk ID 或序号
            final_metadata[
                'chunk_id'] = f"{final_metadata.get('api_name', final_metadata.get('source', 'chunk'))}_{len(all_chunks)}"

            chunk.metadata = final_metadata
            all_chunks.append(chunk)

    print(f"-> 分块完成，共生成 {len(all_chunks)} 个高质量知识块。")
    return all_chunks


# --------------------
# 3. 主流程
# --------------------

def get_processed_chunks() -> List[Document]:
    """
    执行完整的文档加载和处理流程。
    """
    try:
        documents = load_documents()
        if not documents:
            return []
        processed_chunks = split_and_add_metadata(documents)
        return processed_chunks
    except Exception as e:
        print(f"处理文档时发生未知错误: {e}")
        return []