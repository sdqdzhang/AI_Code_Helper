
# RAG 编程助手 (RAG Programming Assistant)

**基于 LLM 的 Pandas 文档检索增强生成 (RAG) 助手，以桌面悬浮窗的形式提供实时代码和API查询帮助。**

本应用旨在帮助开发者通过自然语言快速查询本地化的 Pandas 官方文档，并由 Llama 3.1（或其他配置的 Ollama 模型）提供精准、简洁的中文代码示例和解释。

## ✨ 主要功能

* **精准 RAG 检索：** 利用阿里云 DashScope `text-embedding-v3` 对本地 Pandas 文档进行向量化，确保检索结果的高质量和相关性。

* **本地 LLM 支持：** 通过 **Ollama** 框架调用本地部署的 Llama 3.1 (或用户配置的模型) 进行问答，保护数据隐私并实现快速响应。

* **悬浮窗界面：** 采用 PyQt6 构建美观、简洁的桌面悬浮窗 UI。

* **全局快捷键：** 使用 `Ctrl + Space` 快速呼出/隐藏问答窗口，不干扰日常工作流程。

* **持久化配置：** 用户可在设置页面配置 LLM 模型名称、Ollama Base URL、检索 K 值和应用主题，所有设置自动保存。

* **跨语言问答：** 核心文档为英文，但 LLM 可进行实时翻译和推理，最终输出专业的中文答案。

## ⚙️ 先决条件 (Prerequisites)

为了运行本应用，您必须满足以下环境要求：

1. **Python 3.10+**

2. **Ollama 服务：**

   * 下载并安装 [Ollama](https://ollama.com/)。

   * 拉取并运行 LLM 模型（默认为 Llama 3.1）：

     ```bash
     ollama pull llama3.1
     ollama run llama3.1 # 运行一次以确保模型就绪
     ```

   * 确保 Ollama 服务在后台运行 (`ollama serve`)。

3. **DashScope API Key：** 用于文档向量化和查询向量化。

## 🚀 快速开始

### 1. 环境设置

```bash
# 克隆项目
git clone <your_repository_url>
cd code-helper

# 创建并激活虚拟环境 (推荐)
python -m venv panda_rag_helper
.\panda_rag_helper\Scripts\activate  # Windows
source panda_rag_helper/bin/activate # macOS/Linux

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API 密钥

在项目根目录下创建一个名为 `.env` 的文件，并填入您的 DashScope API 密钥：

```dotenv
# .env 文件内容
# 阿里云 DashScope API Key (用于 Embedding 模型)
DASHSCOPE_API_KEY="sk-..."
```

### 3. 构建 RAG 知识库

这是最关键的一步，它将 Pandas 文档转化为向量数据库。

**注意：此步骤会消耗 DashScope Embedding 的 API 费用。**

```bash
python build_index.py
```

执行后，程序会提示您确认操作，并在 `rag_core/chroma_db` 目录下生成向量索引文件。

### 4. 运行应用

```bash
python main_app.py
```

应用启动后，您将在命令行看到确认信息，但不会看到主窗口。

## 💻 使用指南

1. **呼出/隐藏窗口：** 在任意界面按下全局快捷键 **`Ctrl + Space`**。

2. **提问：** 在输入框中输入您的中文问题（例如：“如何使用 Pandas 对 DataFrame 进行分组并计算每个组的总和？”）。

3. **结果：** 模型将检索相关文档，并生成格式化（Markdown）的中文答案。

4. **退出应用：** 点击窗口右上角的 **`⚙️` (设置)** 按钮，在设置页面中点击 **`🔴 退出应用`**。

## 📦 部署为独立 EXE (Windows)

您可以使用 PyInstaller 将应用打包为独立的 `.exe` 文件，方便在其他 Windows 机器上部署。

1. **安装 PyInstaller:**

   ```bash
   pip install pyinstaller
   ```

2. **执行打包:**
   使用项目中的 `main_app.spec` 文件进行打包，确保所有数据和隐藏依赖被正确包含。

   ```bash
   pyinstaller main_app.spec
   ```

3. **运行要求：** 打包后的 `.exe` 仍然**要求用户的机器上独立安装并运行 Ollama 服务**。

## 🧠 RAG 核心技术原理（工作流程）

| 阶段 | 步骤 | 核心组件 | 关键操作/说明 |
| ----- | ----- | ----- | ----- |
| **I. 离线索引** | 1. 知识加载与分割 | Pandas Docs, `indexing_utils.py` | 加载所有英文文档，并按 ReST 结构分割成语义完整的知识块（Chunks）。 |
|  | 2. 向量化与存储 | DashScope Embedding (`text-embedding-v3`), ChromaDB | 使用 DashScope 将每个知识块转化为高维向量，存储到本地 ChromaDB 数据库。 |
| **II. 在线查询** | 3. 问题向量化 | DashScope Embedding | 使用**同样的** Embedding 模型，将用户输入的**中文问题**转化为查询向量。 |
|  | 4. 检索 (Retrieval) | ChromaDB | 在向量数据库中进行相似度搜索，找到与查询向量最接近的 K 个**英文**知识块。 |
|  | 5. 增强生成 (Generation) | Llama 3.1 (Ollama) | 将检索到的英文上下文和中文问题一起送入 Llama 3.1，模型基于上下文进行推理和翻译。 |
|  | 6. 最终输出 | Llama 3.1, UI | 模型输出格式化（Markdown）的**中文**答案。 |
