import sys
import os
import threading
import asyncio
from dotenv import load_dotenv

# 导入 PyQt 核心和并发模块
try:
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from PyQt6.QtCore import QTimer, pyqtSignal, QObject, QRunnable, QThreadPool, QSettings
except ImportError:
    print("❌ 错误：未找到 PyQt6 库。请确保已运行 'pip install PyQt6'")
    sys.exit(1)

# 导入 RAG 核心模块
from rag_core.db_manager import DBManager
from rag_core.rag_engine import RAGEngine
from rag_core.config import SETTINGS_FILE, DEFAULT_LLM_MODEL_NAME, DEFAULT_LLM_BASE_URL, DEFAULT_RETRIEVAL_K, \
    DEFAULT_THEME

# 导入 UI 模块
from ui_module.floating_window import FloatingWindow
from ui_module.shortcut_listener import ShortcutListener
from ui_module.settings_window import SettingsWindow


# --- 异步 RAG 任务工作器 ---

class RAGWorkerSignals(QObject):
    """定义 RAG 工作线程与主线程通信的信号。"""
    finished = pyqtSignal(str)  # 任务完成，发送结果字符串
    error = pyqtSignal(str)  # 任务失败，发送错误信息


class RAGWorker(QRunnable):
    """
    QRunnable 封装 RAGEngine 的异步调用。
    QRunnable 内部是同步执行的，但我们将用它来运行 asyncio 事件循环。
    """

    def __init__(self, engine: RAGEngine, query: str):
        super().__init__()
        self.engine = engine
        self.query = query
        self.signals = RAGWorkerSignals()

    def run(self):
        """
        在 QThreadPool 提供的线程中启动 asyncio 事件循环并运行 RAG 任务。
        """
        try:
            # 获取当前线程的事件循环，如果不存在则创建一个
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # 运行异步任务并同步等待结果
            result = loop.run_until_complete(
                self.engine.generate_answer(self.query)
            )
            self.signals.finished.emit(result)
        except Exception as e:
            # 捕获任何线程中的 LLM 或检索错误
            self.signals.error.emit(f"RAG 任务失败: {e}")


class RAGAssistantApp(QApplication):
    """
    RAG 助手应用程序的主控制类。
    """

    def __init__(self, argv):
        super().__init__(argv)

        # 1. 初始化设置和配置
        self.settings = QSettings(SETTINGS_FILE, QSettings.Format.IniFormat)
        self.current_config = self._load_initial_config()
        self.threadpool = QThreadPool.globalInstance()
        self.threadpool.setMaxThreadCount(2)  # 限制线程数

        # 2. 初始化核心 RAG 组件
        print("--- 正在初始化 RAG 核心服务 ---")
        self.db_manager = DBManager()
        self.rag_engine = RAGEngine(self.db_manager)

        # 3. 初始化 UI
        self.floating_window = FloatingWindow(self.rag_engine)
        self.settings_window = SettingsWindow()

        # 4. 初始化全局快捷键监听
        self.shortcut_listener = ShortcutListener()

        # 5. 应用初始配置
        self._apply_config(self.current_config)

        # 6. 连接信号和槽
        self.floating_window.query_submitted.connect(self._handle_query)
        # 连接打开设置窗口的信号
        self.floating_window.open_settings.connect(self.settings_window.show)
        self.shortcut_listener.shortcut_pressed.connect(self.floating_window.toggle_visibility)
        self.settings_window.settings_updated.connect(self._handle_settings_update)

        # 7. 启动快捷键监听
        self.shortcut_listener.start()

        # 初始时隐藏窗口
        self.floating_window.hide_window()

        print("--- RAG 核心初始化完成 ---")

    def _load_initial_config(self):
        """从 QSettings 加载启动配置。"""
        return {
            "model": self.settings.value("LLM_MODEL_NAME", DEFAULT_LLM_MODEL_NAME),
            "url": self.settings.value("LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
            # 注意类型转换
            "k": int(self.settings.value("RETRIEVAL_K", DEFAULT_RETRIEVAL_K)),
            "theme": self.settings.value("THEME", DEFAULT_THEME)
        }

    def _apply_config(self, config: dict):
        """应用配置到 RAGEngine 和 UI."""
        try:
            k = int(config.get("k", DEFAULT_RETRIEVAL_K))
            # 重新配置 RAG 引擎
            self.rag_engine.configure(
                llm_model_name=config.get("model", DEFAULT_LLM_MODEL_NAME),
                llm_base_url=config.get("url", DEFAULT_LLM_BASE_URL),
                k_value=k
            )
            # 更新 UI 主题
            self.floating_window.update_theme(config.get("theme", DEFAULT_THEME))
            self.current_config = config
        except Exception as e:
            QMessageBox.critical(
                self.floating_window,
                "配置错误",
                f"无法应用 RAG 配置。请检查 Ollama 模型名称或 Base URL 是否正确。错误: {e}"
            )

    def _handle_settings_update(self):
        """处理设置窗口发出的更新信号。"""
        new_config = self.settings_window.get_current_settings()
        self._apply_config(new_config)

    def _handle_query(self, query: str):
        """
        处理 UI 提交的查询，并将异步 RAG 任务提交给线程池。
        """
        worker = RAGWorker(self.rag_engine, query)

        # 连接信号
        worker.signals.finished.connect(self.floating_window.update_result)
        worker.signals.error.connect(self.floating_window.update_result)

        # 将 QRunnable 提交给线程池。
        self.threadpool.start(worker)


def main():
    """应用程序的主入口点。"""
    # 确保加载环境变量
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

    # 设置 QSettings 的组织和应用名称，用于持久化设置
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, os.path.dirname(__file__))

    app = RAGAssistantApp(sys.argv)

    # 命令行启动信息更新
    print("\n=======================================================")
    print("           RAG 编程助手 启动成功！")
    print("      使用快捷键 [Ctrl + Space] 呼出/隐藏窗口。")
    print("      在设置页 (⚙️) 中点击 '退出应用' 关闭程序。")
    print("=======================================================\n")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()