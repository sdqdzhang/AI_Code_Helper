from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QSpinBox, QComboBox, QPushButton,
    QGridLayout, QGroupBox, QMessageBox, QApplication
)
from PyQt6.QtCore import Qt, QSettings, pyqtSignal
from rag_core.config import SETTINGS_FILE, DEFAULT_THEME


class SettingsWindow(QWidget):
    """
    设置窗口，用于配置 LLM 模型、K 值和主题。
    """
    # 信号：通知主应用设置已更新，需要重新加载配置
    settings_updated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置 - RAG 编程助手")  # 标题更新
        # 初始化 QSettings
        self.settings = QSettings(SETTINGS_FILE, QSettings.Format.IniFormat)
        self._setup_ui()
        self._load_settings()
        self.setFixedSize(450, 450)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)

        # --- 1. LLM/RAG 配置组 ---
        rag_group = QGroupBox("LLM & 检索配置")
        rag_layout = QGridLayout()
        rag_layout.setSpacing(10)

        # LLM 模型名称
        rag_layout.addWidget(QLabel("Ollama 模型名称:"), 0, 0)
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("例如: llama3.1")
        rag_layout.addWidget(self.model_input, 0, 1)

        # Ollama Base URL
        rag_layout.addWidget(QLabel("Ollama Base URL:"), 1, 0)
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("例如: http://localhost:11434")
        rag_layout.addWidget(self.url_input, 1, 1)

        # K 值 (检索数量)
        rag_layout.addWidget(QLabel("检索 K 值 (1-10):"), 2, 0)
        self.k_spinbox = QSpinBox()
        self.k_spinbox.setRange(1, 10)
        rag_layout.addWidget(self.k_spinbox, 2, 1)

        rag_group.setLayout(rag_layout)
        main_layout.addWidget(rag_group)

        # --- 2. 主题配置组 ---
        theme_group = QGroupBox("外观设置")
        theme_layout = QHBoxLayout()

        theme_layout.addWidget(QLabel("应用主题:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        theme_layout.addWidget(self.theme_combo)
        theme_layout.addStretch(1)

        theme_group.setLayout(theme_layout)
        main_layout.addWidget(theme_group)

        main_layout.addStretch(1)  # 填充剩余空间

        # --- 3. 动作按钮 (包含退出) ---
        button_layout = QHBoxLayout()

        # 新增退出按钮 (连接到 QApplication.instance().quit)
        exit_button = QPushButton("🔴 退出应用")
        exit_button.clicked.connect(QApplication.instance().quit)
        exit_button.setStyleSheet(
            "background-color: #E53E3E; color: white; border-radius: 8px; padding: 10px 15px; font-weight: bold;")

        save_button = QPushButton("💾 保存设置")
        save_button.clicked.connect(self._save_settings)
        save_button.setStyleSheet(
            "background-color: #4299E1; color: white; border-radius: 8px; padding: 10px 15px; font-weight: bold;")

        button_layout.addWidget(exit_button)  # 将退出按钮放在左侧
        button_layout.addStretch(1)
        button_layout.addWidget(save_button)

        main_layout.addLayout(button_layout)

    def _load_settings(self):
        """从 QSettings 加载配置并更新 UI 控件。"""
        from rag_core.config import DEFAULT_LLM_MODEL_NAME, DEFAULT_LLM_BASE_URL, DEFAULT_RETRIEVAL_K

        # 读取配置，如果不存在则使用 config.py 中的默认值
        model_name = self.settings.value("LLM_MODEL_NAME", DEFAULT_LLM_MODEL_NAME)
        base_url = self.settings.value("LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
        # QSettings 读取数字时可能是字符串，需要转换
        k_value = int(self.settings.value("RETRIEVAL_K", DEFAULT_RETRIEVAL_K))
        theme = self.settings.value("THEME", DEFAULT_THEME)

        self.model_input.setText(model_name)
        self.url_input.setText(base_url)
        self.k_spinbox.setValue(k_value)
        self.theme_combo.setCurrentText(theme)

    def _save_settings(self):
        """将 UI 控件中的值保存到 QSettings。"""
        model_name = self.model_input.text().strip()
        base_url = self.url_input.text().strip()
        k_value = self.k_spinbox.value()
        theme = self.theme_combo.currentText()

        if not model_name or not base_url:
            QMessageBox.warning(self, "输入错误", "模型名称和 Base URL 不能为空。")
            return

        self.settings.setValue("LLM_MODEL_NAME", model_name)
        self.settings.setValue("LLM_BASE_URL", base_url)
        self.settings.setValue("RETRIEVAL_K", k_value)
        self.settings.setValue("THEME", theme)
        self.settings.sync()  # 确保写入磁盘

        print("✅ 设置已保存，正在通知主应用更新配置...")
        self.settings_updated.emit()
        self.close()

    def get_current_settings(self):
        """提供一个接口让 main_app 获取最新的持久化设置。"""
        # 确保读取最新的值
        self.settings.sync()
        return {
            "model": self.settings.value("LLM_MODEL_NAME"),
            "url": self.settings.value("LLM_BASE_URL"),
            # 确保 K 值返回整数
            "k": int(self.settings.value("RETRIEVAL_K")),
            "theme": self.settings.value("THEME")
        }