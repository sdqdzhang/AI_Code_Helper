import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QTextEdit, QPushButton, QLabel,
    QSizePolicy, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QSettings, QPoint, QRect
from PyQt6.QtGui import QFont, QColor, QPalette
from rag_core.config import SETTINGS_FILE


class FloatingWindow(QWidget):
    """
    悬浮窗口 UI 界面，用于问答交互。
    """
    # 定义信号
    query_submitted = pyqtSignal(str)
    open_settings = pyqtSignal()

    # 窗口默认大小和持久化设置
    DEFAULT_WIDTH = 550
    DEFAULT_HEIGHT = 450

    def __init__(self, rag_engine):
        super().__init__()
        self.rag_engine = rag_engine
        self.is_visible = False
        self.current_theme = "Light"
        # 初始化 QSettings
        self.settings = QSettings(SETTINGS_FILE, QSettings.Format.IniFormat)

        self._setup_ui()
        self._load_position()  # 加载上次保存的位置

    def _setup_ui(self):
        """初始化窗口和控件."""
        self.setWindowTitle("RAG 编程助手")  # 标题更新

        # 1. 窗口样式设置：无边框、悬浮、保持在顶层
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(QSize(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT))

        # 允许窗口使用鼠标拖动
        self.oldPos = None

        # 2. 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 3. 容器卡片 (用于实现圆角和阴影)
        self.container = QWidget()
        self.container.setObjectName("ContainerWidget")
        container_layout = QVBoxLayout(self.container)

        # 4. 标题/设置按钮行
        header_layout = QHBoxLayout()
        # 移除熊猫图标
        self.title_label = QLabel("RAG 编程助手")
        self.title_label.setObjectName("TitleLabel")
        self.title_label.setFont(QFont("Inter", 16, QFont.Weight.Bold))
        header_layout.addWidget(self.title_label)

        header_layout.addStretch(1)

        self.settings_button = QPushButton("⚙️")
        self.settings_button.setObjectName("SettingsButton")
        self.settings_button.setFixedSize(30, 30)
        self.settings_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.settings_button.clicked.connect(self.open_settings.emit)
        header_layout.addWidget(self.settings_button)

        container_layout.addLayout(header_layout)

        # 5. 结果显示区 (只读)
        self.output_area = QTextEdit()
        self.output_area.setObjectName("OutputArea")
        self.output_area.setReadOnly(True)
        self.output_area.setPlaceholderText("💡 按 Ctrl+Space 呼出，在这里输入您的问题...")
        # 启用 Markdown 渲染
        self.output_area.setMarkdown(self.output_area.placeholderText())
        self.output_area.setFont(QFont("Inter", 10))
        self.output_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        container_layout.addWidget(self.output_area)

        # 6. 输入区
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setObjectName("InputField")
        self.input_field.setPlaceholderText("输入查询...")
        self.input_field.setFont(QFont("Inter", 10))
        self.input_field.returnPressed.connect(self._handle_submit)  # 绑定回车键
        input_layout.addWidget(self.input_field)

        submit_button = QPushButton("提问")
        submit_button.setObjectName("SubmitButton")
        submit_button.setCursor(Qt.CursorShape.PointingHandCursor)
        submit_button.clicked.connect(self._handle_submit)
        input_layout.addWidget(submit_button)

        container_layout.addLayout(input_layout)

        # 7. 添加容器到主布局
        main_layout.addWidget(self.container)

    # --- 样式管理 ---
    def _get_light_style(self):
        """浅色模式 CSS."""
        return """
            #ContainerWidget {
                background-color: white;
                border-radius: 12px;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            }
            #TitleLabel { color: #4A5568; }
            #SettingsButton { border: none; font-size: 18px; color: #4A5568;}
            #SettingsButton:hover { background-color: rgba(0,0,0,0.1); border-radius: 5px; }
            #OutputArea { 
                background-color: #F7FAFC; 
                border: 1px solid #E2E8F0; 
                border-radius: 8px; 
                padding: 10px;
                color: #2D3748;
            }
            #InputField { 
                padding: 10px; 
                border: 1px solid #CBD5E0; 
                border-radius: 8px;
                color: #2D3748;
            }
            #SubmitButton {
                background-color: #4299E1;
                color: white;
                border-radius: 8px;
                padding: 10px 15px;
                font-weight: bold;
            }
            #SubmitButton:hover { background-color: #3182CE; }
        """

    def _get_dark_style(self):
        """深色模式 CSS."""
        return """
            #ContainerWidget {
                background-color: #2D3748;
                border-radius: 12px;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.1);
            }
            #TitleLabel { color: #E2E8F0; }
            #SettingsButton { border: none; font-size: 18px; color: #E2E8F0;}
            #SettingsButton:hover { background-color: rgba(255,255,255,0.1); border-radius: 5px; }
            #OutputArea { 
                background-color: #1A202C; 
                border: 1px solid #4A5568; 
                border-radius: 8px; 
                padding: 10px;
                color: #E2E8F0; 
            }
            #InputField { 
                padding: 10px; 
                border: 1px solid #4A5568; 
                border-radius: 8px;
                background-color: #1A202C;
                color: #E2E8F0;
            }
            #SubmitButton {
                background-color: #63B3ED;
                color: #1A202C;
                border-radius: 8px;
                padding: 10px 15px;
                font-weight: bold;
            }
            #SubmitButton:hover { background-color: #4299E1; }
        """

    def update_theme(self, theme: str):
        """根据设置应用主题."""
        self.current_theme = theme
        if theme == "Dark":
            self.container.setStyleSheet(self._get_dark_style())
        else:
            self.container.setStyleSheet(self._get_light_style())

    # --- 窗口位置管理 ---
    def _load_position(self):
        """加载上次保存的窗口位置，如果不存在则居中."""
        # 尝试加载 QPoint 类型
        pos = self.settings.value("WindowPosition", QPoint())
        size = self.settings.value("WindowSize", QSize(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT))

        # 校验位置和尺寸是否有效
        if pos.isNull() or not QApplication.primaryScreen().geometry().contains(QRect(pos, size)):
            # 居中逻辑
            screen = QApplication.primaryScreen().geometry()
            x = (screen.width() - self.DEFAULT_WIDTH) // 2
            y = (screen.height() - self.DEFAULT_HEIGHT) // 2
            self.move(x, y)
            self.resize(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
        else:
            self.move(pos)
            self.resize(size)

    def _save_position(self):
        """保存当前窗口位置和大小."""
        self.settings.setValue("WindowPosition", self.pos())
        self.settings.setValue("WindowSize", self.size())

    # --- 核心交互 ---
    def _handle_submit(self):
        """处理提问事件."""
        query = self.input_field.text().strip()
        if query:
            # 清空输入并显示思考状态
            self.output_area.setPlaceholderText("正在思考...")
            # 将提示设置为纯文本，否则 setMarkdown("") 会显示 placeholder
            self.output_area.setText("正在思考...")
            self.input_field.setEnabled(False)
            self.query_submitted.emit(query)

    def update_result(self, result: str):
        """在输出区域显示 RAG 引擎的返回结果，支持 Markdown。"""
        # 使用 setMarkdown 渲染 LLM 返回的 Markdown 文本
        self.output_area.setMarkdown(result)
        self.output_area.setPlaceholderText("在这里输入您的问题...")
        self.input_field.setEnabled(True)
        self.input_field.clear()

    # --- 剪贴板集成和显示 ---
    def _populate_input_with_clipboard(self):
        """读取剪贴板内容并填充到输入框。"""
        clipboard = QApplication.clipboard()
        text = clipboard.text().strip()

        # 仅在文本非空时填充，否则保持清空
        self.input_field.setText(text)
        self.input_field.selectAll()  # 选中全部内容，方便用户直接修改或覆盖

        # 如果有内容，自动将焦点移动到输入框，方便用户操作
        if text:
            self.input_field.setFocus()

    def show_window(self):
        """显示窗口并加载剪贴板内容。"""
        self._populate_input_with_clipboard()
        self.show()
        self.is_visible = True
        # 确保焦点在输入框，不论剪贴板是否有内容
        self.input_field.setFocus()

    def hide_window(self):
        """隐藏窗口并保存位置."""
        self._save_position()
        self.hide()
        self.is_visible = False

    def toggle_visibility(self):
        """切换窗口的可见性."""
        if self.is_visible:
            self.hide_window()
        else:
            self.show_window()

    # --- 鼠标拖动实现无边框窗口移动 ---
    def mousePressEvent(self, event):
        # 仅允许在标题栏区域拖动
        if event.button() == Qt.MouseButton.LeftButton and self.title_label.geometry().contains(event.pos()):
            self.oldPos = event.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.oldPos:
            delta = event.pos() - self.oldPos
            self.move(self.pos() + delta)
            event.accept()

    def mouseReleaseEvent(self, event):
        self.oldPos = None
        event.accept()

    # 确保退出时保存位置
    def closeEvent(self, event):
        self._save_position()
        super().closeEvent(event)