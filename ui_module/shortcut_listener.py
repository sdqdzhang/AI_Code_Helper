import threading
from pynput import keyboard
from PyQt6.QtCore import QObject, pyqtSignal


class ShortcutListener(QObject):
    """
    使用 pynput 在后台监听全局快捷键 (Ctrl+Space)。
    """
    # 定义信号，用于通知主线程切换窗口可见性
    shortcut_pressed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.listener_thread = None
        self.hotkey = keyboard.Key.space
        self.modifier = keyboard.Key.ctrl_l  # 左侧 Ctrl 键

    def _on_press(self, key):
        """按键按下时的回调函数。"""
        try:
            # 检查是否同时按下了 Ctrl 和 Space
            if key == self.hotkey and self.modifier in self.current_keys:
                # 在按 Space 时触发信号
                # 必须使用 emit 才能安全地与 PyQt 主线程交互
                self.shortcut_pressed.emit()

        except AttributeError:
            pass  # 忽略非字符键

    def _on_release(self, key):
        """按键释放时的回调函数。"""
        try:
            if key in self.current_keys:
                self.current_keys.remove(key)
        except KeyError:
            pass
        except Exception as e:
            print(f"快捷键监听释放错误: {e}")

    def _start_listening(self):
        """在新的线程中启动 pynput 监听器。"""
        print("⌨️ 快捷键监听器启动: Ctrl + Space...")
        # 当前按下的键集合
        self.current_keys = set()

        # pynput 的 KeyCode 无法直接与 Key 进行集合操作，需要特殊处理
        def on_press_wrapper(key):
            if key not in self.current_keys:
                self.current_keys.add(key)
            self._on_press(key)

        with keyboard.Listener(on_press=on_press_wrapper, on_release=self._on_release) as listener:
            try:
                listener.join()
            except Exception as e:
                print(f"❌ 快捷键监听器线程异常终止: {e}")

    def start(self):
        """启动监听器线程。"""
        if self.listener_thread is None:
            self.listener_thread = threading.Thread(target=self._start_listening, daemon=True)
            self.listener_thread.start()

    def stop(self):
        """停止监听器 (通常在应用关闭时调用)。"""
        # pynput 的 listener.stop() 需要在 listener 内部调用
        # 这里只是提供一个停止的接口，实际应用中依赖 daemon 线程在主程序退出时自动关闭
        print("快捷键监听器停止。")