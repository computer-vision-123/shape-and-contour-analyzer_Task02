"""Main_window.py — Application entry point and main window."""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir   = os.path.abspath(os.path.join(current_dir, '..', 'build'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(build_dir)
sys.path.append(os.path.join(build_dir, 'Release'))

from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from styles import COLORS, get_base_styles, get_title_style, get_hint_style

try:
    from tab_canny import CannyTab
    HAS_CANNY = True
except ImportError:
    HAS_CANNY = False

from tab_contour import ContourTab


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing Pipeline")
        self.resize(1100, 750)
        self.setStyleSheet(get_base_styles())
        self._setup_ui()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # Header
        header = QLabel("Image Processing Suite")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet(get_title_style())
        root.addWidget(header)

        # Tabs
        self.tabs = QTabWidget()
        if HAS_CANNY:
            self.tabs.addTab(CannyTab(), "Edge Detection")
        self.tabs.addTab(ContourTab(), "Active Contour")
        root.addWidget(self.tabs)

        # Footer
        footer = QLabel("Load an image to get started | Adjust parameters in real-time")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet(f"""
            color: {COLORS['text_light']};
            background-color: {COLORS['white']};
            border-radius: 6px;
            padding: 6px;
            font-size: 11px;
        """)
        root.addWidget(footer)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())