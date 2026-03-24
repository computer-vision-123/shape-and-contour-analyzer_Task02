"""tab_canny.py — Canny edge-detection tab with line/circle/ellipse detection."""

import os

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QGroupBox, QSlider, QFileDialog, QFrame,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap

import cv_backend
from Helpers.image_utils import bytes_to_pixmap
from styles import COLORS, get_base_styles, get_title_style, get_hint_style

__all__ = ["CannyTab"]


class CannyTab(QWidget):
    """Tab for Canny edge detection and basic shape recognition."""

    def __init__(self):
        super().__init__()
        self._original_bytes: bytes | None = None
        self._result_bytes: bytes | None = None
        self._fixed_display_size: QSize | None = None  # locked in when image is first loaded

        self.setStyleSheet(get_base_styles())
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(15)
        root.setContentsMargins(15, 15, 15, 15)

        # Title
        title = QLabel("Edge Detection")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(get_title_style())
        root.addWidget(title)

        # Image panels
        panels = QHBoxLayout()
        panels.setSpacing(15)

        orig_box = QGroupBox("Original Image")
        orig_layout = QVBoxLayout(orig_box)
        self._orig_label = self._make_image_label()
        orig_layout.addWidget(self._orig_label)
        open_btn = QPushButton("Load Image")
        open_btn.clicked.connect(self._open_image)
        orig_layout.addWidget(open_btn)
        panels.addWidget(orig_box)

        result_box = QGroupBox("Edge Detection Result")
        result_layout = QVBoxLayout(result_box)
        self._result_label = self._make_image_label("Result will appear here")
        result_layout.addWidget(self._result_label)
        panels.addWidget(result_box)

        root.addLayout(panels, stretch=2)

        # Controls
        ctrl_box = QGroupBox("Detection Controls")
        ctrl_layout = QHBoxLayout(ctrl_box)
        ctrl_layout.setSpacing(15)

        # Threshold slider
        thresh_group = QWidget()
        thresh_layout = QHBoxLayout(thresh_group)
        thresh_layout.setContentsMargins(0, 0, 0, 0)
        thresh_layout.addWidget(QLabel("Threshold:"))

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(1, 100)
        self._slider.setValue(10)
        self._slider.setFixedWidth(200)
        self._slider.valueChanged.connect(
            lambda v: self._val_label.setText(str(v))
        )
        thresh_layout.addWidget(self._slider)

        self._val_label = QLabel("10")
        self._val_label.setFixedWidth(35)
        self._val_label.setStyleSheet(f"""
            background-color: {COLORS['white']};
            border: 1px solid {COLORS['accent_teal']};
            border-radius: 4px;
            padding: 2px 6px;
            font-weight: bold;
        """)
        thresh_layout.addWidget(self._val_label)
        ctrl_layout.addWidget(thresh_group)

        apply_btn = QPushButton("Apply Canny")
        apply_btn.clicked.connect(self._apply_canny)
        ctrl_layout.addWidget(apply_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"background-color: {COLORS['accent_teal']};")
        ctrl_layout.addWidget(sep)

        self._lines_btn = QPushButton("Detect Lines")
        self._lines_btn.clicked.connect(self._detect_lines)
        ctrl_layout.addWidget(self._lines_btn)

        self._circles_btn = QPushButton("Detect Circles")
        self._circles_btn.clicked.connect(self._detect_circles)
        ctrl_layout.addWidget(self._circles_btn)

        self._ellipses_btn = QPushButton("Detect Ellipses")
        self._ellipses_btn.clicked.connect(self._detect_ellipses)
        ctrl_layout.addWidget(self._ellipses_btn)

        ctrl_layout.addStretch()
        root.addWidget(ctrl_box)

        # Status bar
        self._status = QLabel("Ready. Load an image to begin.")
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setStyleSheet(get_hint_style())
        root.addWidget(self._status)

    @staticmethod
    def _make_image_label(placeholder: str = "No image loaded") -> QLabel:
        """Return a styled placeholder label for image display."""
        label = QLabel(placeholder)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(350, 300)
        label.setStyleSheet(f"""
            border: 2px dashed {COLORS['accent_teal']};
            border-radius: 8px;
            background-color: {COLORS['bg_warm']};
            padding: 20px;
            color: {COLORS['text_light']};
        """)
        return label

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scale_pixmap(self, pixmap: QPixmap) -> QPixmap:
        """Scale pixmap to the fixed display size captured at load time."""
        if self._fixed_display_size is None:
            return pixmap
        return pixmap.scaled(self._fixed_display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _set_status(self, text: str) -> None:
        self._status.setText(text)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        with open(path, "rb") as f:
            self._original_bytes = f.read()

        # Lock in the display size based on the label's current size at load time.
        # All subsequent pixmaps (Canny, lines, circles, ellipses) will scale to
        # this same size so the displayed image never jumps around.
        self._fixed_display_size = self._orig_label.size()

        self._orig_label.setPixmap(
            self._scale_pixmap(bytes_to_pixmap(self._original_bytes))
        )
        self._result_label.setText("Result will appear here")
        self._result_bytes = None
        self._set_status(f"Loaded: {os.path.basename(path)}")

    def _apply_canny(self) -> None:
        if not self._original_bytes:
            self._set_status("Please open an image first.")
            return
        try:
            percent = self._slider.value()
            self._result_bytes = cv_backend.run_canny(self._original_bytes, percent)
            self._result_label.setPixmap(
                self._scale_pixmap(bytes_to_pixmap(self._result_bytes))
            )
            self._set_status(f"Canny applied — threshold {percent}%.")
        except Exception as exc:
            self._set_status(f"Error: {exc}")

    def _detect_shapes(self, shape_func, label: str) -> None:
        """Run shape_func(original_bytes, edge_bytes) and display the result."""
        if not self._result_bytes:
            self._set_status("Apply Canny first.")
            return
        if not self._original_bytes:
            self._set_status("No original image available.")
            return
        try:
            annotated = shape_func(self._original_bytes, self._result_bytes)
            self._result_label.setPixmap(
                self._scale_pixmap(bytes_to_pixmap(annotated))
            )
            self._set_status(f"{label} detected.")
        except Exception as exc:
            self._set_status(f"Error: {exc}")

    def _detect_lines(self) -> None:
        self._detect_shapes(cv_backend.detect_lines, "Lines")

    def _detect_circles(self) -> None:
        self._detect_shapes(cv_backend.detect_circles, "Circles")

    def _detect_ellipses(self) -> None:
        self._detect_shapes(cv_backend.detect_ellipses, "Ellipses")