import os
import sys
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QGroupBox, QSlider, QFileDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

import cv_backend

# Handles any file format (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)
def _bytes_to_pixmap(raw: bytes) -> QPixmap:
    buf = np.frombuffer(raw, dtype=np.uint8)
    mat = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if len(mat.shape) == 2:
        h, w = mat.shape
        qimg = QImage(mat.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, _ = mat.shape
        rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class CannyTab(QWidget):
    def __init__(self):
        super().__init__()
        self._original_bytes = None
        self._result_bytes = None

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(10, 10, 10, 10)

        # ── Image panels ─────────────────────────────────────────────────────
        panels = QHBoxLayout()

        orig_box = QGroupBox("Original")
        orig_layout = QVBoxLayout(orig_box)
        self._orig_label = QLabel("No image loaded.")
        self._orig_label.setAlignment(Qt.AlignCenter)
        self._orig_label.setMinimumSize(320, 280)
        orig_layout.addWidget(self._orig_label)
        open_btn = QPushButton("Open Image")
        open_btn.clicked.connect(self._open_image)
        orig_layout.addWidget(open_btn)
        panels.addWidget(orig_box)

        result_box = QGroupBox("Canny Output")
        result_layout = QVBoxLayout(result_box)
        self._result_label = QLabel("Result will appear here.")
        self._result_label.setAlignment(Qt.AlignCenter)
        self._result_label.setMinimumSize(320, 280)
        result_layout.addWidget(self._result_label)
        # Save button removed from here
        panels.addWidget(result_box)

        root.addLayout(panels, stretch=1)

        # ── Controls ─────────────────────────────────────────────────────────
        ctrl_box = QGroupBox("Canny Controls")
        ctrl_layout = QHBoxLayout(ctrl_box)

        ctrl_layout.addWidget(QLabel("Threshold %:"))

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(1, 100)
        self._slider.setValue(10)
        self._slider.setFixedWidth(200)
        self._slider.valueChanged.connect(lambda v: self._val_label.setText(str(v)))
        ctrl_layout.addWidget(self._slider)

        self._val_label = QLabel("10")
        self._val_label.setFixedWidth(30)
        ctrl_layout.addWidget(self._val_label)

        apply_btn = QPushButton("Apply Canny")
        apply_btn.clicked.connect(self._apply_canny)
        ctrl_layout.addWidget(apply_btn)

        ctrl_layout.addStretch()
        root.addWidget(ctrl_box)

        # ── Status ───────────────────────────────────────────────────────────
        self._status = QLabel("Open an image to get started.")
        self._status.setAlignment(Qt.AlignCenter)
        root.addWidget(self._status)

    # -------------------------------------------------------------------------

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        with open(path, "rb") as f:
            self._original_bytes = f.read()
        self._orig_label.setPixmap(
            _bytes_to_pixmap(self._original_bytes).scaled(
                self._orig_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        self._result_label.setText("Result will appear here.")
        self._result_bytes = None
        self._status.setText(f"Loaded: {os.path.basename(path)}")

    def _apply_canny(self):
        if not self._original_bytes:
            self._status.setText("Please open an image first.")
            return
        try:
            percent = self._slider.value()
            self._result_bytes = cv_backend.run_canny(self._original_bytes, percent)
            self._result_label.setPixmap(
                _bytes_to_pixmap(self._result_bytes).scaled(
                    self._result_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
            self._status.setText(f"Canny applied — threshold {percent}%.")
        except Exception as e:
            self._status.setText(f"Error: {e}")
