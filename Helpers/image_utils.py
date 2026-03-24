"""image_utils.py — shared image conversion helpers."""

import numpy as np
import cv2
from PyQt5.QtGui import QPixmap, QImage

__all__ = ["bytes_to_pixmap", "bgr_to_pixmap"]


def bytes_to_pixmap(raw: bytes) -> QPixmap:
    """Decode raw image bytes (any OpenCV-supported format) into a QPixmap."""
    buf = np.frombuffer(raw, dtype=np.uint8)
    mat = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if mat is None:
        return QPixmap()
    return bgr_to_pixmap(mat)


def bgr_to_pixmap(mat: np.ndarray) -> QPixmap:
    """Convert an OpenCV BGR (or grayscale) numpy array into a QPixmap."""
    if mat.ndim == 2:
        h, w = mat.shape
        qimg = QImage(mat.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, _ = mat.shape
        rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())