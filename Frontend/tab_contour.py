"""
tab_contour.py  –  Active-Contour (Snake) tab
  • Sub-tab 1: Snake canvas (draw + run)
  • Sub-tab 2: Test cases table (name + params + apply button)
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QDoubleSpinBox, QSpinBox, QFileDialog,
                              QTextEdit, QGroupBox, QFormLayout, QApplication,
                              QSizePolicy, QTabWidget, QTableWidget,
                              QTableWidgetItem, QHeaderView, QAbstractItemView)
from PyQt5.QtGui  import (QPixmap, QPainter, QPen, QColor, QImage,
                           QCursor, QPolygon, QFont)
from PyQt5.QtCore import Qt, QPoint
import cv2
import numpy as np
import cv_backend


# ─────────────────────────────────────────────────────────────────────────────
#  Test-case data
# ─────────────────────────────────────────────────────────────────────────────
TEST_CASES = [
    {"name": "hand.png",                              "alpha": 0.5, "beta": 0.8, "gamma": 2.0, "window": 3, "iters": 300},
    {"name": "apple.png",                    "alpha": 0.8, "beta": 0.4, "gamma": 2.5, "window": 3, "iters": 300},
    {"name": "leaves.png", "alpha": 0.3, "beta": 0.2, "gamma": 3.0, "window": 4, "iters": 400},
]


# ─────────────────────────────────────────────────────────────────────────────
#  DrawableLabel
# ─────────────────────────────────────────────────────────────────────────────
class DrawableLabel(QLabel):
    DRAG_MIN_DIST = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Load an image to begin")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(500, 400)
        self.setStyleSheet("border: 1px solid #555; background: #1e1e1e; color: #aaa;")
        self.setCursor(QCursor(Qt.CrossCursor))
        self._orig_bgr    = None
        self._base_pixmap = None
        self.contour_points = []
        self._dragging = False

    def load_image(self, path: str) -> bool:
        img = cv2.imread(path)
        if img is None:
            return False
        self._orig_bgr = img
        self.contour_points = []
        self._rebuild_pixmap()
        return True

    def _rebuild_pixmap(self):
        if self._orig_bgr is None:
            return
        rgb = cv2.cvtColor(self._orig_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        rgb_c = np.ascontiguousarray(rgb)
        qimg = QImage(rgb_c.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self._base_pixmap = QPixmap.fromImage(qimg)
        self.setPixmap(self._base_pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.setText("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._base_pixmap:
            self.setPixmap(self._base_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def paintEvent(self, event):
        super().paintEvent(event)
        n = len(self.contour_points)
        if n < 1:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pts = [self._img_to_label(p) for p in self.contour_points]
        if n >= 3:
            painter.setBrush(QColor(0, 180, 255, 40))
            painter.setPen(Qt.NoPen)
            painter.drawPolygon(QPolygon(pts))
        pen = QPen(QColor(0, 220, 80), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        for i in range(n - 1):
            painter.drawLine(pts[i], pts[i + 1])
        if n >= 3:
            painter.setPen(QPen(QColor(0, 220, 80), 2,
                                Qt.DashLine if self._dragging else Qt.SolidLine))
            painter.drawLine(pts[-1], pts[0])
        for i, p in enumerate(pts):
            if i == 0:
                painter.setBrush(QColor(255, 220, 0))
                painter.setPen(QPen(QColor(180, 140, 0), 1))
                painter.drawEllipse(p, 6, 6)
            else:
                painter.setBrush(QColor(255, 80, 80))
                painter.setPen(QPen(QColor(180, 30, 30), 1))
                painter.drawEllipse(p, 3, 3)
        painter.end()

    def mousePressEvent(self, event):
        if self._orig_bgr is None:
            return
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self.contour_points = []
            pt = self._label_to_img(event.pos())
            if pt:
                self.contour_points.append(pt)
            self.update()
        elif event.button() == Qt.RightButton:
            self.contour_points = []
            self._dragging = False
            self.update()

    def mouseMoveEvent(self, event):
        if not self._dragging or self._orig_bgr is None:
            return
        pt = self._label_to_img(event.pos())
        if pt is None:
            return
        if self.contour_points:
            last = self.contour_points[-1]
            dx, dy = pt.x() - last.x(), pt.y() - last.y()
            if dx * dx + dy * dy < self.DRAG_MIN_DIST ** 2:
                return
        self.contour_points.append(pt)
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.update()

    def _displayed_rect(self):
        if self._base_pixmap is None:
            return 0.0, 0.0, 1.0
        pw, ph = self._base_pixmap.width(), self._base_pixmap.height()
        lw, lh = self.width(), self.height()
        scale = min(lw / pw, lh / ph)
        return (lw - pw * scale) / 2, (lh - ph * scale) / 2, scale

    def _label_to_img(self, pos):
        if self._base_pixmap is None:
            return None
        ox, oy, scale = self._displayed_rect()
        ix = int((pos.x() - ox) / scale)
        iy = int((pos.y() - oy) / scale)
        pw, ph = self._base_pixmap.width(), self._base_pixmap.height()
        if 0 <= ix < pw and 0 <= iy < ph:
            return QPoint(ix, iy)
        return None

    def _img_to_label(self, pt):
        ox, oy, scale = self._displayed_rect()
        return QPoint(int(pt.x() * scale + ox), int(pt.y() * scale + oy))

    def encoded_image(self):
        if self._orig_bgr is None:
            return None
        ok, buf = cv2.imencode(".png", self._orig_bgr)
        return bytes(buf) if ok else None


# ─────────────────────────────────────────────────────────────────────────────
#  Snake canvas sub-tab
# ─────────────────────────────────────────────────────────────────────────────
class SnakeCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        self.img_label = DrawableLabel()
        left.addWidget(self.img_label)
        hint = QLabel("Hold & drag to draw contour  |  Right-click to clear")
        hint.setStyleSheet("color: #888; font-size: 11px; padding: 2px;")
        left.addWidget(hint)
        root.addLayout(left, stretch=3)

        right = QVBoxLayout()
        right.setAlignment(Qt.AlignTop)

        btn_load = QPushButton("Load Image")
        btn_load.clicked.connect(self._load_image)
        right.addWidget(btn_load)

        param_box = QGroupBox("Snake Parameters")
        form = QFormLayout(param_box)

        def dbl(val):
            s = QDoubleSpinBox()
            s.setRange(0, 100); s.setValue(val); s.setSingleStep(0.1)
            return s
        def spin(val, hi=5000):
            s = QSpinBox(); s.setRange(1, hi); s.setValue(val); return s

        self.sp_alpha = dbl(1.0)
        self.sp_beta  = dbl(1.0)
        self.sp_gamma = dbl(1.0)
        self.sp_iters = spin(100)
        self.sp_win   = spin(2, hi=20)

        form.addRow("α  (elasticity):", self.sp_alpha)
        form.addRow("β  (curvature):",  self.sp_beta)
        form.addRow("γ  (image grad):", self.sp_gamma)
        form.addRow("Max iterations:",  self.sp_iters)
        form.addRow("Search window ½:", self.sp_win)
        right.addWidget(param_box)

        btn_row = QHBoxLayout()
        self.btn_run  = QPushButton("Run Snake")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_run.clicked.connect(self._run_snake)
        self.btn_stop.clicked.connect(lambda: setattr(self, '_running', False))
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_stop)
        right.addLayout(btn_row)

        res_box = QGroupBox("Results")
        res_l = QVBoxLayout(res_box)
        self.lbl_perimeter = QLabel("Perimeter: —")
        self.lbl_area      = QLabel("Area: —")
        self.lbl_pts       = QLabel("Points: —")
        res_l.addWidget(self.lbl_perimeter)
        res_l.addWidget(self.lbl_area)
        res_l.addWidget(self.lbl_pts)
        right.addWidget(res_box)

        cc_box = QGroupBox("Chain Code (8-direction)")
        cc_l = QVBoxLayout(cc_box)
        self.txt_chain = QTextEdit()
        self.txt_chain.setReadOnly(True)
        self.txt_chain.setMaximumHeight(120)
        self.txt_chain.setStyleSheet("font-family: monospace; font-size: 11px;")
        cc_l.addWidget(self.txt_chain)
        right.addWidget(cc_box)

        right.addStretch()
        root.addLayout(right, stretch=1)

    def apply_params(self, alpha, beta, gamma, window, iters):
        self.sp_alpha.setValue(alpha)
        self.sp_beta.setValue(beta)
        self.sp_gamma.setValue(gamma)
        self.sp_win.setValue(window)
        self.sp_iters.setValue(iters)

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path:
            if not self.img_label.load_image(path):
                self.lbl_perimeter.setText("Failed to load image")
            else:
                self.lbl_perimeter.setText("Perimeter: —")
                self.lbl_area.setText("Area: —")
                self.lbl_pts.setText("Points: —")
                self.txt_chain.clear()

    def _run_snake(self):
        pts = self.img_label.contour_points
        if len(pts) < 3:
            self.lbl_perimeter.setText("Draw a contour first (need ≥ 3 points)")
            return
        img_bytes = self.img_label.encoded_image()
        if img_bytes is None:
            self.lbl_perimeter.setText("No image loaded")
            return

        points = [(p.x(), p.y()) for p in pts]
        self._running = True
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

        for _ in range(self.sp_iters.value()):
            if not self._running:
                break
            new_points = cv_backend.snake_evolve_once(
                img_bytes, points,
                self.sp_alpha.value(), self.sp_beta.value(),
                self.sp_gamma.value(), self.sp_win.value())
            self.img_label.contour_points = [QPoint(x, y) for x, y in new_points]
            self.img_label.update()
            QApplication.processEvents()
            if new_points == points:
                break
            points = new_points

        chain_raw = cv_backend.snake_chain_code(points)
        perimeter = cv_backend.snake_perimeter(chain_raw)
        area      = cv_backend.snake_area(points, chain_raw)
        chain_fmt = cv_backend.snake_format_chain_code(chain_raw, 6)

        self.lbl_perimeter.setText(f"Perimeter: {perimeter:.1f} px")
        self.lbl_area.setText(f"Area: {area:.1f} px²")
        self.lbl_pts.setText(f"Points: {len(points)}")
        self.txt_chain.setPlainText(chain_fmt)

        self._running = False
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)


# ─────────────────────────────────────────────────────────────────────────────
#  Test cases sub-tab
# ─────────────────────────────────────────────────────────────────────────────
class TestCasesTab(QWidget):
    def __init__(self, snake_canvas: SnakeCanvas, parent=None):
        super().__init__(parent)
        self._canvas = snake_canvas
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        cols = ["Image name", "α", "β", "γ", "Max iter", "Window", ""]
        self.table = QTableWidget(len(TEST_CASES), len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(True)

        mono = QFont("Consolas", 10)

        for row, tc in enumerate(TEST_CASES):
            def cell(text, font=None, center=True):
                it = QTableWidgetItem(str(text))
                it.setTextAlignment((Qt.AlignCenter if center else Qt.AlignLeft | Qt.AlignVCenter))
                if font:
                    it.setFont(font)
                return it

            self.table.setItem(row, 0, cell(tc["name"],          center=False))
            self.table.setItem(row, 1, cell(tc["alpha"],         font=mono))
            self.table.setItem(row, 2, cell(tc["beta"],          font=mono))
            self.table.setItem(row, 3, cell(tc["gamma"],         font=mono))
            self.table.setItem(row, 4, cell(tc["iters"],         font=mono))
            self.table.setItem(row, 5, cell(tc["window"],        font=mono))

            btn = QPushButton("Apply")
            btn.setStyleSheet("font-size: 11px; padding: 3px 10px;")
            btn.clicked.connect(lambda _, r=row: self._apply(r))
            self.table.setCellWidget(row, 6, btn)

        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 6):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(6, QHeaderView.ResizeToContents)

        self.table.resizeRowsToContents()
        layout.addWidget(self.table)
        layout.addStretch()

    def _apply(self, row: int):
        tc = TEST_CASES[row]
        self._canvas.apply_params(
            tc["alpha"], tc["beta"], tc["gamma"], tc["window"], tc["iters"])


# ─────────────────────────────────────────────────────────────────────────────
#  ContourTab
# ─────────────────────────────────────────────────────────────────────────────
class ContourTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.snake_canvas = SnakeCanvas()
        self.test_cases   = TestCasesTab(self.snake_canvas)

        self.tabs.addTab(self.snake_canvas, "Snake")
        self.tabs.addTab(self.test_cases,   "Test Cases")

        layout.addWidget(self.tabs)