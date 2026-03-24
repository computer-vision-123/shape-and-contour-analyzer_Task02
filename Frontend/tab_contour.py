"""tab_contour.py — Active Contour (Snake) tab with drawable canvas and test cases."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QDoubleSpinBox, QSpinBox, QFileDialog,
    QTextEdit, QGroupBox, QFormLayout, QApplication,
    QSizePolicy, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView,
)
from PyQt5.QtGui import (
    QPixmap, QPainter, QPen, QColor,
    QCursor, QPolygon, QFont,
)
from PyQt5.QtCore import Qt, QPoint
import cv2

import cv_backend
from Helpers.image_utils import bgr_to_pixmap
from styles import COLORS, get_base_styles, get_title_style, get_hint_style

__all__ = ["ContourTab"]

# ---------------------------------------------------------------------------
# Test-case presets
# ---------------------------------------------------------------------------

TEST_CASES = [
    {"name": "hand.png",   "alpha": 0.5, "beta": 0.8, "gamma": 2.0, "window": 3, "iters": 300},
    {"name": "apple.png",  "alpha": 0.8, "beta": 0.4, "gamma": 2.5, "window": 3, "iters": 300},
    {"name": "leaves.png", "alpha": 0.3, "beta": 0.2, "gamma": 3.0, "window": 4, "iters": 400},
]

# Minimum squared pixel distance between successive drag samples
_DRAG_MIN_DIST_SQ = 8 ** 2

# Contour fill colour (R, G, B, A)
_FILL_COLOR = (0, 180, 230, 100)


# ---------------------------------------------------------------------------
# DrawableLabel
# ---------------------------------------------------------------------------

class DrawableLabel(QLabel):
    """QLabel that lets the user draw a closed contour by click-and-drag."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Load an image\n\nClick and drag to draw contour")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(500, 400)
        self.setStyleSheet(f"""
            border: 2px dashed {COLORS['accent_teal']};
            border-radius: 12px;
            background: {COLORS['bg_warm']};
            color: {COLORS['text_light']};
        """)
        self.setCursor(QCursor(Qt.CrossCursor))

        self._orig_bgr = None
        self._base_pixmap: QPixmap | None = None
        self.contour_points: list[QPoint] = []
        self._dragging = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_image(self, path: str) -> bool:
        """Load an image from *path*. Returns True on success."""
        img = cv2.imread(path)
        if img is None:
            return False
        self._orig_bgr = img
        self.contour_points = []
        self._rebuild_pixmap()
        return True

    def encoded_image(self) -> bytes | None:
        """Return the loaded image as PNG bytes, or None if no image is loaded."""
        if self._orig_bgr is None:
            return None
        ok, buf = cv2.imencode(".png", self._orig_bgr)
        return bytes(buf) if ok else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_pixmap(self) -> None:
        if self._orig_bgr is None:
            return
        self._base_pixmap = bgr_to_pixmap(self._orig_bgr)
        self.setPixmap(
            self._base_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.setText("")

    def _displayed_rect(self) -> tuple[float, float, float]:
        """Return (offset_x, offset_y, scale) for the currently displayed pixmap."""
        if self._base_pixmap is None:
            return 0.0, 0.0, 1.0
        pw, ph = self._base_pixmap.width(), self._base_pixmap.height()
        lw, lh = self.width(), self.height()
        scale = min(lw / pw, lh / ph)
        return (lw - pw * scale) / 2, (lh - ph * scale) / 2, scale

    def _label_to_img(self, pos: QPoint) -> QPoint | None:
        """Convert a label-space position to image-space. Returns None if out of bounds."""
        if self._base_pixmap is None:
            return None
        ox, oy, scale = self._displayed_rect()
        ix = int((pos.x() - ox) / scale)
        iy = int((pos.y() - oy) / scale)
        if 0 <= ix < self._base_pixmap.width() and 0 <= iy < self._base_pixmap.height():
            return QPoint(ix, iy)
        return None

    def _img_to_label(self, pt: QPoint) -> QPoint:
        """Convert an image-space point to label-space."""
        ox, oy, scale = self._displayed_rect()
        return QPoint(int(pt.x() * scale + ox), int(pt.y() * scale + oy))

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._base_pixmap:
            self.setPixmap(
                self._base_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        n = len(self.contour_points)
        if n < 1:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pts = [self._img_to_label(p) for p in self.contour_points]

        # Semi-transparent fill
        if n >= 3:
            painter.setBrush(QColor(*_FILL_COLOR))
            painter.setPen(Qt.NoPen)
            painter.drawPolygon(QPolygon(pts))

        # Outline
        pen = QPen(QColor(COLORS['accent_red']), 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        for i in range(n - 1):
            painter.drawLine(pts[i], pts[i + 1])

        # Closing segment (dashed while dragging, solid when done)
        if n >= 3:
            close_style = Qt.DashLine if self._dragging else Qt.SolidLine
            painter.setPen(QPen(QColor(COLORS['accent_red']), 2, close_style))
            painter.drawLine(pts[-1], pts[0])

        # Control points — first point highlighted in teal
        for i, p in enumerate(pts):
            color = COLORS['accent_teal'] if i == 0 else COLORS['accent_red']
            size = 8 if i == 0 else 5
            painter.setBrush(QColor(color))
            painter.setPen(QPen(QColor(COLORS['primary_dark']), 1))
            painter.drawEllipse(p, size, size)

        painter.end()

    def mousePressEvent(self, event) -> None:
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

    def mouseMoveEvent(self, event) -> None:
        if not self._dragging or self._orig_bgr is None:
            return
        pt = self._label_to_img(event.pos())
        if pt is None:
            return
        if self.contour_points:
            last = self.contour_points[-1]
            dx, dy = pt.x() - last.x(), pt.y() - last.y()
            if dx * dx + dy * dy < _DRAG_MIN_DIST_SQ:
                return
        self.contour_points.append(pt)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.update()


# ---------------------------------------------------------------------------
# SnakeCanvas
# ---------------------------------------------------------------------------

class SnakeCanvas(QWidget):
    """Widget containing the drawable image canvas and snake parameter controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self.setStyleSheet(get_base_styles())
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setSpacing(15)
        root.setContentsMargins(15, 15, 15, 15)

        # ---- Left: canvas ----
        left = QVBoxLayout()

        title = QLabel("Active Contour (Snake)")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(get_title_style())
        left.addWidget(title)

        self.img_label = DrawableLabel()
        left.addWidget(self.img_label)

        hint = QLabel("Left click & drag to draw contour | Right click to clear")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(get_hint_style())
        left.addWidget(hint)

        root.addLayout(left, stretch=3)

        # ---- Right: controls ----
        right = QVBoxLayout()
        right.setSpacing(10)

        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self._load_image)
        right.addWidget(load_btn)

        # Parameters
        param_box = QGroupBox("Parameters")
        form = QFormLayout(param_box)
        form.setSpacing(8)

        self.sp_alpha = self._make_double_spinbox(1.0)
        self.sp_beta  = self._make_double_spinbox(1.0)
        self.sp_gamma = self._make_double_spinbox(1.0)
        self.sp_iters = self._make_spinbox(100, max_val=5000)
        self.sp_win   = self._make_spinbox(2,   max_val=20)

        form.addRow("Elasticity (α):",      self.sp_alpha)
        form.addRow("Curvature (β):",       self.sp_beta)
        form.addRow("Image Gradient (γ):",  self.sp_gamma)
        form.addRow("Max Iterations:",      self.sp_iters)
        form.addRow("Search Window:",       self.sp_win)
        right.addWidget(param_box)

        # Run / Stop
        btn_row = QHBoxLayout()
        self.btn_run  = QPushButton("Run Snake")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_run.clicked.connect(self._run_snake)
        self.btn_stop.clicked.connect(self._stop_snake)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_stop)
        right.addLayout(btn_row)

        # Results
        res_box = QGroupBox("Results")
        res_layout = QVBoxLayout(res_box)
        self.lbl_perimeter = QLabel("Perimeter: —")
        self.lbl_area      = QLabel("Area: —")
        self.lbl_pts       = QLabel("Points: —")
        for lbl in (self.lbl_perimeter, self.lbl_area, self.lbl_pts):
            lbl.setStyleSheet("padding: 4px;")
            res_layout.addWidget(lbl)
        right.addWidget(res_box)

        # Chain code
        cc_box = QGroupBox("Chain Code (8-direction)")
        cc_layout = QVBoxLayout(cc_box)
        self.txt_chain = QTextEdit()
        self.txt_chain.setReadOnly(True)
        self.txt_chain.setMaximumHeight(100)
        cc_layout.addWidget(self.txt_chain)
        right.addWidget(cc_box)

        right.addStretch()
        root.addLayout(right, stretch=1)

    # ------------------------------------------------------------------
    # Widget factories
    # ------------------------------------------------------------------

    @staticmethod
    def _make_double_spinbox(value: float) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(0.0, 100.0)
        s.setValue(value)
        s.setSingleStep(0.1)
        s.setDecimals(2)
        return s

    @staticmethod
    def _make_spinbox(value: int, max_val: int) -> QSpinBox:
        s = QSpinBox()
        s.setRange(1, max_val)
        s.setValue(value)
        return s

    # ------------------------------------------------------------------
    # Public API (used by TestCasesTab)
    # ------------------------------------------------------------------

    def apply_params(self, alpha: float, beta: float, gamma: float,
                     window: int, iters: int) -> None:
        """Load a preset into the parameter spinboxes."""
        self.sp_alpha.setValue(alpha)
        self.sp_beta.setValue(beta)
        self.sp_gamma.setValue(gamma)
        self.sp_win.setValue(window)
        self.sp_iters.setValue(iters)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if path and self.img_label.load_image(path):
            self.lbl_perimeter.setText("Perimeter: —")
            self.lbl_area.setText("Area: —")
            self.lbl_pts.setText("Points: —")
            self.txt_chain.clear()

    def _stop_snake(self) -> None:
        self._running = False

    def _run_snake(self) -> None:
        pts = self.img_label.contour_points
        if len(pts) < 3:
            self.lbl_perimeter.setText("Draw a contour first (≥ 3 points)")
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
                self.sp_alpha.value(),
                self.sp_beta.value(),
                self.sp_gamma.value(),
                self.sp_win.value(),
            )
            self.img_label.contour_points = [QPoint(x, y) for x, y in new_points]
            self.img_label.update()
            QApplication.processEvents()
            if new_points == points:
                break
            points = new_points

        chain_raw   = cv_backend.snake_chain_code(points)
        perimeter   = cv_backend.snake_perimeter(chain_raw)
        area        = cv_backend.snake_area(points, chain_raw)
        chain_fmt   = cv_backend.snake_format_chain_code(chain_raw, 6)

        self.lbl_perimeter.setText(f"Perimeter: {perimeter:.1f} px")
        self.lbl_area.setText(f"Area: {area:.1f} px²")
        self.lbl_pts.setText(f"Points: {len(points)}")
        self.txt_chain.setPlainText(chain_fmt)

        self._running = False
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)


# ---------------------------------------------------------------------------
# TestCasesTab
# ---------------------------------------------------------------------------

class TestCasesTab(QWidget):
    """Table of preset test cases that populate SnakeCanvas parameters."""

    _PARAM_COLS  = ["α", "β", "γ", "Iterations", "Window"]
    _ALL_COLS    = ["Image"] + _PARAM_COLS + ["Action"]
    _PARAM_WIDTH = 70
    _BTN_WIDTH   = 90
    _ROW_HEIGHT  = 40

    def __init__(self, snake_canvas: SnakeCanvas, parent=None):
        super().__init__(parent)
        self._canvas = snake_canvas
        self.setStyleSheet(get_base_styles())
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        title = QLabel("Test Cases")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(get_title_style())
        layout.addWidget(title)

        self.table = QTableWidget(len(TEST_CASES), len(self._ALL_COLS))
        self.table.setHorizontalHeaderLabels(self._ALL_COLS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)                   # Image column
        for col in range(1, 6):                                             # Parameter columns
            hdr.setSectionResizeMode(col, QHeaderView.Fixed)
            self.table.setColumnWidth(col, self._PARAM_WIDTH)
        hdr.setSectionResizeMode(6, QHeaderView.Fixed)                     # Action column
        self.table.setColumnWidth(6, self._BTN_WIDTH)
        self.table.verticalHeader().setDefaultSectionSize(self._ROW_HEIGHT)

        mono = QFont("Consolas", 10)
        for row, tc in enumerate(TEST_CASES):
            self.table.setItem(row, 0, self._cell(tc["name"], center=False))
            self.table.setItem(row, 1, self._cell(f"{tc['alpha']:.1f}",  font=mono))
            self.table.setItem(row, 2, self._cell(f"{tc['beta']:.1f}",   font=mono))
            self.table.setItem(row, 3, self._cell(f"{tc['gamma']:.1f}",  font=mono))
            self.table.setItem(row, 4, self._cell(str(tc["iters"]),       font=mono))
            self.table.setItem(row, 5, self._cell(str(tc["window"]),      font=mono))
            self.table.setCellWidget(row, 6, self._make_apply_btn(row))

        layout.addWidget(self.table)

        hint = QLabel("💡 Click 'Apply' to load parameters into the Snake canvas")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(get_hint_style())
        layout.addWidget(hint)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cell(text: str, font: QFont | None = None, center: bool = True) -> QTableWidgetItem:
        item = QTableWidgetItem(str(text))
        item.setTextAlignment(Qt.AlignCenter if center else Qt.AlignLeft | Qt.AlignVCenter)
        if font:
            item.setFont(font)
        return item

    def _make_apply_btn(self, row: int) -> QWidget:
        """Return a centred 'Apply' button wrapped in a transparent QWidget."""
        btn = QPushButton("Apply")
        btn.setFixedSize(70, 28)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_teal']};
                color: {COLORS['primary_dark']};
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_red']};
                color: white;
            }}
        """)
        btn.clicked.connect(lambda _, r=row: self._apply(r))

        wrapper = QWidget()
        wrapper.setStyleSheet("background: transparent;")
        btn_layout = QHBoxLayout(wrapper)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setAlignment(Qt.AlignCenter)
        btn_layout.addWidget(btn)
        return wrapper

    def _apply(self, row: int) -> None:
        tc = TEST_CASES[row]
        self._canvas.apply_params(
            tc["alpha"], tc["beta"], tc["gamma"], tc["window"], tc["iters"]
        )


# ---------------------------------------------------------------------------
# ContourTab  (top-level tab inserted into MainWindow)
# ---------------------------------------------------------------------------

class ContourTab(QWidget):
    """Top-level tab that hosts SnakeCanvas and TestCasesTab as sub-tabs."""

    def __init__(self):
        super().__init__()
        self.setStyleSheet(get_base_styles())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.snake_canvas = SnakeCanvas()
        self.test_cases   = TestCasesTab(self.snake_canvas)

        self.tabs.addTab(self.snake_canvas, "Snake")
        self.tabs.addTab(self.test_cases,   "Test Cases")

        layout.addWidget(self.tabs)