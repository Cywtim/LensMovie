"""Main window: left parameter panel + right 2D image canvas.

Phase 1 wires the real-time slider updates to a 2D lensed-image render. The
right side is a matplotlib canvas; a 3D vispy scene slot will be added in
Phase 2 (see DESIGN.md).
"""

from __future__ import annotations

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QHBoxLayout, QMainWindow, QWidget

from . import sim2d
from .controls import ParameterPanel
from .plotting import ImageCanvas


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LensMovie — Interactive Lensing Viewer")
        self.resize(1100, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left: parameter panel
        self.controls = ParameterPanel()
        layout.addWidget(self.controls, 0)

        # Right: 2D image canvas (3D scene appended in Phase 2)
        self.canvas = ImageCanvas()
        layout.addWidget(self.canvas, 1)

        # Throttled real-time re-render
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(40)  # ~25 Hz max redraw
        self._timer.timeout.connect(self._rerender)

        self.controls.parametersChanged.connect(self._schedule_rerender)

        self._render_ok = False
        self._rerender()

    def _schedule_rerender(self, _params):
        # Restart the debounce timer on every change.
        self._timer.start()

    def _rerender(self):
        params = self.controls.as_dict()
        lens = params["lens"]
        source = params["source"]
        display = params["display"]
        num_pix = int(display["num_pix"])

        # Field of view matches num_pix * pixel scale.
        delta_pix = 0.05
        half = num_pix / 2 * delta_pix
        extent = (-half, half, -half, half)

        try:
            image = sim2d.render(
                lens=lens,
                source=source,
                num_pix=num_pix,
                delta_pix=delta_pix,
            )
        except Exception as exc:  # keep the UI alive if a render fails
            self.canvas.clear()
            self.statusBar().showMessage(f"render error: {exc}", 5000)
            self._render_ok = False
            return

        self.canvas.update_image(
            image,
            extent_arcsec=extent,
            colormap=display["colormap"],
            stretch=display["stretch"],
        )
        self.statusBar().showMessage(
            f"grid {num_pix}²  max={image.max():.3e}", 2000
        )
        self._render_ok = True
