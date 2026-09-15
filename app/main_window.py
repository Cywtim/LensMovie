"""Main window: left parameter panel + right views (2D image + 3D scene).

Phase 2 wires the 3D vispy scene (lens mass plane + rays) alongside the 2D
lensed image. Both views update from the same slider parameters. The 3D view is
optional: if vispy cannot initialize, the app degrades gracefully to 2D only.
"""

from __future__ import annotations

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QSplitter,
    QWidget,
)

from . import sim2d
from .controls import ParameterPanel
from .plotting import ImageCanvas


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LensMovie — Interactive Lensing Viewer")
        self.resize(1300, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left: parameter panel
        self.controls = ParameterPanel()
        layout.addWidget(self.controls, 0)

        # Right: vertical stack of 2D image and 3D scene
        right = QSplitter()
        right.setOrientation(1)  # vertical

        self.canvas = ImageCanvas()
        right.addWidget(self.canvas)

        # Load 3D scene (defensive: degrade gracefully if unavailable).
        self.scene3d = None
        try:
            from .scene3d import Scene3D

            self.scene3d = Scene3D(size=(460, 360))
            right.addWidget(self.scene3d.native)
        except Exception as exc:  # vispy/GL unavailable
            self.statusBar().showMessage(f"3D scene unavailable: {exc}")

        right.setSizes([350, 300])
        layout.addWidget(right, 1)

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

        # Update the 3D scene with the same parameters (best effort).
        if self.scene3d is not None:
            try:
                self.scene3d.update_scene(lens=lens, source=source)
            except Exception as exc:
                self.statusBar().showMessage(f"3D update error: {exc}", 5000)

        self.statusBar().showMessage(
            f"grid {num_pix}²  max={image.max():.3e}", 2000
        )
        self._render_ok = True
