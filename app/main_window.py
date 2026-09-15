"""Main window: top 3D bar, middle 2D two-column area, right config panel.

Layout (single window):

  +-------------------------------------------------------------+
  |                        3D scene (Vispy)                      |
  +---------------------+---------------------+-----------------+
  | Fermat potential    | Lensed image        | Config panel     |
  | Time delay          | + cc/caustic/images |  lenses/sources  |
  +---------------------+---------------------+-----------------+

All views are driven by one :class:`lensing_calc.Config` supplied by the panel.
"""

from __future__ import annotations

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from . import lensing_calc as lc
from .controls import ConfigPanel
from .plotting import FieldCanvas, ImageCanvas


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LensMovie — Interactive Lensing Viewer")
        self.resize(1440, 900)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ----------------------------------------------------------------- top 3D bar
        self.scene3d = None
        self._3d_widget = None
        self._3d_wrap = QWidget()
        self._3d_lay = QHBoxLayout(self._3d_wrap)
        self._3d_lay.setContentsMargins(0, 0, 0, 0)
        try:
            from .scene3d import Scene3D

            self.scene3d = Scene3D(size=(1400, 300))
            self._3d_widget = self.scene3d.native
            self._3d_lay.addWidget(self._3d_widget)
        except Exception as exc:
            self.statusBar().showMessage(f"3D scene unavailable: {exc}")
            self._3d_wrap.hide()
        root.addWidget(self._3d_wrap)

        # ----------------------------------------------------------------- middle
        middle = QSplitter()
        middle.setOrientation(0)  # horizontal

        # Left 2D two-column block
        leftblock = QWidget()
        leftlay = QHBoxLayout(leftblock)
        leftlay.setContentsMargins(4, 4, 4, 4)
        col1 = QVBoxLayout()
        self.fermat_canvas = FieldCanvas("Fermat potential (relative)")
        self.delay_canvas = FieldCanvas("Time delay (relative)")
        col1.addWidget(self.fermat_canvas)
        col1.addWidget(self.delay_canvas)
        col2 = QVBoxLayout()
        self.image_canvas = ImageCanvas()
        col2.addWidget(self.image_canvas)
        leftlay.addLayout(col1, 1)
        leftlay.addLayout(col2, 1)
        middle.addWidget(leftblock)

        # Right config panel
        self.controls = ConfigPanel()
        middle.addWidget(self.controls)

        middle.setSizes([900, 340])
        root.addWidget(middle, 1)

        # Throttled redraw.
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._rerender)

        self.controls.configChanged.connect(self._schedule)
        self._last_config = self.controls.as_config()
        self._render_ok = False
        self._rerender()

    def _schedule(self, *a):
        self._timer.start()

    def _rerender(self):
        config = self.controls.as_config()
        display = self.controls.display()

        result = lc.compute(config)
        if not result.ok:
            self.statusBar().showMessage(f"render error: {result.error}", 6000)
            self._render_ok = False
            return

        num_pix, delta = config.num_pix, config.delta_pix
        self.image_canvas.update_image(
            result.image, num_pix, delta,
            result.cc_ra, result.cc_dec, result.caustic_ra, result.caustic_dec,
            result.image_positions,
            colormap=display["colormap"], stretch=display["stretch"],
        )
        self.fermat_canvas.update_field(
            result.fermat, num_pix, delta,
            colormap=display["colormap"], stretch="linear", sym=True,
        )
        self.delay_canvas.update_field(
            result.time_delay, num_pix, delta,
            colormap=display["colormap"], stretch=display["stretch"],
        )

        if self.scene3d is not None:
            try:
                self.scene3d.update_scene(config, result)
            except Exception as exc:
                self.statusBar().showMessage(f"3D update error: {exc}", 5000)

        self.statusBar().showMessage(
            f"lenses={len(config.lenses)} sources={len(config.sources)} "
            f"grid={num_pix}²  ref_z={result.ref_z_source:.2f}", 3000
        )
        self._render_ok = True
