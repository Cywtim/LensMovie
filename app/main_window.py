"""Main window: top edge-on 3D bar, then a 2-row x 3-column display grid.

Layout (single window):

  +---------------------------------------------------------------------------+
  |                        3D scene (Vispy) — edge-on, full width              |
  +---------------------------------------------------------------------------+
  |  numPix [..]   colormap [..]   stretch [..]          (DisplayBar)          |
  +---------------------+---------------------+-------------------------------+
  |  Fermat potential   |  Lens image          |  Lenses configuration        |
  |                     |                     |   lens1: model/params/z  [+x] |
  +---------------------+---------------------+-------------------------------+
  |  Time delay         |  Critical curve      |  Sources configuration       |
  |                     |  + caustic           |   source1: pos/shape/z [+x]  |
  +---------------------+---------------------+-------------------------------+

All views are driven by one :class:`lensing_calc.Config` assembled from the
Lenses panel, Sources panel and DisplayBar.
"""

from __future__ import annotations

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from . import lensing_calc as lc
from .controls import DisplayBar, LensesPanel, SourcesPanel
from .plotting import CurvesCanvas, FieldCanvas, ImageCanvas


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LensMovie — Interactive Lensing Viewer")
        self.resize(1500, 1050)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ----------------------------------------------------------------- top 3D bar
        self.scene3d = None
        self._3d_wrap = QWidget()
        _3d_lay = QHBoxLayout(self._3d_wrap)
        _3d_lay.setContentsMargins(0, 0, 0, 0)
        try:
            from .scene3d import Scene3D

            self.scene3d = Scene3D(size=(1500, 280))
            _3d_lay.addWidget(self.scene3d.native)
        except Exception as exc:
            self.statusBar().showMessage(f"3D scene unavailable: {exc}")
            self._3d_wrap.hide()
        root.addWidget(self._3d_wrap)

        # ----------------------------------------------------------------- display bar
        self.display_bar = DisplayBar()
        root.addWidget(self.display_bar)

        # -------------------------------------------------------------- 2x3 grid
        grid = QGridLayout()
        grid.setContentsMargins(4, 2, 4, 4)

        # 2D canvases
        self.fermat_canvas = FieldCanvas("Fermat potential (relative)")
        self.image_canvas = ImageCanvas()
        self.delay_canvas = FieldCanvas("Time delay (relative)")
        self.curves_canvas = CurvesCanvas()

        # Config columns
        self.lenses_panel = LensesPanel()
        self.sources_panel = SourcesPanel()

        # Row 1: Fermat potential | Lens image | Lenses config
        grid.addWidget(self.fermat_canvas, 0, 0)
        grid.addWidget(self.image_canvas, 0, 1)
        grid.addWidget(self.lenses_panel, 0, 2)
        # Row 2: Time delay | Critical curve + caustic | Sources config
        grid.addWidget(self.delay_canvas, 1, 0)
        grid.addWidget(self.curves_canvas, 1, 1)
        grid.addWidget(self.sources_panel, 1, 2)

        # Column widths: the two 2D view columns get more room than the config
        # column (sliders need less space), keeping the three in balance.
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 3)
        grid.setColumnStretch(2, 2)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        root.addLayout(grid, 1)

        # Throttled redraw.
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._rerender)

        self.lenses_panel.changed.connect(self._schedule)
        self.sources_panel.changed.connect(self._schedule)
        self.display_bar.changed.connect(self._schedule)

        self._last_display = self.display_bar.display()
        self._render_ok = False
        self._rerender()

    def _schedule(self, *a):
        self._timer.start()

    def _build_config(self) -> lc.Config:
        d = self.display_bar.display()
        return lc.Config(
            lenses=self.lenses_panel.lens_list(),
            sources=self.sources_panel.source_list(),
            num_pix=d["num_pix"],
            delta_pix=0.05,
        )

    def _rerender(self):
        config = self._build_config()
        display = self.display_bar.display()

        result = lc.compute(config)
        if not result.ok:
            self.statusBar().showMessage(f"render error: {result.error}", 6000)
            self._render_ok = False
            return

        num_pix, delta = config.num_pix, config.delta_pix
        self.fermat_canvas.update_field(
            result.fermat, num_pix, delta,
            colormap=display["colormap"], stretch="linear", sym=True,
        )
        self.image_canvas.update_image(
            result.image, num_pix, delta, result.image_positions,
            colormap=display["colormap"], stretch=display["stretch"],
        )
        self.delay_canvas.update_field(
            result.time_delay, num_pix, delta,
            colormap=display["colormap"], stretch=display["stretch"],
        )
        self.curves_canvas.update_curves(
            result.cc_ra, result.cc_dec, result.caustic_ra, result.caustic_dec,
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
