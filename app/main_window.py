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

This class is the **presenter** (the UI half).  All program state lives in
:class:`app.controller.LensMovieController`, which this window owns and
subscribes to.  The controller never knows about widgets; this class never
duplicates program state — it maps widgets <-> controller and draws the
results.  Physics stays in ``lensing_calc`` and all *look & feel* lives in
``app/theme.qss``, so the two can be updated independently.
"""

from __future__ import annotations

import os

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from . import lensing_calc as lc
from .controller import LensMovieController
from .controls import (DataBar, DisplayBar, FitBar, LensesPanel, PointSourcesPanel,
                       SourcesPanel)
from .plotting import CurvesCanvas, ExternalCanvas, FieldCanvas, ImageCanvas


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LensMovie — Interactive Lensing Viewer")
        self.resize(1500, 1050)

        # --------------------------------------------------------------------
        # Program state lives in the controller; this window is its presenter.
        # --------------------------------------------------------------------
        self.controller = LensMovieController(self)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ------------------------------------------------- top area: 3D bar + external image
        # The external-image panel now spans BOTH the 3D bar row and the numPix
        # display row — a tall strip on the right instead of a small square.
        self._3d_height = 230

        self.scene3d = None
        self._3d_wrap = QWidget()
        self._3d_wrap.setFixedHeight(self._3d_height)
        # Black background: when 3D is switched off only this black area remains
        # (the space is preserved instead of collapsing).
        self._3d_wrap.setAutoFillBackground(True)
        self._3d_wrap.setStyleSheet("background-color: black;")
        _3d_lay = QHBoxLayout(self._3d_wrap)
        _3d_lay.setContentsMargins(0, 0, 0, 0)
        try:
            from .scene3d import Scene3D

            self.scene3d = Scene3D(size=(1300, self._3d_height))
            self._3d_native = self.scene3d.native
            self._3d_native.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self._3d_native.setMinimumHeight(self._3d_height)
            self._3d_native.setMaximumHeight(self._3d_height)
            self._3d_native.setMinimumWidth(200)
            _3d_lay.addWidget(self._3d_native, 1)
        except Exception as exc:
            self._3d_native = None
            self.statusBar().showMessage(f"3D scene unavailable: {exc}")

        # External-image panel: taller than the 3D bar (spans the 3D + display
        # rows) and a bit wider than before.  It is no longer a fixed square.
        # It sits in a horizontal splitter with the 3D/left column, so the user
        # can drag the divider; a floor keeps it usable, no ceiling caps it.
        self.ext_panel = QGroupBox("External image / fit data")
        self.ext_panel.setMinimumWidth(280)
        # Vertical size hint ignored: the matplotlib canvas inside would push the
        # whole top block to ~390 px tall, shrinking the 2D grid.  The cap keeps
        # the top block's minimum modest, and the panel is still stretched to
        # fill the top block's full height.
        self.ext_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored)
        self.ext_panel.setMinimumHeight(250)
        ext_lay = QVBoxLayout(self.ext_panel)
        ext_lay.setContentsMargins(4, 4, 4, 4)
        self.external_canvas = ExternalCanvas()
        ext_lay.addWidget(self.external_canvas, 1)
        # What the external panel shows: the raw data, the data resampled onto
        # the model grid, or the fit's best-fit model / residual.
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("show:"))
        self._ext_mode = QComboBox()
        self._ext_mode.addItems(["data", "on model grid", "best-fit model", "residual"])
        self._ext_mode.setToolTip(
            "data = raw file; on model grid = resampled (checks alignment);\n"
            "best-fit model / residual = available after a fit")
        self._ext_mode.currentTextChanged.connect(self._schedule)
        mode_row.addWidget(self._ext_mode, 1)
        ext_lay.addLayout(mode_row)

        self._ext_label = QLabel("no file loaded")
        self._ext_label.setWordWrap(True)
        self._ext_label.setObjectName("extStatus")
        ext_lay.addWidget(self._ext_label)

        # ------------------------------------------ display row (settings | data)
        # Display settings and the external-image file buttons share one row but
        # are visually separated so they read as two distinct groups.
        self.display_bar = DisplayBar()
        self.data_bar = DataBar()
        self.data_bar.loadImageRequested.connect(self._load_external_image)
        self.data_bar.clearRequested.connect(self._clear_external_image)
        self.data_bar.loadAuxRequested.connect(self._load_aux)

        display_row = QWidget()
        dr = QHBoxLayout(display_row)
        dr.setContentsMargins(0, 0, 0, 0)
        dr.setSpacing(6)
        dr.addWidget(self.display_bar, 0)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setObjectName("vsep")
        sep.setToolTip("external image data")
        dr.addWidget(sep)

        data_label = QLabel("<b>data:</b>")
        data_label.setObjectName("dataLabel")
        dr.addWidget(data_label, 0)
        dr.addWidget(self.data_bar, 0)
        dr.addStretch(1)

        # The row has a wide natural minimum (display settings + data buttons), so
        # it scrolls horizontally instead of clipping on a narrow window.
        self.display_row = QScrollArea()
        self.display_row.setWidgetResizable(True)
        self.display_row.setWidget(display_row)
        self.display_row.setFrameShape(QFrame.NoFrame)
        self.display_row.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.display_row.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._display_row_h = display_row.sizeHint().height() + 16
        self.display_row.setFixedHeight(self._display_row_h)

        # Fitting strip, under the display row (kept in the same left column).
        self.fit_bar = FitBar()
        self.fit_bar.fitRequested.connect(self._start_fit)
        self.fit_bar.cancelRequested.connect(self._cancel_fit)
        self.fit_bar.saveResultRequested.connect(self._save_fit_report)
        self.fit_bar.saveChainRequested.connect(self._save_fit_chain)
        # The strip's natural width (~1136 px) would otherwise pin the whole left
        # column, blocking the 3D/external splitter from shrinking it. Like the
        # display row, wrap it in a scroll area so it compresses and scrolls when
        # the column narrows.
        _fit_scroll = QScrollArea()
        _fit_scroll.setWidgetResizable(True)
        _fit_scroll.setWidget(self.fit_bar)
        _fit_scroll.setFrameShape(QFrame.NoFrame)
        _fit_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        _fit_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._fit_scroll_h = self.fit_bar.sizeHint().height() + 16
        _fit_scroll.setFixedHeight(self._fit_scroll_h)

        # Top area: a left column (3D bar + numPix/display row + fitting strip)
        # side by side with the external panel — the panel therefore spans ALL of
        # the left column's rows, a tall strip on the right of the window.
        left_col = QWidget()
        lc = QVBoxLayout(left_col)
        lc.setContentsMargins(0, 0, 0, 0)
        lc.setSpacing(0)
        lc.addWidget(self._3d_wrap)
        lc.addWidget(self.display_row)
        lc.addWidget(_fit_scroll)
        # Extra height (e.g. the user dragging the row splitter to enlarge the
        # external panel) pools *below* the compact content instead of stretching
        # gaps between the 3D bar / display row / fit strip.
        lc.addStretch(1)

        # A horizontal splitter between the 3D/left column and the external
        # panel: drag the divider to trade width between the 3D bar and the
        # "External image / fit data" panel.  The 3D bar keeps its fixed height
        # (it is grid rows, not width, that a 3D scene depends on).
        self.top_split = QSplitter(Qt.Horizontal)
        self.top_split.setChildrenCollapsible(False)
        self.top_split.addWidget(left_col)
        self.top_split.addWidget(self.ext_panel)
        # Seed widths: favour the 3D/left column; the user can then drag.
        self.top_split.setSizes([850, 400])

        top_block = QWidget()
        tb = QHBoxLayout(top_block)
        tb.setContentsMargins(0, 0, 0, 0)
        tb.setSpacing(0)
        tb.addWidget(self.top_split, 1)

        # ------------------------------------------------------------ 2D grid (resizable)
        # The three columns live in a horizontal QSplitter (drag to resize), and
        # the whole grid vs the top area live in a vertical QSplitter (drag the
        # row heights).  The view columns hold *square* sky fields, so the whole
        # grid sits in a centred, width-capped band to avoid wide empty gutters.
        self.fermat_canvas = FieldCanvas("Fermat potential (relative)")
        self.image_canvas = ImageCanvas()
        self.delay_canvas = FieldCanvas("Time delay (relative)")
        self.curves_canvas = CurvesCanvas()

        # Config columns
        self.lenses_panel = LensesPanel()
        self.sources_panel = SourcesPanel()
        self.point_sources_panel = PointSourcesPanel()

        # Column 0 pane: Fermat potential | Time delay
        pane_view0 = QWidget()
        v0 = QVBoxLayout(pane_view0)
        v0.setContentsMargins(4, 2, 4, 4)
        v0.setSpacing(4)
        v0.addWidget(self.fermat_canvas, 1)
        v0.addWidget(self.delay_canvas, 1)

        # Column 1 pane: Lens image | Critical curve + caustic
        pane_view1 = QWidget()
        v1 = QVBoxLayout(pane_view1)
        v1.setContentsMargins(4, 2, 4, 4)
        v1.setSpacing(4)
        v1.addWidget(self.image_canvas, 1)
        v1.addWidget(self.curves_canvas, 1)

        # Column 2 pane: Lenses | Sources | Point sources configuration
        pane_config = QWidget()
        vc = QVBoxLayout(pane_config)
        vc.setContentsMargins(4, 2, 4, 4)
        vc.setSpacing(4)
        # The three config panels share the column; a vertical splitter lets the
        # user resize lenses / sources / point sources independently.
        self.cfg_split = QSplitter(Qt.Vertical)
        self.cfg_split.setChildrenCollapsible(False)
        self.cfg_split.addWidget(self.lenses_panel)
        self.cfg_split.addWidget(self.sources_panel)
        self.cfg_split.addWidget(self.point_sources_panel)
        self.cfg_split.setStretchFactor(0, 1)
        self.cfg_split.setStretchFactor(1, 1)
        self.cfg_split.setStretchFactor(2, 1)
        vc.addWidget(self.cfg_split, 1)

        self.col_split = QSplitter(Qt.Horizontal)
        self.col_split.setChildrenCollapsible(False)
        self.col_split.addWidget(pane_view0)
        self.col_split.addWidget(pane_view1)
        self.col_split.addWidget(pane_config)
        self.col_split.setStretchFactor(0, 3)
        self.col_split.setStretchFactor(1, 3)
        self.col_split.setStretchFactor(2, 4)

        self._grid_host = QWidget()
        gh = QVBoxLayout(self._grid_host)
        gh.setContentsMargins(0, 0, 0, 0)
        gh.addWidget(self.col_split)
        self._update_grid_width(self.width())

        grid_band = QHBoxLayout()
        grid_band.setContentsMargins(0, 0, 0, 0)
        grid_band.addWidget(self._grid_host, 1)   # fill the full window width

        grid_block = QWidget()
        gb = QVBoxLayout(grid_block)
        gb.setContentsMargins(0, 0, 0, 0)
        gb.addLayout(grid_band, 1)

        # Vertical splitter: the whole top block vs the 2D grid.
        self.row_split = QSplitter(Qt.Vertical)
        self.row_split.setChildrenCollapsible(False)
        self.row_split.addWidget(top_block)
        self.row_split.addWidget(grid_block)
        self.row_split.setStretchFactor(0, 0)
        self.row_split.setStretchFactor(1, 1)
        # Default split (same mechanism as the column splitter): keep the top
        # block at its content height so the 2D grid gets the taller share.  The
        # external panel's canvas hint would otherwise inflate the top block.
        # Sizes are seeded from size hints (measurable before any layout), not
        # from live widget heights, which are unreliable mid-layout.
        top_content = int(self._3d_height + self._display_row_h
                          + self.fit_bar.sizeHint().height())
        self.row_split.setSizes([top_content, max(400, self.height() - top_content)])
        root.addWidget(self.row_split, 1)

        # ---------------------------------------------------- controller wiring
        # The controller owns program state; this presenter only subscribes.
        self.controller.fitRunningChanged.connect(self.fit_bar.set_running)
        self.controller.fitStatus.connect(self.fit_bar.set_status)
        self.controller.fitPreview.connect(self._fit_preview)
        self.controller.fitFinished.connect(self._fit_finished)
        self.controller.fitFailed.connect(self._fit_failed)
        self.controller.statusMessage.connect(self.statusBar().showMessage)

        # Throttled redraw.
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._rerender)

        self.lenses_panel.changed.connect(self._schedule)
        self.sources_panel.changed.connect(self._schedule)
        self.point_sources_panel.changed.connect(self._schedule)
        # Two-way point-source link: the Sources card "point" checkbox and the
        # Point sources panel stay in sync (per-source AGN convenience).
        self.sources_panel.point_source_toggled.connect(
            self.point_sources_panel.set_attached)
        self.point_sources_panel.attached_changed.connect(
            self.sources_panel.set_point_checked)
        self.display_bar.changed.connect(self._schedule)
        # Toggling 3D shows/hides the bar and re-renders.
        self.display_bar._three_d.toggled.connect(self._on_3d_toggled)

        self._last_display = self.display_bar.display()
        self._fit_preview_count = 0
        self._render_ok = False
        self._rerender()

    # ------------------------------------------------------ state delegation
    # The window no longer *stores* program state: it forwards reads/writes to
    # the controller, so the two halves share no mutable state directly.
    @property
    def _external_array(self):
        # The external panel shows the loaded image exactly as-is.  (A loaded
        # noise/sigma map is not applied here — noise decorates the "Lens image"
        # model panel and weights the fit's chi², nothing else.)
        return self.controller.external_array

    @_external_array.setter
    def _external_array(self, value):
        self.controller.external_array = value

    @property
    def _noise_array(self):
        return self.controller.noise_array

    @_noise_array.setter
    def _noise_array(self, value):
        self.controller.noise_array = value

    @property
    def _mask_array(self):
        return self.controller.mask_array

    @_mask_array.setter
    def _mask_array(self, value):
        self.controller.mask_array = value

    @property
    def _psf_kernel(self):
        return self.controller.psf_kernel

    @_psf_kernel.setter
    def _psf_kernel(self, value):
        self.controller.psf_kernel = value

    @property
    def _fit_data(self):
        return self.controller.fit_data

    @_fit_data.setter
    def _fit_data(self, value):
        self.controller.fit_data = value

    @property
    def _fit_result(self):
        return self.controller.fit_result

    @_fit_result.setter
    def _fit_result(self, value):
        self.controller.fit_result = value

    @property
    def _fit_worker(self):
        return self.controller._fit_worker

    @property
    def _ext_desc(self):
        return self.controller.ext_desc

    @_ext_desc.setter
    def _ext_desc(self, value):
        self.controller.ext_desc = value

    @property
    def _center_offset(self):
        return self.controller.center_offset

    @_center_offset.setter
    def _center_offset(self, value):
        self.controller.center_offset = value

    def _update_grid_width(self, window_w: int):
        """Seed the column splitter once with the 4:3:3 ratio (config : view1 :
        view2).  The lower half spans the full window width — no centred cap,
        so there are no empty gutters on wide windows.  The stretch factors keep
        the ratio on resize; the user can still drag the splitter.
        """
        if not getattr(self, "_col_split_seeded", False):
            self._col_split_seeded = True
            self.col_split.setSizes(
                [int(window_w * 3 / 10), int(window_w * 3 / 10),
                 int(window_w * 4 / 10)]
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_grid_width(self.width())

    def _on_3d_toggled(self, enabled: bool):
        # When off we hide only the rendering canvas and keep the black area, so
        # the layout does not reflow; the expensive rebuild is skipped too.
        if self._3d_native is not None:
            self._3d_native.setVisible(bool(enabled))
        if enabled:
            self._schedule()
        self.statusBar().showMessage(
            "3D scene on" if enabled else "3D scene off (black background)", 2000
        )

    # ------------------------------------------------------- external image
    def _load_external_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load a lensed image / matrix",
            "",
            "All supported (*.npy *.npz *.fits *.fit *.fts *.mat *.txt *.csv "
            "*.dat *.tsv *.png *.jpg *.jpeg *.tif *.tiff *.bmp);;"
            "NumPy (*.npy *.npz);;FITS (*.fits *.fit *.fts);;"
            "MATLAB (*.mat);;Text (*.txt *.csv *.dat *.tsv);;"
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*)",
        )
        if not path:
            return
        self.load_external_image_file(path)

    def load_external_image_file(self, path: str) -> bool:
        """Load ``path`` into the external panel. Returns True on success."""
        from .external_image import ImageLoadError

        try:
            array, desc = self.controller.load_external(path)
        except ImageLoadError as exc:
            self.external_canvas.show_message(f"could not load:\n{exc}")
            self._ext_label.setText(f"error: {exc}")
            self.statusBar().showMessage(f"external image error: {exc}", 6000)
            return False

        display = self.display_bar.display()
        self.external_canvas.update_external(
            array,
            title=f"External image {array.shape[0]}x{array.shape[1]}",
            colormap=display["colormap"],
            stretch=display["stretch"],
        )
        self._ext_label.setText(desc)
        self.statusBar().showMessage(f"loaded {desc}", 4000)
        return True

    def _clear_external_image(self):
        self.controller.clear_external()
        self.external_canvas.show_message("no file loaded\n\nUse “Load image…” below")
        self._ext_label.setText("no file loaded")

    # ------------------------------------------------- noise / mask / PSF
    def _load_aux(self, kind: str):
        """Load the optional noise, mask or PSF file."""
        from .external_image import ImageLoadError

        path, _ = QFileDialog.getOpenFileName(
            self, f"Load {kind} file", "",
            "All supported (*.npy *.npz *.fits *.fit *.fts *.mat *.txt *.csv "
            "*.dat *.tsv *.png *.jpg *.tif *.bmp);;All files (*)",
        )
        if not path:
            return
        try:
            _, desc = self.controller.load_aux(kind, path)
        except ImageLoadError as exc:
            self._ext_label.setText(f"{kind} error: {exc}")
            self.statusBar().showMessage(f"{kind} load error: {exc}", 6000)
            return
        self.statusBar().showMessage(f"loaded {kind}: {desc}", 4000)
        self._schedule()

    # ------------------------------------------------- prepared fit data
    def prepare_fit_data(self):
        """Build the resampled data bundle on the model grid, or None."""
        from .fit_data import DataPrepError

        try:
            return self.controller.prepare_fit_data(self.display_bar.display())
        except DataPrepError as exc:
            self._ext_label.setText(f"prepare error: {exc}")
            self.statusBar().showMessage(f"fit-data error: {exc}", 6000)
            return None

    def _update_external_view(self, result, display):
        """Draw the external panel according to its mode selector."""
        mode = self._ext_mode.currentText()
        cmap, stretch = display["colormap"], display["stretch"]

        if mode == "best-fit model" and self._fit_result is not None:
            arr = self._fit_result.model
            self.external_canvas.update_external(
                arr, title="Best-fit model", colormap=cmap, stretch=stretch)
            r = self._fit_result
            self._ext_label.setText(
                f"best fit: \u03c7\u00b2 {r.chi2_before:.4g} \u2192 {r.chi2_after:.4g}"
                f"  reduced \u03c7\u00b2_\u03bd {r.reduced_chi2:.4g}"
                f"  ({r.n_free} free, ndof {r.ndof})")
            return

        if mode == "residual" and self._fit_result is not None:
            model = self._fit_result.model
            resid = np.asarray(self._external_array) - model \
                if model.shape == np.asarray(self._external_array).shape \
                else self._fit_result.residual
            self.external_canvas.update_external(
                resid, title="Residual (data \u2212 model)", colormap=cmap,
                stretch="linear")
            rms = float(np.sqrt(np.mean(np.asarray(resid) ** 2)))
            self._ext_label.setText(f"residual rms = {rms:.4g}")
            return

        if mode == "on model grid":
            fd = self.prepare_fit_data()
            if fd is not None:
                self.external_canvas.update_external(
                    fd.image, title=f"On model grid {fd.num_pix}x{fd.num_pix}",
                    colormap=cmap, stretch=stretch)
                self._ext_label.setText(fd.summary())
            return

        self.external_canvas.update_external(
            self._external_array,
            title=f"External image {self._external_array.shape[0]}"
                  f"x{self._external_array.shape[1]}",
            colormap=cmap, stretch=stretch)
        self._ext_label.setText(getattr(self, "_ext_desc", "no file loaded"))

    # ------------------------------------------------------------ fitting
    def _start_fit(self):
        """Launch a PSO fit in a background thread (via the controller)."""
        if self._fit_worker is not None and self._fit_worker.isRunning():
            return

        data = self.prepare_fit_data()
        if data is None:
            self.fit_bar.set_status("load an image first")
            return
        if data.noise is None:
            self.fit_bar.set_status("load a noise map (chi-squared needs one)")
            return

        config = self._build_config()
        settings = self.fit_bar.settings()
        self._fit_preview_count = 0
        self.fit_bar.set_status("fitting…")

        self.controller.start_fit(
            config, data, self.lenses_panel.param_specs(),
            self.sources_panel.param_specs(),
            self.point_sources_panel.param_specs(), settings,
        )
        # Show the live model in the data panel while the fit runs (only when
        # previews are enabled).
        if self.fit_bar.preview_enabled():
            self._ext_mode.setCurrentText("best-fit model")
        self._fit_preview_count = 0

    def _fit_preview(self, iteration, total, chi2, image):
        """Draw the swarm's current best model so the fit is visible as it runs."""
        self._fit_preview_count = getattr(self, "_fit_preview_count", 0) + 1
        display = self.display_bar.display()
        try:
            self.external_canvas.update_external(
                image,
                title=f"fitting\u2026 iter {iteration}/{total}",
                colormap=display["colormap"], stretch=display["stretch"],
            )
        except Exception:
            pass
        self.fit_bar.set_status(
            f"fitting\u2026 iter {iteration}/{total}   \u03c7\u00b2 {chi2:.4g}"
            f"   (preview {self._fit_preview_count})"
        )

    def _cancel_fit(self):
        self.controller.cancel_fit()
        self.fit_bar.set_status("cancelling… (finishes the current restart)")

    def _fit_failed(self, message: str):
        self.fit_bar.set_status(f"fit failed: {message}")
        self.statusBar().showMessage(f"fit failed: {message}", 6000)

    def _fit_finished(self, result):
        # Write the best-fit values back into the sliders so the whole UI shows
        # the fitted model (locked parameters are not touched: set_value is a
        # no-op while a parameter is fixed).
        self._apply_fitted_config(result.config)

        self.fit_bar.set_status(
            f"\u03c7\u00b2 {result.chi2_before:.4g} \u2192 {result.chi2_after:.4g}"
            f"   ({result.n_free} free, ndof {result.ndof})"
        )
        self.statusBar().showMessage(
            f"fit done: chi2 {result.chi2_before:.4g} -> {result.chi2_after:.4g}", 6000
        )
        # Show the finished model in the data panel and switch it to the fit
        # view (the panel currently shows "fitting…" previews).
        self._ext_mode.setCurrentText("best-fit model")
        self._rerender()
        # Fit succeeded -> the export buttons light up (report PNG + chain).
        self.fit_bar.set_has_result(result.ok)

    def _save_fit_report(self):
        """Save the fit's data | model | residual report image via a dialog."""
        result = self.controller.fit_result
        data = self.controller.fit_data
        if result is None or data is None or not result.ok:
            return
        default = os.path.join(os.getcwd(), "lensmovie_fit_result.png")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save fit result image", default, "PNG image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            from . import export_fit as exp
            exp.save_fit_report_png(path, data, result)
        except Exception as exc:
            QMessageBox.warning(self, "Save failed",
                                f"Could not save the fit result:\n{exc}")
            return
        self.statusBar().showMessage(f"fit result saved: {path}", 6000)

    def _save_fit_chain(self):
        """Save the fitted parameter chain as CSV (+ trajectory figure PNG)."""
        result = self.controller.fit_result
        data = self.controller.fit_data
        if result is None or data is None or not result.ok:
            return
        if not result.chain_iter:
            self.statusBar().showMessage("no parameter chain recorded", 6000)
            return
        default = os.path.join(os.getcwd(), "lensmovie_fit_chain.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save fit parameter chain", default, "CSV file (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        png_path = os.path.splitext(path)[0] + "_trajectories.png"
        try:
            from . import export_fit as exp
            exp.save_chain_csv(path, result)
            exp.save_chain_png(png_path, result)
        except Exception as exc:
            QMessageBox.warning(self, "Save failed",
                                f"Could not save the parameter chain:\n{exc}")
            return
        self.statusBar().showMessage(
            f"parameter chain saved: {path} (+ {png_path})", 6000)
        self._ext_mode.setCurrentText("best-fit model")
        self._rerender()

    def _apply_fitted_config(self, config: lc.Config):
        """Push fitted lens/source values back into the panel sliders."""
        fields = ("theta_E", "gamma1", "gamma2", "e1", "e2", "gamma",
                  "center_x", "center_y", "light_amp", "light_R_sersic",
                  "light_n_sersic", "light_sigma", "light_e1", "light_e2")
        for card, params in zip(self.lenses_panel._cards, config.lenses):
            for name in fields:
                row = card.sliders.get(name)
                if row is not None:
                    row.set_value(getattr(params, name, row.value()))
        for card, params in zip(self.sources_panel._cards, config.sources):
            for name in ("amp", "R_sersic", "sigma", "n_sersic", "e1", "e2",
                         "center_x", "center_y"):
                row = card.sliders.get(name)
                if row is not None:
                    row.set_value(getattr(params, name, row.value()))
        # Point sources: rebuild the cards when the count changed (a fit may add
        # or drop one), otherwise just push the values back onto the sliders.
        fitted = config.point_sources
        if len(fitted) != len(self.point_sources_panel._cards):
            for card in list(self.point_sources_panel._cards):
                self.point_sources_panel._remove_card(card)
            for ps in fitted:
                self.point_sources_panel._add_point(point=ps)
            self.point_sources_panel.update_sources(config.sources)
        else:
            for card, params in zip(self.point_sources_panel._cards, fitted):
                for name in ("source_amp", "point_amp", "center_x", "center_y"):
                    row = card.sliders.get(name)
                    if row is not None:
                        row.set_value(getattr(params, name, row.value()))
                card.set_sources(config.sources, params.ref_source)

    def _schedule(self, *a):
        self._timer.start()

    def _build_config(self) -> lc.Config:
        d = self.display_bar.display()
        sources = self.sources_panel.source_list()
        # Keep the point-source anchor combos in step with the source list
        # (rebuild on count change only; dragging a source needn't rebuild them).
        if len(sources) != len(self.point_sources_panel._sources):
            self.point_sources_panel.update_sources(sources)
        return lc.Config(
            lenses=self.lenses_panel.lens_list(),
            sources=sources,
            point_sources=self.point_sources_panel.point_source_list(),
            num_pix=d["num_pix"],
            delta_pix=d["delta_pix"],
            psf_kernel=self.controller.current_psf_kernel(d),
            sky_amp=d["sky_amp"],
        )

    def current_psf_kernel(self):
        """The convolution kernel for the model: a loaded kernel, else from FWHM."""
        return self.controller.current_psf_kernel(self.display_bar.display())

    def _lens_image_with_noise(self, model: "np.ndarray", num_pix: int,
                               delta: float) -> "np.ndarray":
        """The model image to show in the "Lens image" panel.

        A loaded noise (sigma) map of the same shape as the model grid adds a
        *display-only* Gaussian noise realisation to the model, so the user can
        see what the noisy observation looks like without the noise being part of
        the fit (the fit uses ``external_array`` and the sigma map as the chi²
        weight).  The draw is deterministic (same seed each render) so the panel
        never flickers between re-renders.  With no / mismatched sigma map the
        clean model is returned.
        """
        if self.controller.noise_array is None:
            return model
        sig = np.asarray(self.controller.noise_array, dtype=float)
        if sig.shape != (num_pix, num_pix) or not np.any(sig > 0):
            return model
        rng = np.random.RandomState(7)     # fixed: stable across re-renders
        draw = np.where(sig > 0.0, rng.normal(0.0, 1.0, sig.shape) * sig, 0.0)
        return np.asarray(model, dtype=float) + draw

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
            image_positions=result.image_positions,
        )
        self.image_canvas.update_image(
            self._lens_image_with_noise(result.image, num_pix, delta),
            num_pix, delta, result.image_positions,
            colormap=display["colormap"], stretch=display["stretch"],
        )
        self.delay_canvas.update_field(
            result.time_delay, num_pix, delta,
            colormap=display["colormap"], stretch=display["stretch"],
            image_positions=result.image_positions,
        )
        # Same field of view as the image so the two panels line up in x.  Image
        # positions live in the lens plane (← critical curve side); each source's
        # own centre sits in the source plane (← caustic side, gold star).
        source_positions = [
            (np.array([s.center_x]), np.array([s.center_y]), i)
            for i, s in enumerate(config.sources)
        ]
        self.curves_canvas.update_curves(
            result.cc_ra, result.cc_dec, result.caustic_ra, result.caustic_dec,
            num_pix, delta,
            image_positions=result.image_positions,
            source_positions=source_positions,
        )

        # Keep an already-loaded external matrix in sync with the display
        # settings, honouring the panel's mode selector.
        if self._external_array is not None:
            try:
                self._update_external_view(result, display)
            except Exception:
                pass

        # The 3D scene is only rebuilt when it is actually visible: this is the
        # expensive part (mesh construction + GL upload per update).
        if self.scene3d is not None and self.display_bar.three_d_enabled():
            try:
                self.scene3d.update_scene(config, result)
            except Exception as exc:
                self.statusBar().showMessage(f"3D update error: {exc}", 5000)

        self.statusBar().showMessage(
            f"lenses={len(config.lenses)} sources={len(config.sources)} "
            f"grid={num_pix}²  ref_z={result.ref_z_source:.2f}", 3000
        )
        self._render_ok = True
