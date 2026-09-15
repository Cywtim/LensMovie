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

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import lensing_calc as lc
from .controls import DisplayBar, FitBar, LensesPanel, SourcesPanel
from .fitting import FitError
from .plotting import CurvesCanvas, ExternalCanvas, FieldCanvas, ImageCanvas


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LensMovie — Interactive Lensing Viewer")
        self.resize(1500, 1050)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ------------------------------------------------- top row: 3D bar + external image
        # The 3D scene stretches across the row; a fixed square panel on the right
        # displays a user-supplied matrix. Both keep the same fixed height.
        self._3d_height = 280
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)

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
        top_row.addWidget(self._3d_wrap, 1)

        # Square external-image panel, same height as the 3D bar.
        self.ext_panel = QGroupBox("External image / fit data")
        self.ext_panel.setFixedSize(self._3d_height, self._3d_height)
        ext_lay = QVBoxLayout(self.ext_panel)
        ext_lay.setContentsMargins(4, 4, 4, 4)
        self.external_canvas = ExternalCanvas()
        ext_lay.addWidget(self.external_canvas, 1)
        btn_row = QHBoxLayout()
        self._load_btn = QPushButton("Load image…")
        self._load_btn.clicked.connect(self._load_external_image)
        self._clear_ext_btn = QPushButton("Clear")
        self._clear_ext_btn.clicked.connect(self._clear_external_image)
        btn_row.addWidget(self._load_btn, 1)
        btn_row.addWidget(self._clear_ext_btn, 0)
        ext_lay.addLayout(btn_row)

        # Optional fitting inputs: noise (1-sigma), mask, PSF kernel.
        aux_row = QHBoxLayout()
        self._noise_btn = QPushButton("Noise…")
        self._noise_btn.clicked.connect(lambda: self._load_aux("noise"))
        self._mask_btn = QPushButton("Mask…")
        self._mask_btn.clicked.connect(lambda: self._load_aux("mask"))
        self._psf_btn = QPushButton("PSF…")
        self._psf_btn.clicked.connect(lambda: self._load_aux("psf"))
        for b in (self._noise_btn, self._mask_btn, self._psf_btn):
            b.setToolTip("Load a same-shape file (npy/fits/…); PSF is a kernel")
            aux_row.addWidget(b, 1)
        ext_lay.addLayout(aux_row)

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
        self._ext_label.setStyleSheet("color: gray; font-size: 10px;")
        ext_lay.addWidget(self._ext_label)
        top_row.addWidget(self.ext_panel, 0)

        root.addLayout(top_row)

        # ----------------------------------------------------------------- display bar
        self.display_bar = DisplayBar()
        root.addWidget(self.display_bar)

        # Fitting strip.
        self.fit_bar = FitBar()
        self.fit_bar.fitRequested.connect(self._start_fit)
        self.fit_bar.cancelRequested.connect(self._cancel_fit)
        root.addWidget(self.fit_bar)

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
        # Toggling 3D shows/hides the bar and re-renders.
        self.display_bar._three_d.toggled.connect(self._on_3d_toggled)

        self._last_display = self.display_bar.display()
        self._external_array = None
        self._noise_array = None
        self._mask_array = None
        self._psf_kernel = None
        self._fit_data = None
        self._fit_result = None
        self._ext_desc = "no file loaded"
        self._fit_worker = None
        self._center_offset = (0.0, 0.0)
        self._render_ok = False
        self._rerender()

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
        from .external_image import ImageLoadError, load_image_file

        try:
            array, desc = load_image_file(path)
        except ImageLoadError as exc:
            self.external_canvas.show_message(f"could not load:\n{exc}")
            self._ext_label.setText(f"error: {exc}")
            self.statusBar().showMessage(f"external image error: {exc}", 6000)
            return False

        self._external_array = array
        self._ext_desc = desc
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
        self._external_array = None
        self._ext_desc = "no file loaded"
        self._fit_data = None
        self.external_canvas.show_message("no file loaded\n\nUse “Load image…” below")
        self._ext_label.setText("no file loaded")

    # ------------------------------------------------- noise / mask / PSF
    def _load_aux(self, kind: str):
        """Load the optional noise, mask or PSF file."""
        from .external_image import ImageLoadError, load_image_file

        path, _ = QFileDialog.getOpenFileName(
            self, f"Load {kind} file", "",
            "All supported (*.npy *.npz *.fits *.fit *.fts *.mat *.txt *.csv "
            "*.dat *.tsv *.png *.jpg *.tif *.bmp);;All files (*)",
        )
        if not path:
            return
        try:
            array, desc = load_image_file(path)
        except ImageLoadError as exc:
            self._ext_label.setText(f"{kind} error: {exc}")
            self.statusBar().showMessage(f"{kind} load error: {exc}", 6000)
            return

        if kind == "noise":
            self._noise_array = array
        elif kind == "mask":
            self._mask_array = (array != 0)
        else:
            # Normalise the PSF kernel so it integrates to 1.
            k = np.asarray(array, dtype=float)
            if k.ndim == 2 and k.size and k.sum() > 0:
                k = k / k.sum()
            self._psf_kernel = k
        self.statusBar().showMessage(f"loaded {kind}: {desc}", 4000)
        self._schedule()

    # ------------------------------------------------- prepared fit data
    def prepare_fit_data(self):
        """Build the resampled data bundle on the model grid, or None."""
        from .fit_data import DataPrepError, prepare_fit_data

        if self._external_array is None:
            self._fit_data = None
            return None
        d = self.display_bar.display()
        try:
            self._fit_data = prepare_fit_data(
                self._external_array,
                source_delta_pix=d["delta_pix"],   # data scale set by the same control
                model_num_pix=d["num_pix"],
                model_delta_pix=d["delta_pix"],
                center_offset=self._center_offset,
                noise=self._noise_array,
                mask=self._mask_array,
                psf_kernel=self._psf_kernel,
                psf_fwhm=d["psf_fwhm"],
            )
        except DataPrepError as exc:
            self._fit_data = None
            self._ext_label.setText(f"prepare error: {exc}")
            self.statusBar().showMessage(f"fit-data error: {exc}", 6000)
            return None
        return self._fit_data

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
        """Launch a PSO fit in a background thread."""
        from .fit_worker import FitWorker

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
        self.fit_bar.set_running(True)
        self._fit_preview_count = 0
        self.fit_bar.set_status("fitting…")

        # Lens light parameters live on the same cards as the lens itself.
        lens_specs = self.lenses_panel.param_specs()
        self._fit_worker = FitWorker(
            config, data, lens_specs, lens_specs,
            self.sources_panel.param_specs(), parent=self, **settings,
        )
        self._fit_worker.progressed.connect(self.fit_bar.set_status)
        self._fit_worker.previewed.connect(self._fit_preview)
        self._fit_worker.finished_ok.connect(self._fit_finished)
        self._fit_worker.failed.connect(self._fit_failed)
        # Show the live model in the data panel while the fit runs (only when
        # previews are enabled).
        if self.fit_bar.preview_enabled():
            self._ext_mode.setCurrentText("best-fit model")
        self._fit_preview_count = 0
        self._fit_worker.start()

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
        if self._fit_worker is not None:
            self._fit_worker.cancel()
            self.fit_bar.set_status("cancelling… (finishes the current restart)")

    def _fit_failed(self, message: str):
        self.fit_bar.set_running(False)
        self.fit_bar.set_status(f"fit failed: {message}")
        self.statusBar().showMessage(f"fit failed: {message}", 6000)

    def _fit_finished(self, result):
        self.fit_bar.set_running(False)
        self._fit_result = result

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

    def _schedule(self, *a):
        self._timer.start()

    def _build_config(self) -> lc.Config:
        d = self.display_bar.display()
        return lc.Config(
            lenses=self.lenses_panel.lens_list(),
            sources=self.sources_panel.source_list(),
            num_pix=d["num_pix"],
            delta_pix=d["delta_pix"],
            psf_kernel=self.current_psf_kernel(),
            sky_amp=d["sky_amp"],
        )

    def current_psf_kernel(self):
        """The convolution kernel for the model: a loaded kernel, else from FWHM."""
        if self._psf_kernel is not None:
            return self._psf_kernel
        d = self.display_bar.display()
        return lc.gaussian_psf_kernel(d["psf_fwhm"], d["delta_pix"])

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
        # Same field of view as the image so the two panels line up in x.
        self.curves_canvas.update_curves(
            result.cc_ra, result.cc_dec, result.caustic_ra, result.caustic_dec,
            num_pix, delta,
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
