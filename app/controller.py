"""LensMovieController — the stateful "program" side of the application.

The window used to own everything (state, widgets, rendering, fitting) in one
550-line class.  This controller extracts the *non-widget* half of that:

  * it owns the mutable program state — the external image, noise / mask / PSF
    arrays, the prepared fit data and the latest fit result,
  * it runs the long-lived background fit worker and re-emits its progress,
  * it exposes a small set of operations (``load_aux``, ``prepare_fit_data``,
    ``start_fit``, ...) plus Qt signals the UI layer subscribes to.

Conventions that keep the two halves decoupled:

  * the controller knows **nothing about widgets** — it never imports
    ``QWidget`` subclasses, never builds a layout, never touches a slider,
  * the UI reads program state *through* this controller and is fed *by* its
    signals; there is no shared mutable state elsewhere,
  * physics stays in ``lensing_calc`` / ``fitting``; this layer only
    coordinates them.

The exchange types are plain dataclasses / dicts (``lensing_calc.Config`` /
``SimResult``, the ``display`` dict), so a presenter can swap freely without
the program noticing — and vice versa, the program can be improved without
touching a single widget as long as these interfaces stay stable.
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from . import lensing_calc as lc


class LensMovieController(QObject):
    """Owns program state and drives long-running work off the UI thread.

    All signals are emitted on the GUI thread (the worker marshals its own
    results), so presenters can touch widgets directly in their slots.
    """

    # -- generic notifications -----------------------------------------------
    statusMessage = pyqtSignal(str, int)   # (message, timeout_ms) -> status bar

    # -- fit lifecycle ---------------------------------------------------------
    fitRunningChanged = pyqtSignal(bool)
    fitStatus = pyqtSignal(str)                            # progress line
    fitPreview = pyqtSignal(int, int, float, object)       # iter, total, chi2, image
    fitFinished = pyqtSignal(object)                       # FitResult
    fitFailed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # --- program state ------------------------------------------------
        self.external_array: np.ndarray | None = None
        self.noise_array: np.ndarray | None = None
        self.mask_array: np.ndarray | None = None
        self.psf_kernel: np.ndarray | None = None
        self.fit_data = None        # FitData (resampled data on the model grid)
        self.fit_result = None      # FitResult from the last finished fit
        self.center_offset: tuple[float, float] = (0.0, 0.0)
        self.ext_desc = "no file loaded"

        self._fit_worker = None     # FitWorker while a fit is running

    # ------------------------------------------------------------ data files
    def load_external(self, path: str):
        """Parse ``path`` and adopt it as the external image.

        Returns ``(array, description)``; raises ``external_image.ImageLoadError``
        on failure (the previous image is left untouched).
        """
        from .external_image import load_image_file

        array, desc = load_image_file(path)
        self.external_array = array
        self.ext_desc = desc
        return array, desc

    def clear_external(self):
        """Forget the external image and anything derived from it."""
        self.external_array = None
        self.fit_data = None
        self.ext_desc = "no file loaded"

    def load_aux(self, kind: str, path: str):
        """Adopt a noise / mask / PSF file.  ``kind`` in (``noise``, ``mask``,
        ``psf``).  Returns ``(array, description)``; a PSF kernel is normalised
        to integrate to 1 so it cannot rescale model brightness.
        """
        from .external_image import load_image_file

        array, desc = load_image_file(path)
        if kind == "noise":
            self.noise_array = array
        elif kind == "mask":
            self.mask_array = np.asarray(array != 0)
        else:  # psf
            k = np.asarray(array, dtype=float)
            if k.ndim == 2 and k.size and k.sum() > 0:
                k = k / k.sum()
            self.psf_kernel = k
        return array, desc

    # ------------------------------------------------------------- psf / data
    def current_psf_kernel(self, display: dict) -> np.ndarray:
        """The convolution kernel: a loaded kernel, else Gaussian from the FWHM."""
        if self.psf_kernel is not None:
            return self.psf_kernel
        return lc.gaussian_psf_kernel(display["psf_fwhm"], display["delta_pix"])

    def prepare_fit_data(self, display: dict):
        """Build the resampled data bundle on the model grid.

        Returns ``None`` when no external image is loaded; raises
        ``fit_data.DataPrepError`` when the data cannot be prepared.
        """
        from .fit_data import DataPrepError, prepare_fit_data

        if self.external_array is None:
            self.fit_data = None
            return None
        try:
            self.fit_data = prepare_fit_data(
                self.external_array,
                source_delta_pix=display["delta_pix"],
                model_num_pix=display["num_pix"],
                model_delta_pix=display["delta_pix"],
                center_offset=self.center_offset,
                noise=self.noise_array,
                mask=self.mask_array,
                psf_kernel=self.psf_kernel,
                psf_fwhm=display["psf_fwhm"],
            )
        except DataPrepError:
            self.fit_data = None
            raise
        return self.fit_data

    # ------------------------------------------------------------------- fit
    @property
    def fit_worker(self):
        """The running :class:`FitWorker`, or ``None``."""
        return self._fit_worker

    def start_fit(self, config: lc.Config, data, lens_specs, source_specs,
                  settings: dict) -> bool:
        """Launch a PSO fit on a background thread.

        ``lens_specs`` / ``source_specs`` are the per-entry
        ``{name: (value, lower, upper, fixed)}`` dicts that describe which
        parameters are free.  ``settings`` comes from the fit strip.  Returns
        ``False`` (and does nothing) if a fit is already running.
        """
        from .fit_worker import FitWorker

        if self._fit_worker is not None and self._fit_worker.isRunning():
            return False

        # Deflector-light specs live on the same cards as the lens itself.
        worker = FitWorker(
            config, data, lens_specs, lens_specs, source_specs,
            parent=self, **settings,
        )
        self._fit_worker = worker
        worker.progressed.connect(self.fitStatus)
        worker.previewed.connect(self.fitPreview)
        worker.finished_ok.connect(self._on_fit_done)
        worker.failed.connect(self._on_fit_failed)
        self.fitRunningChanged.emit(True)
        worker.start()
        return True

    def cancel_fit(self):
        """Ask the running fit to stop after the current restart."""
        if self._fit_worker is not None:
            self._fit_worker.cancel()

    def _on_fit_done(self, result):
        self.fit_result = result
        self.fitFinished.emit(result)
        self.fitRunningChanged.emit(False)

    def _on_fit_failed(self, message: str):
        self.fitFailed.emit(message)
        self.fitRunningChanged.emit(False)
