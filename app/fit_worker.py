"""Background worker so a PSO fit does not block the GUI.

``run_pso`` is a pure function, so a small :class:`QThread` subclass is enough:
``run`` calls it and emits the result (or an error) back on the GUI thread.
"""

from __future__ import annotations

from PyQt5.QtCore import QThread, pyqtSignal

from . import fitting as ft


class FitWorker(QThread):
    """Runs a PSO fit in a background thread."""

    progressed = pyqtSignal(str)              # human-readable progress line
    previewed = pyqtSignal(int, int, float, object, object)  # iter, total, chi2, sim, config
    finished_ok = pyqtSignal(object)          # FitResult
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()                  # user pressed Cancel

    def __init__(self, config, data, lens_specs, lens_light_specs, source_specs,
                 point_source_specs=None, n_particles=30, n_iterations=100,
                 n_restarts=2, sigma_scale=4.0, polish=True,
                 preview_enabled=True, preview_interval=0.5,
                 early_stop_reduced=0.0, parent=None):
        super().__init__(parent)
        self._args = (config, data, lens_specs, lens_light_specs, source_specs,
                      point_source_specs)
        # When previews are switched off, pass no callback at all so the fit loop
        # does not touch the renderer.
        self._preview_enabled = bool(preview_enabled)
        self._kwargs = dict(
            n_particles=n_particles, n_iterations=n_iterations,
            n_restarts=n_restarts, sigma_scale=sigma_scale, polish=polish,
            preview_interval=float(preview_interval),
            early_stop_reduced=float(early_stop_reduced),
        )
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _emit_preview(self, iteration, total, chi2, sim, config):
        """Called from the fit loop; a preview must never break the fit."""
        if not self._cancelled:
            self.previewed.emit(int(iteration), int(total), float(chi2), sim, config)

    def run(self):
        try:
            result = ft.run_pso(
                *self._args,
                progress=lambda msg: self.progressed.emit(msg),
                preview=self._emit_preview if self._preview_enabled else None,
                is_cancelled=lambda: self._cancelled,
                **self._kwargs,
            )
        except Exception as exc:                      # never kill the thread silently
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        # A cancelled fit must still unwind the UI: emit `cancelled` (never a
        # silent return, which left the strip permanently in the running state).
        if self._cancelled:
            self.cancelled.emit()
            return
        if not result.ok:
            self.failed.emit(result.error or "fit failed")
            return
        self.finished_ok.emit(result)
