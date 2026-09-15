"""Matplotlib canvases embedded in Qt for the 2D views.

Four functions of the lensing config are shown:
  * ``ImageCanvas``  — the lensed image with critical curve + caustic + images.
  * ``FieldCanvas``  — a generic 2D field (Fermat potential, time delay).
"""

from __future__ import annotations

import numpy as np
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def _extent(num_pix, delta_pix):
    half = num_pix / 2 * delta_pix
    return (-half, half, -half, half)


_COLORMAPS = {
    "viridis": cm.viridis,
    "plasma": cm.plasma,
    "inferno": cm.inferno,
    "magma": cm.magma,
    "gray": cm.gray,
    "turbo": cm.turbo,
}


class FieldCanvas(FigureCanvas):
    """Shows a 2D field with a colorbar; data is updated in place (fast)."""

    def __init__(self, title, parent=None):
        self._figure = Figure(figsize=(3.4, 3.4), tight_layout=True)
        super().__init__(self._figure)
        self.setParent(parent)
        self._ax = self._figure.add_subplot(111)
        self._ax.set_title(title, fontsize=9)
        self._ax.set_xlabel("arcsec")
        self._ax.set_ylabel("arcsec")
        self._im = None
        self._cb = None

    def update_field(self, field: np.ndarray, num_pix: int, delta_pix: float,
                     colormap="magma", stretch="linear", sym=False):
        data = np.asarray(field, dtype=float)
        if stretch == "log" and data.min() >= 0:
            vmin = max(data.max() * 1e-5, np.finfo(float).eps)
            view = np.log10(np.clip(data, vmin, None))
        else:
            view = data
        bounds = _extent(num_pix, delta_pix)
        if self._im is None:
            vmax = np.abs(view).max() if sym else view.max()
            self._im = self._ax.imshow(
                view, origin="lower", extent=bounds,
                cmap=_COLORMAPS.get(colormap, cm.magma),
                interpolation="nearest", vmin=-vmax if sym else None, vmax=vmax,
            )
            self._cb = self._figure.colorbar(self._im, ax=self._ax, fraction=0.046, pad=0.04)
        else:
            vmax = np.abs(view).max() if sym else view.max()
            self._im.set_data(view)
            self._im.set_extent(bounds)
            self._im.set_cmap(_COLORMAPS.get(colormap, cm.magma))
            self._im.set_clim(-vmax if sym else view.min(), vmax)
        self.draw_idle()

    def clear(self):
        if self._im is not None:
            self._im.set_data(np.zeros_like(self._im.get_array()))
            self.draw_idle()


class ImageCanvas(FigureCanvas):
    """Lensed image with overlaid critical curve, caustic and image positions."""

    # Colours for per-source image-position markers.
    _IMG_COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    def __init__(self, parent=None):
        self._figure = Figure(figsize=(3.6, 3.6), tight_layout=True)
        super().__init__(self._figure)
        self.setParent(parent)
        self._ax = self._figure.add_subplot(111)
        self._ax.set_title("Lensed image", fontsize=9)
        self._ax.set_xlabel("arcsec")
        self._ax.set_ylabel("arcsec")
        self._im = None
        self._cc = None
        self._caustic = None
        self._markers = []
        self._cb = None

    def update_image(self, image, num_pix, delta_pix, cc_ra, cc_dec,
                     caustic_ra, caustic_dec, image_positions, colormap="magma",
                     stretch="log"):
        data = np.asarray(image, dtype=float)
        if stretch == "log" and data.min() >= 0:
            vmin = max(data.max() * 1e-5, np.finfo(float).eps)
            view = np.log10(np.clip(data, vmin, None))
        else:
            view = data
        bounds = _extent(num_pix, delta_pix)

        if self._im is None:
            self._im = self._ax.imshow(view, origin="lower", extent=bounds,
                                       cmap=_COLORMAPS.get(colormap, cm.magma),
                                       interpolation="nearest")
            self._cb = self._figure.colorbar(self._im, ax=self._ax, fraction=0.046, pad=0.04)
            (self._cc,) = self._ax.plot([], [], lw=1.0, color="cyan", label="critical curve")
            (self._caustic,) = self._ax.plot([], [], lw=1.0, ls="--", color="red", label="caustic")
        else:
            self._im.set_data(view)
            self._im.set_extent(bounds)
            self._im.set_cmap(_COLORMAPS.get(colormap, cm.magma))
            self._im.set_clim(view.min(), view.max())

        # Clear previous image-position markers.
        for m in self._markers:
            try:
                m.remove()
            except Exception:
                pass
        self._markers = []
        num = len(image_positions)
        for i, (x, y, _) in enumerate(image_positions):
            if len(x):
                (mk,) = self._ax.plot(x, y, "o", ms=5, mec="k", mfc=self._IMG_COLORS[i % 8])
                self._markers.append(mk)

        self._draw_curve(self._cc, cc_ra, cc_dec)
        self._draw_curve(self._caustic, caustic_ra, caustic_dec)
        self._ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
        self.draw_idle()

    @staticmethod
    def _draw_curve(line, xs, ys):
        if len(xs):
            line.set_data(xs, ys)
            line.set_visible(True)
        else:
            line.set_visible(False)

    def clear(self):
        if self._im is not None:
            self._im.set_data(np.zeros_like(self._im.get_array()))
            self.draw_idle()
