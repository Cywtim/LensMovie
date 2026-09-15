"""Matplotlib canvases embedded in Qt for the 2D views.

Grid cells map to canvases as follows:
  * ``FieldCanvas``   — a generic 2D field (Fermat potential, time delay).
  * ``ImageCanvas``   — the lensed image with per-source image positions.
  * ``CurvesCanvas``  — the critical curve + caustic curves only.
"""

from __future__ import annotations

import numpy as np
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QSizePolicy


def _extent(num_pix, delta_pix):
    # Matches lenstronomy's make_grid span so overlays (critical curve, caustic,
    # image positions) line up exactly with the rendered image.
    half = (num_pix / 2 - 0.5) * delta_pix
    return (-half, half, -half, half)


_COLORMAPS = {
    "viridis": cm.viridis,
    "plasma": cm.plasma,
    "inferno": cm.inferno,
    "magma": cm.magma,
    "gray": cm.gray,
    "turbo": cm.turbo,
}

# All canvases share one figure size so every grid cell gets the same size hint,
# which is what keeps the grid aligned.
_FIGSIZE = (3.0, 3.0)

# Fixed, identical axes rectangle for every canvas. The colorbar gets its own
# axes *outside* this rectangle instead of stealing space from the main axes, so
# stacked panels (e.g. Lens image above Critical curve + caustic) line up exactly
# in x even though only some of them have a colorbar.
_AXES_RECT = [0.155, 0.145, 0.755, 0.775]
_CBAR_RECT = [0.930, 0.145, 0.020, 0.775]


class _MplCanvas(FigureCanvas):
    """Base matplotlib canvas that expands to fill its layout cell.

    Setting an Ignored/Expanding size policy and a tiny minimum size lets the
    canvas grow or shrink with the window instead of pinning the grid to the
    figure's natural size. ``tight_layout`` is deliberately *not* used: it would
    reflow the axes whenever a colorbar is present, breaking alignment between
    canvases with and without one.
    """

    def __init__(self, parent=None):
        self._figure = Figure(figsize=_FIGSIZE)
        super().__init__(self._figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(80, 80)
        self._ax = self._figure.add_axes(_AXES_RECT)
        self._cax = None

    def _add_colorbar(self, mappable):
        """Attach a colorbar in its own axes, leaving the main axes geometry
        untouched so panels stay aligned."""
        if self._cax is None:
            self._cax = self._figure.add_axes(_CBAR_RECT)
            self._cb = self._figure.colorbar(mappable, cax=self._cax)
        else:
            self._cb.update_normal(mappable)
        return self._cb

    def _style_axes(self, title):
        self._ax.set_title(title, fontsize=9)
        self._ax.set_xlabel("arcsec", fontsize=8)
        self._ax.set_ylabel("arcsec", fontsize=8)
        self._ax.tick_params(labelsize=7)


class FieldCanvas(_MplCanvas):
    """Shows a 2D field with a colorbar; data is updated in place (fast)."""

    def __init__(self, title, parent=None):
        super().__init__(parent=parent)
        self._style_axes(title)
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
            self._add_colorbar(self._im)
        else:
            vmax = np.abs(view).max() if sym else view.max()
            self._im.set_data(view)
            self._im.set_extent(bounds)
            self._im.set_cmap(_COLORMAPS.get(colormap, cm.magma))
            self._im.set_clim(-vmax if sym else view.min(), vmax)
        self._ax.set_aspect("equal", adjustable="box")
        self.draw_idle()

    def clear(self):
        if self._im is not None:
            self._im.set_data(np.zeros_like(self._im.get_array()))
            self.draw_idle()


class ImageCanvas(_MplCanvas):
    """Lensed image with per-source image-position markers (no curves)."""

    _IMG_COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._style_axes("Lens image")
        self._im = None
        self._markers = []
        self._cb = None

    def update_image(self, image, num_pix, delta_pix, image_positions,
                     colormap="magma", stretch="log"):
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
            self._add_colorbar(self._im)
        else:
            self._im.set_data(view)
            self._im.set_extent(bounds)
            self._im.set_cmap(_COLORMAPS.get(colormap, cm.magma))
            self._im.set_clim(view.min(), view.max())

        # Refresh image-position markers.
        for m in self._markers:
            try:
                m.remove()
            except Exception:
                pass
        self._markers = []
        for i, (x, y, _) in enumerate(image_positions):
            if len(x):
                (mk,) = self._ax.plot(x, y, "o", ms=5, mec="k", mfc=self._IMG_COLORS[i % 8])
                self._markers.append(mk)
        self._ax.set_aspect("equal", adjustable="box")
        self.draw_idle()

    def clear(self):
        if self._im is not None:
            self._im.set_data(np.zeros_like(self._im.get_array()))
            self.draw_idle()


class CurvesCanvas(_MplCanvas):
    """Critical curve + caustic plotted on an empty sky grid."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._style_axes("Critical curve + caustic")
        self._ax.grid(True, alpha=0.3)
        (self._cc,) = self._ax.plot([], [], lw=1.6, color="cyan", label="critical curve")
        (self._caustic,) = self._ax.plot([], [], lw=1.6, ls="--", color="red", label="caustic")
        self._ax.legend(loc="upper right", fontsize=7, framealpha=0.6)

    def update_curves(self, cc_ra, cc_dec, caustic_ra, caustic_dec,
                      num_pix=None, delta_pix=None):
        """Draw the curves using the *same* field of view as the image panel.

        Matching limits (rather than auto-scaling to the curve extents) is what
        makes this panel line up with the Lens image directly above it: the same
        sky coordinate lands at the same place in both.
        """
        self._set_curve(self._cc, cc_ra, cc_dec)
        self._set_curve(self._caustic, caustic_ra, caustic_dec)

        if num_pix and delta_pix:
            lo, hi = _extent(num_pix, delta_pix)[0], _extent(num_pix, delta_pix)[1]
        else:
            # Fall back to the curve extents if no grid was supplied.
            coords = [np.asarray(a, dtype=float)
                      for a in (cc_ra, cc_dec, caustic_ra, caustic_dec)]
            pts = np.concatenate([c for c in coords if c.size]) \
                if any(c.size for c in coords) else None
            if pts is not None and pts.size:
                m = float(np.abs(pts).max()) * 1.2
                lo, hi = -m, m
            else:
                lo, hi = -2.5, 2.5
        self._ax.set_xlim(lo, hi)
        self._ax.set_ylim(lo, hi)
        self._ax.set_aspect("equal", adjustable="box")
        self.draw_idle()

    @staticmethod
    def _set_curve(line, xs, ys):
        if len(xs):
            line.set_data(xs, ys)
            line.set_visible(True)
        else:
            line.set_visible(False)


class ExternalCanvas(_MplCanvas):
    """Displays a user-supplied 2D matrix (npy / fits / image / text ...).

    Axes are pixel indices because an external file carries no pixel scale. The
    array keeps its own aspect ratio, so a non-square matrix is not distorted.
    """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._style_axes("External image")
        self._ax.set_xlabel("pixel x", fontsize=8)
        self._ax.set_ylabel("pixel y", fontsize=8)
        self._im = None
        self._cb = None
        self._show_message("no file loaded\n\nUse “Load image…” below")

    def _show_message(self, text):
        if self._im is not None:
            self._im = None
            self._cb = None
            self._ax.clear()
            self._style_axes("External image")
            self._ax.set_xlabel("pixel x", fontsize=8)
            self._ax.set_ylabel("pixel y", fontsize=8)
        self._ax.text(0.5, 0.5, text, ha="center", va="center",
                      transform=self._ax.transAxes, fontsize=9, color="0.4")
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self.draw_idle()

    def show_message(self, text):
        """Public helper so the window can report load errors in-place."""
        self._show_message(text)

    def update_external(self, array, title="External image", colormap="magma",
                        stretch="log"):
        data = np.asarray(array, dtype=float)
        if stretch == "log":
            # Shift so the smallest value is > 0 before taking a log.
            shifted = data - data.min()
            vmin = max(shifted.max() * 1e-5, np.finfo(float).eps)
            view = np.log10(np.clip(shifted, vmin, None))
        else:
            view = data

        self._ax.clear()
        self._style_axes(title)
        self._ax.set_xlabel("pixel x", fontsize=8)
        self._ax.set_ylabel("pixel y", fontsize=8)
        self._im = self._ax.imshow(
            view, origin="lower",
            cmap=_COLORMAPS.get(colormap, cm.magma), interpolation="nearest",
        )
        self._add_colorbar(self._im)
        self._ax.set_aspect("equal", adjustable="box")
        self.draw_idle()

