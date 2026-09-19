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
from matplotlib.ticker import MaxNLocator
from PyQt5.QtWidgets import QSizePolicy

from . import theme


def source_contour(source, n=48):
    """Sampled outline of an extended source on the source plane.

    An ellipse with the source's effective radius and ellipticity (from e1/e2),
    centred at (center_x, center_y) — the shape to compare against the caustic:
    a source straddling the caustic gets strongly magnified / extra images.
    Returns (ra, dec) arrays.
    """
    r = float(source.effective_radius()) * 2.0   # readable blob scale
    r = min(max(r, 0.02), 1.0)
    e = min(float(np.hypot(source.e1, source.e2)), 0.98)
    phi = 0.5 * np.arctan2(source.e2, source.e1)
    a, b = r, r * (1.0 - e)
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    cb, sb = np.cos(phi), np.sin(phi)
    ra = source.center_x + a * cb * ct - b * sb * st
    dec = source.center_y + a * sb * ct + b * cb * st
    return np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)


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

# Initial axes rectangle, identical for every canvas, used the moment a canvas
# is created and as the alignment reference between stacked panels (e.g. Lens
# image above Critical curve + caustic).  From the first resize on, the axes are
# re-placed by ``_fit_axes_to_widget`` so the square sky field fills the cell as
# much as the equal-aspect constraint allows.
_AXES_RECT = [0.155, 0.145, 0.755, 0.775]
_CBAR_RECT = [0.930, 0.145, 0.020, 0.775]

# Pixel margins reserved per canvas for tick labels / axis labels / title, plus
# the horizontal room a colourbar strip needs (only field/image/external have
# one).  These are in *widget pixels*, so the axes adapt to the actual size of
# the cell instead of staying a fixed fraction of it.
_MARGIN = {"left": 46, "right": 14, "top": 22, "bottom": 30}
_COLORBAR_STRIP = 26    # extra pixels on the right when a colourbar is present
_COLORBAR_GAP = 8
_COLORBAR_WIDTH = 14

# The square sky field is drawn at 90% of the largest square that fits, so its
# title, tick labels, axis labels and colourbar always keep a safe margin inside
# the widget — no part of the plot crowds or clips at the cell edge in either
# aspect (wide or tall cells).
_SQUARE_SCALE = 0.9


class _MplCanvas(FigureCanvas):
    """Base matplotlib canvas that expands to fill its layout cell.

    Setting an Ignored/Expanding size policy and a tiny minimum size lets the
    canvas grow or shrink with the window instead of pinning the grid to the
    figure's natural size. ``tight_layout`` is deliberately *not* used: it would
    reflow the axes whenever a colorbar is present, breaking alignment between
    canvases with and without one.

    The sky field is a square in arcsec and is drawn with equal aspect, so a
    wide cell cannot be *stretched*; instead :meth:`_fit_axes_to_widget` sizes
    the axes to ~90% of the largest square that fits the cell and centres it,
    which keeps the plotted square as large as possible while leaving a safe
    margin for the title, labels and colourbar in any cell aspect.
    """

    def __init__(self, parent=None):
        self._figure = Figure(
            figsize=_FIGSIZE, facecolor=theme.CANVAS_STYLE["figure_face"]
        )
        super().__init__(self._figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(80, 80)
        self._ax = self._figure.add_axes(_AXES_RECT)
        self._cax = None
        self._markers = []   # per-source point markers (image / source positions)
        self._fit_axes_to_widget()

    def _set_markers(self, positions, marker="o", ms=5, container=None):
        """Refresh per-source point markers: (x_arr, y_arr, color_idx) triples.

        Empty arrays draw nothing.  Called on every update so stale markers are
        removed; the same colour scheme as the lens-image panel is used.  Pass
        ``container`` to draw several independent marker sets on one canvas
        (e.g. image circles and the source star) without wiping each other.
        """
        target = self._markers if container is None else container
        for m in target:
            try:
                m.remove()
            except Exception:
                pass
        target.clear()
        colors = theme.MARKER_COLORS
        for x, y, ci in positions:
            if len(np.asarray(x)):
                (mk,) = self._ax.plot(
                    x, y, marker, ms=ms, mec="k", mfc=colors[ci % len(colors)]
                )
                target.append(mk)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_axes_to_widget()

    def _fit_axes_to_widget(self):
        """Place the main axes at ~90% of the largest square that fits.

        A 10% shrink (``_SQUARE_SCALE``) guarantees the title, tick labels, axis
        labels and colourbar all keep a margin inside the widget however the
        cell's aspect (x vs y) works out; the square is centred in the usable
        area so the margin is symmetric.

        Alignment invariant: every canvas reserves the *same* right-side room
        (the colourbar strip is always accounted for, even by panels that draw
        no colourbar) and uses the same margins/scaling, so stacked panels (Lens
        image above Critical curve + caustic) get byte-identical axes geometry
        and a given sky coordinate lands on the same canvas x in both.
        """
        w, h = max(int(self.width()), 8), max(int(self.height()), 8)
        m = _MARGIN
        left = m["left"]
        # Reserve the colourbar strip for *every* panel so geometry is identical
        # with or without an actual colourbar (keeps stacked panels aligned).
        right = m["right"] + _COLORBAR_STRIP
        top, bottom = m["top"], m["bottom"]

        avail_w = w - left - right
        avail_h = h - top - bottom
        # 90% of the largest fitting square: a universal breathing margin so
        # title / labels / colourbar never reach the widget edge, in wide or
        # tall cells alike.
        side = max(min(avail_w, avail_h) * _SQUARE_SCALE, 16)

        cx = (left + (w - right)) / 2.0
        cy = (bottom + (h - top)) / 2.0
        sx0 = (cx - side / 2) / w
        sy0 = (cy - side / 2) / h
        sx1 = (cx + side / 2) / w
        sy1 = (cy + side / 2) / h
        self._ax.set_position([sx0, sy0, sx1 - sx0, sy1 - sy0])
        if self._cax is not None:
            # colourbar hugs the square's right edge, same vertical span
            self._cax.set_position(
                [sx1 + _COLORBAR_GAP / w, sy0, _COLORBAR_WIDTH / w, sy1 - sy0]
            )

    def _add_colorbar(self, mappable):
        """Attach a colorbar in its own axes, leaving the main axes geometry
        untouched so panels stay aligned.

        ``_cax`` / ``_cb`` are reset together (e.g. by :meth:`ExternalCanvas.
        show_message`); if only one is stale, the old colourbar axes is dropped
        and a fresh one is built rather than crashing."""
        if self._cax is None or self._cb is None:
            if self._cax is not None:
                try:
                    self._figure.delaxes(self._cax)
                except Exception:
                    pass
            self._cax = self._figure.add_axes(_CBAR_RECT)
            self._cb = self._figure.colorbar(mappable, cax=self._cax)
            self._cb.outline.set_edgecolor(theme.CANVAS_STYLE["spine"])
            self._cb.ax.tick_params(colors=theme.CANVAS_STYLE["dim"], labelsize=7)
            self._fit_axes_to_widget()   # make room for the colourbar
        else:
            self._cb.update_normal(mappable)
        return self._cb

    def _style_axes(self, title):
        s = theme.CANVAS_STYLE
        self._ax.set_title(title, fontsize=9, color=s["text"])
        self._ax.set_xlabel("arcsec", fontsize=8, color=s["dim"])
        self._ax.set_ylabel("arcsec", fontsize=8, color=s["dim"])
        self._ax.set_facecolor(s["axes_face"])
        self._ax.tick_params(colors=s["dim"], labelsize=7)
        for spine in self._ax.spines.values():
            spine.set_color(s["spine"])
        # Ticks adapt to the panel's actual range: a handful of round values
        # *inside* the data limits, instead of the locator rounding the extent
        # up past it (which used to draw ticks at +/-4.5 on a +/-3.725 field).
        # ``prune="both"`` drops the outermost candidates that land outside the
        # visible field, so ticks never float past the image edge.
        self._ax.xaxis.set_major_locator(
            MaxNLocator(nbins=5, min_n_ticks=3, prune="both"))
        self._ax.yaxis.set_major_locator(
            MaxNLocator(nbins=5, min_n_ticks=3, prune="both"))


class FieldCanvas(_MplCanvas):
    """Shows a 2D field with a colorbar; data is updated in place (fast)."""

    def __init__(self, title, parent=None):
        super().__init__(parent=parent)
        self._style_axes(title)
        self._im = None
        self._cb = None

    def update_field(self, field: np.ndarray, num_pix: int, delta_pix: float,
                     colormap="magma", stretch="linear", sym=False,
                     image_positions=()):
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
        self._set_markers(image_positions, marker="o", ms=5)
        self._ax.set_aspect("equal", adjustable="box")
        self.draw_idle()

    def clear(self):
        if self._im is not None:
            self._im.set_data(np.zeros_like(self._im.get_array()))
            self.draw_idle()


class ImageCanvas(_MplCanvas):
    """Lensed image with per-source image-position markers (no curves)."""

    # Marker colours, visible on the dark canvas (from the central theme).
    _IMG_COLORS = theme.MARKER_COLORS

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._style_axes("Lens image")
        self._im = None
        self._markers = []
        self._cc_lines = []
        self._cb = None

    def update_image(self, image, num_pix, delta_pix, image_positions,
                     colormap="magma", stretch="log", critical_curve=()):
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

        # Overlay the critical curve(s) on the image: where it runs is where
        # arcs / rings / multiple images concentrate on the sky.
        for line in self._cc_lines:
            try:
                line.remove()
            except Exception:
                pass
        self._cc_lines = []
        ra, dec = critical_curve if len(critical_curve) == 2 else ((), ())
        ra, dec = np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)
        if ra.size:
            (cc,) = self._ax.plot(ra, dec, lw=1.0, color="cyan", alpha=0.9,
                                  zorder=4)
            self._cc_lines.append(cc)

        self._ax.set_aspect("equal", adjustable="box")
        self.draw_idle()

    def clear(self):
        if self._im is not None:
            self._im.set_data(np.zeros_like(self._im.get_array()))
            self.draw_idle()


class CurvesCanvas(_MplCanvas):
    """Critical curve + caustic plotted on an empty sky grid."""

    # Legend handles for the point markers (patterns must match those drawn).
    _SOURCE_MARKER = "*"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._style_axes("Critical curve + caustic")
        s = theme.CANVAS_STYLE
        self._ax.grid(True, color=s["grid"], alpha=0.5)
        self._img_markers = []     # image-position circles (lens plane)
        self._src_markers = []     # source-position stars (source plane)
        self._outline_lines = []   # per-source extent outlines (source plane)
        (self._cc,) = self._ax.plot([], [], lw=1.6, color="cyan", label="critical curve")
        (self._caustic,) = self._ax.plot([], [], lw=1.6, ls="--", color="red", label="caustic")
        (self._src_marker,) = self._ax.plot(
            [], [], self._SOURCE_MARKER, ms=13, mec="k", mfc="gold", label="source")
        (self._extent_handle,) = self._ax.plot(
            [], [], lw=1.5, color="gold", alpha=0.85, label="source extent")
        (self._img_marker,) = self._ax.plot(
            [], [], "o", ms=5, mec="k", mfc="w", label="image")
        self._ax.legend(loc="upper right", fontsize=7, framealpha=0.6,
                        facecolor=s["legend_face"], edgecolor=s["legend_edge"],
                        labelcolor=s["text"])

    def update_curves(self, cc_ra, cc_dec, caustic_ra, caustic_dec,
                      num_pix=None, delta_pix=None, image_positions=(),
                      source_positions=(), source_outlines=()):
        """Draw the curves **auto-scaled to their own extents**.

        Deliberately *not* the lens-image grid FOV: a critical curve / caustic
        that extends past the model grid would otherwise be clipped at the panel
        edge.  Scaling to the curves (with ~15% padding, symmetric about the
        lens centre) always shows the whole curve.  ``num_pix``/``delta_pix``
        are only used as a fallback when there are no curve points at all.
        Image and source markers are drawn on top and included in the scaling,
        so they can never fall outside the visible window.
        ``source_outlines`` is [(ra, dec, colour_idx), ...]: each extended
        source's own extent ellipse on the source plane, to compare against the
        caustic.
        """
        self._set_curve(self._cc, cc_ra, cc_dec)
        self._set_curve(self._caustic, caustic_ra, caustic_dec)
        self._set_markers(image_positions, marker="o", ms=5,
                          container=self._img_markers)
        # The reference source sits in the *source* plane, next to the caustic.
        self._set_markers(source_positions, marker=self._SOURCE_MARKER, ms=13,
                          container=self._src_markers)
        # Per-source extent ellipses (redrawn each update so stale ones vanish).
        for line in self._outline_lines:
            try:
                line.remove()
            except Exception:
                pass
        self._outline_lines = []
        for ra, dec, _ci in source_outlines:
            self._outline_lines.append(
                self._ax.plot(ra, dec, lw=1.5, color="gold", alpha=0.85,
                              zorder=3)[0]
            )

        coords = [np.asarray(a, dtype=float)
                  for a in (cc_ra, cc_dec, caustic_ra, caustic_dec)]
        for x, y, _ci in list(image_positions) + list(source_positions):
            xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            if xa.size:
                coords += [xa, ya]
        for ra, dec, _ci in source_outlines:
            coords += [np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)]
        pts = np.concatenate([c for c in coords if c.size]) \
            if any(c.size for c in coords) else None
        if pts is not None and pts.size:
            r = float(np.abs(pts).max())
            m = r * 1.15 if r > 0 else 1.5
            lo, hi = -m, m
        elif num_pix and delta_pix:
            # no data at all: fall back to the model grid's FOV
            h = _extent(num_pix, delta_pix)[1]
            lo, hi = -h, h
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
            if self._cax is not None:
                try:
                    self._figure.delaxes(self._cax)
                except Exception:
                    pass
            self._cax = None
            self._cb = None
            self._ax.clear()
            self._style_axes("External image")
            self._ax.set_xlabel("pixel x", fontsize=8)
            self._ax.set_ylabel("pixel y", fontsize=8)
        self._ax.text(0.5, 0.5, text, ha="center", va="center",
                      transform=self._ax.transAxes, fontsize=9,
                      color=theme.CANVAS_STYLE["dim"])
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

