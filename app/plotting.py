"""Matplotlib canvas embedded in Qt for the 2D image-plane view."""

from __future__ import annotations

from matplotlib import cm
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ImageCanvas(FigureCanvas):
    """A Qt-embedded matplotlib canvas that shows the lensed image.

    The underlying imshow object is kept and only its data is updated on redraw,
    which keeps real-time slider updates fast.
    """

    _COLORMAPS = {
        "viridis": cm.viridis,
        "plasma": cm.plasma,
        "inferno": cm.inferno,
        "magma": cm.magma,
        "gray": cm.gray,
        "turbo": cm.turbo,
    }

    def __init__(self, parent=None):
        self._figure = Figure(figsize=(4, 4), tight_layout=True)
        super().__init__(self._figure)
        self.setParent(parent)

        self._ax = self._figure.add_subplot(111)
        self._ax.set_xlabel("arcsec")
        self._ax.set_ylabel("arcsec")
        self._im = None

    def update_image(self, image: np.ndarray, extent_arcsec: tuple, colormap="viridis", stretch="log"):
        """Render a 2D image with a linear or log stretch.

        Args:
            image: (n, n) float array.
            extent_arcsec: (xmin, xmax, ymin, ymax) in arcsec.
            colormap: key into ``_COLORMAPS``.
            stretch: "log" or "linear".
        """
        data = np.asarray(image, dtype=float)
        if stretch == "log":
            # Avoid log(0); add a small floor set to signal.
            vmin = max(data.max() * 1e-6, np.finfo(float).eps)
            view = np.log10(np.clip(data, vmin, None))
        else:
            view = data.copy()

        if self._im is None:
            self._im = self._ax.imshow(
                view,
                origin="lower",
                extent=extent_arcsec,
                cmap=self._COLORMAPS.get(colormap, cm.viridis),
                interpolation="nearest",
            )
            self._ax.figure.colorbar(self._im, ax=self._ax, fraction=0.046, pad=0.04)
        else:
            self._im.set_data(view)
            self._im.set_cmap(self._COLORMAPS.get(colormap, cm.viridis))
            self._im.set_extent(extent_arcsec)
            self._im.set_clim(vmin=view.min(), vmax=view.max())

        self.draw_idle()

    def clear(self):
        if self._im is not None:
            self._im.set_data(np.zeros_like(self._im.get_array()))
            self.draw_idle()
