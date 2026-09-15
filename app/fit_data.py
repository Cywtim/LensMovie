"""Prepare user-supplied data for fitting a modelled lens image.

The model is rendered on a *model grid* (``num_pix`` pixels of ``delta_pix``
arcsec, centred on the lens). A loaded external image has its own shape and
pixel scale, so it must be resampled/cropped onto that same grid before the two
can be compared. This module does that and bundles the result with the optional
noise, mask and PSF needed for a chi-squared likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class DataPrepError(Exception):
    """Raised when external data cannot be put on the model grid."""


@dataclass
class FitData:
    """Data prepared on the model grid, ready to compare with a model image."""

    image: np.ndarray                     # data on the model grid
    delta_pix: float                      # arcsec / pixel of the grid
    noise: Optional[np.ndarray] = None    # 1-sigma map, same shape as image
    mask: Optional[np.ndarray] = None     # bool; True = pixel used in the fit
    psf_kernel: Optional[np.ndarray] = None
    psf_fwhm: float = 0.0
    source_shape: tuple = ()              # original loaded shape
    source_delta_pix: float = 0.0         # original pixel scale
    resampled: bool = False
    notes: list = field(default_factory=list)

    @property
    def num_pix(self) -> int:
        return int(self.image.shape[0])

    def usable_pixels(self) -> int:
        """Number of pixels that participate in the fit."""
        if self.mask is None:
            return int(self.image.size)
        return int(np.count_nonzero(self.mask))

    def summary(self) -> str:
        bits = [
            f"grid {self.image.shape[0]}x{self.image.shape[1]}",
            f"{self.delta_pix:.3f}\"/px",
        ]
        if self.source_shape:
            bits.append(f"from {self.source_shape[0]}x{self.source_shape[1]}")
        if self.resampled:
            bits.append("resampled")
        if self.noise is not None:
            bits.append("noise" if self.noise.ndim == 2 else "noise(σ)")
        if self.mask is not None:
            bits.append(f"mask {self.usable_pixels()}/{self.image.size} px")
        if self.psf_kernel is not None:
            bits.append(f"PSF {self.psf_kernel.shape[0]}x{self.psf_kernel.shape[1]}")
        elif self.psf_fwhm > 0:
            bits.append(f"PSF FWHM {self.psf_fwhm:.2f}\"")
        else:
            bits.append("PSF delta")
        return "  ".join(bits)


def _resample(arr, src_delta, dst_delta, dst_num, center_offset=(0.0, 0.0), order=1):
    """Resample ``arr`` onto a dst_num x dst_num grid centred on the lens.

    ``center_offset`` is the sky position (arcsec) of the lens centre measured in
    the *source* image, so the resampled grid is centred on the lens.
    """
    from scipy.ndimage import map_coordinates

    ny, nx = arr.shape
    # Pixel of the *source* image that corresponds to the lens centre: the data
    # is assumed centred on sky (0,0), and the lens sits at sky offset
    # ``center_offset`` from it. Rows increase downward, hence the sign flip on y.
    cx = (nx - 1) / 2.0 + center_offset[0] / src_delta
    cy = (ny - 1) / 2.0 - center_offset[1] / src_delta

    # Model grid indices (in arcsec offsets from the lens centre).
    half = (dst_num - 1) / 2.0
    jj, ii = np.meshgrid(np.arange(dst_num), np.arange(dst_num))
    # Model y increases upward; image rows increase downward -> flip y.
    sx = cx + (jj - half) * (dst_delta / src_delta)
    sy = cy - (ii - half) * (dst_delta / src_delta)

    out = map_coordinates(arr.astype(float), [sy.ravel(), sx.ravel()],
                          order=order, mode="constant", cval=0.0)
    return out.reshape(dst_num, dst_num)


def prepare_fit_data(
    image: np.ndarray,
    *,
    source_delta_pix: float,
    model_num_pix: int,
    model_delta_pix: float,
    center_offset=(0.0, 0.0),
    noise: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    psf_kernel: Optional[np.ndarray] = None,
    psf_fwhm: float = 0.0,
) -> FitData:
    """Put a loaded image (and optional noise/mask) onto the model grid.

    Args:
        image: the loaded 2-D data array.
        source_delta_pix: pixel scale of ``image`` in arcsec/pixel.
        model_num_pix: size of the model grid in pixels.
        model_delta_pix: pixel scale of the model grid in arcsec/pixel.
        center_offset: (dx, dy) arcsec of the lens centre within the data; the
            resampled grid is centred there.
        noise: optional 1-sigma map (same shape as ``image``).
        mask: optional boolean map (same shape as ``image``), True = keep.
        psf_kernel: optional convolution kernel (already at model_delta_pix).
        psf_fwhm: Gaussian FWHM in arcsec; used only if no kernel is given.

    Returns:
        FitData on the model grid.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise DataPrepError(f"expected a 2-D image, got shape {image.shape}")
    if source_delta_pix <= 0:
        raise DataPrepError("source pixel scale must be > 0")
    if model_delta_pix <= 0:
        raise DataPrepError("model pixel scale must be > 0")
    if model_num_pix < 2:
        raise DataPrepError("model grid must be at least 2x2")

    notes = []
    same_grid = (
        image.shape == (model_num_pix, model_num_pix)
        and abs(source_delta_pix - model_delta_pix) < 1e-9
        and abs(center_offset[0]) < 1e-9 and abs(center_offset[1]) < 1e-9
    )

    if same_grid:
        grid_image = image.copy()
        notes.append("data already on the model grid (no resampling)")
        resampled = False
    else:
        grid_image = _resample(image, source_delta_pix, model_delta_pix,
                               model_num_pix, center_offset)
        resampled = True
        notes.append(
            f"resampled {image.shape[0]}x{image.shape[1]} @ {source_delta_pix:.3f}\"/px "
            f"-> {model_num_pix}x{model_num_pix} @ {model_delta_pix:.3f}\"/px"
        )

    def _bring(a, name, dtype=None):
        a = np.asarray(a)
        if a.ndim == 3:
            a = a[..., 0]
            notes.append(f"{name}: reduced 3-D input to a single plane")
        if a.shape != image.shape:
            raise DataPrepError(
                f"{name} shape {a.shape} does not match image shape {image.shape}"
            )
        if same_grid:
            out = a.astype(dtype) if dtype is not None else a.copy()
        else:
            out = _resample(a.astype(float), source_delta_pix, model_delta_pix,
                            model_num_pix, center_offset, order=0 if dtype is bool else 1)
            if dtype is bool:
                out = out > 0.5
            elif dtype is not None:
                out = out.astype(dtype)
        return out

    grid_noise = _bring(noise, "noise") if noise is not None else None
    grid_mask = _bring(mask, "mask", dtype=bool) if mask is not None else None

    if grid_noise is not None:
        bad = ~np.isfinite(grid_noise) | (grid_noise <= 0)
        if bad.any():
            notes.append(f"noise: {int(bad.sum())} non-positive/invalid pixel(s) excluded")
            if grid_mask is None:
                grid_mask = np.ones(grid_image.shape, dtype=bool)
            grid_mask[bad] = False

    if psf_kernel is not None:
        psf_kernel = np.asarray(psf_kernel, dtype=float)
        if psf_kernel.ndim != 2 or psf_kernel.size == 0:
            notes.append("PSF kernel ignored (not a non-empty 2-D array)")
            psf_kernel = None
        else:
            s = psf_kernel.sum()
            if s > 0:
                psf_kernel = psf_kernel / s
            notes.append(f"PSF kernel {psf_kernel.shape[0]}x{psf_kernel.shape[1]}")

    return FitData(
        image=grid_image,
        delta_pix=float(model_delta_pix),
        noise=grid_noise,
        mask=grid_mask,
        psf_kernel=psf_kernel,
        psf_fwhm=float(psf_fwhm or 0.0),
        source_shape=tuple(image.shape),
        source_delta_pix=float(source_delta_pix),
        resampled=resampled,
        notes=notes,
    )


def effective_psf_kernel(data: FitData) -> np.ndarray:
    """Return the convolution kernel to use: the loaded one, else from FWHM."""
    if data.psf_kernel is not None:
        return data.psf_kernel
    from .lensing_calc import gaussian_psf_kernel

    return gaussian_psf_kernel(data.psf_fwhm, data.delta_pix)


def chi2(model: np.ndarray, data: FitData) -> float:
    """Chi-squared between a model image and the prepared data (needs noise)."""
    if data.noise is None:
        raise DataPrepError("no noise map: cannot compute a chi-squared")
    model = np.asarray(model, dtype=float)
    if model.shape != data.image.shape:
        raise DataPrepError(
            f"model shape {model.shape} != data shape {data.image.shape}"
        )
    resid = (model - data.image) / data.noise
    if data.mask is not None:
        resid = resid[data.mask]
    return float(np.sum(resid ** 2))
