"""Lenstronomy-based 2D image-plane simulation.

This module insulates the UI layer from lenstronomy details. The UI collects
parameters into plain dicts and calls :func:`render` to get back a 2D image array.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Lazy import so the module import itself never requires a display / heavy deps.
_LENS_MODEL = None
_LIGHT_MODEL = None
_IMAGE_DATA = None
_PSF = None
_IMAGE_MODEL = None


def _get_components(num_pix: int, delta_pix: float, lens_model_list, light_model_list):
    """Build (and cache) the lenstronomy component objects for a given grid size.

    The grid-dependent objects (ImageData/PSF) are rebuilt when num_pix/delta_pix
    changes.
    """
    global _LENS_MODEL, _LIGHT_MODEL, _IMAGE_DATA, _PSF, _IMAGE_MODEL

    if _LENS_MODEL is None or _LIGHT_MODEL is None:
        from lenstronomy.LensModel.lens_model import LensModel
        from lenstronomy.LightModel.light_model import LightModel

        _LENS_MODEL = LensModel(
            lens_model_list=lens_model_list, z_lens=0.5, z_source=1.5, cosmo=None
        )
        _LIGHT_MODEL = LightModel(light_model_list=light_model_list)

    if (
        _IMAGE_MODEL is None
        or _IMAGE_DATA is None
        or _PSF is None
        or getattr(_IMAGE_DATA, "_num_pix", None) != num_pix
        or getattr(_IMAGE_DATA, "_delta_pix", None) != delta_pix
    ):
        from lenstronomy.Data.imaging_data import ImageData
        from lenstronomy.Data.psf import PSF
        from lenstronomy.ImSim.image_model import ImageModel

        kwargs_data = {
            "ra_at_xy_0": -num_pix / 2 * delta_pix,
            "dec_at_xy_0": num_pix / 2 * delta_pix,
            "transform_pix2angle": np.array([[1, 0], [0, 1]]) * delta_pix,
            "image_data": np.zeros((num_pix, num_pix)),
        }
        _IMAGE_DATA = ImageData(**kwargs_data)
        _IMAGE_DATA._num_pix = num_pix
        _IMAGE_DATA._delta_pix = delta_pix
        _PSF = PSF(
            psf_type="PIXEL",
            pixel_size=delta_pix,
            kernel_point_source=np.array([[1.0]]),
        )
        _IMAGE_MODEL = ImageModel(
            data_class=_IMAGE_DATA,
            psf_class=_PSF,
            lens_model_class=_LENS_MODEL,
            source_model_class=_LIGHT_MODEL,
            kwargs_numerics={"supersampling_factor": 4},
        )

    return _IMAGE_MODEL


def default_lens_kwargs() -> dict[str, Any]:
    """Default SIS + external shear lens parameters."""
    return {
        "theta_E": 1.0,
        "gamma1": 0.05,
        "gamma2": 0.0,
        "center_x": 0.0,
        "center_y": 0.0,
    }


def default_source_kwargs() -> dict[str, Any]:
    """Default Sersic-ellipse source parameters."""
    return {
        "amp": 1.0,
        "R_sersic": 0.1,
        "n_sersic": 4.0,
        "e1": 0.1,
        "e2": -0.2,
        "center_x": 0.1,
        "center_y": -0.1,
    }


def build_args(lens: dict[str, Any], source: dict[str, Any]):
    """Convert UI dicts into lenstronomy kwargs lists (SIS + SHEAR / SERSIC_ELLIPSE)."""
    kwargs_lens = [
        {
            "theta_E": lens["theta_E"],
            "center_x": lens["center_x"],
            "center_y": lens["center_y"],
        },
        {"gamma1": lens["gamma1"], "gamma2": lens["gamma2"]},
    ]
    kwargs_light = [
        {
            "amp": source["amp"],
            "R_sersic": source["R_sersic"],
            "n_sersic": source.get("n_sersic", 4.0),
            "e1": source["e1"],
            "e2": source["e2"],
            "center_x": source["center_x"],
            "center_y": source["center_y"],
        }
    ]
    return kwargs_lens, kwargs_light


def render(
    lens: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
    num_pix: int = 150,
    delta_pix: float = 0.05,
) -> np.ndarray:
    """Simulate the lensed image plane and return a 2D float array.

    Args:
        lens: lens kwargs dict (see :func:`default_lens_kwargs`).
        source: source kwargs dict (see :func:`default_source_kwargs`).
        num_pix: image grid size in pixels (square).
        delta_pix: pixel scale in arcsec.

    Returns:
        (num_pix, num_pix) float array of the lensed image.
    """
    if lens is None:
        lens = default_lens_kwargs()
    if source is None:
        source = default_source_kwargs()

    kwargs_lens, kwargs_light = build_args(lens, source)
    model = _get_components(num_pix, delta_pix, ["SIS", "SHEAR"], ["SERSIC_ELLIPSE"])
    image = model.image(kwargs_lens, kwargs_light)
    return np.asarray(image, dtype=float)
