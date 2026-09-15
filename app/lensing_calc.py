"""Lenstronomy physics core for LensMovie.

Supports multiple lens planes (each with a model type, parameters and redshift)
and multiple sources (each with position/shape and redshift), modelled with
lenstronomy multi-plane lensing. Provides, for a chosen reference source:

  * the lensed 2D image (all sources summed),
  * the Fermat potential 2D field,
  * the relative time-delay 2D field,
  * the critical curve and caustic,
  * the point-source image positions for each source.

The UI only talks to :func:`compute`, which returns a :class:`SimResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------- models


@dataclass
class LensParams:
    """One lens plane."""

    model: str = "SIS"          # SIS | SPEP | PEMD | SIS_SHEAR ...
    theta_E: float = 1.0
    gamma1: float = 0.0         # external shear (for *_SHEAR variants)
    gamma2: float = 0.0
    center_x: float = 0.0
    center_y: float = 0.0
    redshift: float = 0.5
    # Ellipticity-only fields used by SPEP / PEMD
    e1: float = 0.0
    e2: float = 0.0
    gamma: float = 2.0          # PEMD power-law index


# Extended-source light profiles and the kwargs each one needs. All of these are
# resolved (extended) sources, not point sources.
SOURCE_MODELS = ["SERSIC_ELLIPSE", "SERSIC", "GAUSSIAN_ELLIPSE", "GAUSSIAN"]


@dataclass
class SourceParams:
    """One extended source (Sersic or Gaussian profile)."""

    amp: float = 1.0
    R_sersic: float = 0.1      # Sersic half-light radius (arcsec)
    n_sersic: float = 4.0      # Sersic index
    sigma: float = 0.1         # Gaussian width (arcsec), used by GAUSSIAN*
    e1: float = 0.0
    e2: float = 0.0
    center_x: float = 0.1
    center_y: float = -0.1
    redshift: float = 1.5
    model: str = "SERSIC_ELLIPSE"

    def profile_kwargs(self) -> dict:
        """Build the lenstronomy kwargs dict for this source's light profile."""
        common = {
            "amp": float(self.amp),
            "center_x": float(self.center_x),
            "center_y": float(self.center_y),
        }
        model = self.model if self.model in SOURCE_MODELS else "SERSIC_ELLIPSE"
        if model.startswith("SERSIC"):
            common["R_sersic"] = float(self.R_sersic)
            common["n_sersic"] = float(self.n_sersic)
            if model == "SERSIC_ELLIPSE":
                common["e1"] = float(self.e1)
                common["e2"] = float(self.e2)
        else:  # GAUSSIAN / GAUSSIAN_ELLIPSE
            common["sigma"] = float(self.sigma)
            if model == "GAUSSIAN_ELLIPSE":
                common["e1"] = float(self.e1)
                common["e2"] = float(self.e2)
        return common

    def effective_radius(self) -> float:
        """A representative angular size used for the 3D source blob."""
        if self.model.startswith("SERSIC"):
            return float(max(self.R_sersic, 1e-3))
        return float(max(self.sigma, 1e-3))


@dataclass
class Config:
    """Full user configuration passed to :func:`compute`."""

    lenses: list[LensParams] = field(default_factory=lambda: [LensParams()])
    sources: list[SourceParams] = field(default_factory=lambda: [SourceParams()])
    num_pix: int = 150
    delta_pix: float = 0.05


@dataclass
class SimResult:
    """Everything the UI needs to draw one frame."""

    image: np.ndarray
    fermat: np.ndarray
    time_delay: np.ndarray          # arcsec^2 (or time-like), relative to min
    cc_ra: np.ndarray               # critical curve
    cc_dec: np.ndarray
    caustic_ra: np.ndarray          # caustic
    caustic_dec: np.ndarray
    image_positions: list           # [(x_arr, y_arr, color_idx), ...] per source
    num_pix: int
    delta_pix: float
    ref_z_source: float
    ok: bool = True
    error: str = ""


# ------------------------------------------------------------------- model maps

# Map a friendly lens-model name to the lenstronomy profile name and which
# kwargs it needs (beyond common center/theta_E).
_MODEL_PROFILE = {
    "SIS": ("SIS", ["theta_E"]),
    "SIE": ("SIE", ["theta_E", "e1", "e2"]),   # elliptical isothermal (no gamma)
    "SPEP": ("SPEP", ["theta_E", "gamma", "e1", "e2"]),
    "PEMD": ("PEMD", ["theta_E", "gamma", "e1", "e2"]),
}


def lens_kwargs(lens: LensParams) -> dict:
    """Build a single lenstronomy kwargs dict for one lens profile (no shear)."""
    base = {
        "theta_E": float(lens.theta_E),
        "center_x": float(lens.center_x),
        "center_y": float(lens.center_y),
    }
    if lens.model == "SIS":
        return base
    base["e1"] = float(lens.e1)
    base["e2"] = float(lens.e2)
    if lens.model in ("SPEP", "PEMD"):
        base["gamma"] = float(lens.gamma)
    return base


def expand_lenses(
    lenses: list[LensParams],
) -> tuple[list[str], list[float], list[dict]]:
    """Turn the lens list into (model_list, redshift_list, kwargs_list).

    A non-SIS-with-shear profile is emitted as [profile, SHEAR]; a pure SIS as
    just [SIS]. The SHEAR sits at the same plane redshift as its profile.
    """
    model_list: list[str] = []
    redshift_list: list[float] = []
    kwargs_list: list[dict] = []

    for lens in lenses:
        profile = _MODEL_PROFILE.get(lens.model, _MODEL_PROFILE["SIS"])[0]
        if lens.model.endswith("_SHEAR") or (
            lens.model in _MODEL_PROFILE and abs(lens.gamma1) + abs(lens.gamma2) > 1e-9
        ):
            model_list += [profile, "SHEAR"]
            redshift_list += [lens.redshift, lens.redshift]
            kwargs_list += [
                lens_kwargs(lens),
                {"gamma1": lens.gamma1, "gamma2": lens.gamma2},
            ]
        else:
            model_list.append(profile)
            redshift_list.append(lens.redshift)
            kwargs_list.append(lens_kwargs(lens))
    return model_list, redshift_list, kwargs_list


# ------------------------------------------------------------------ the engine


def _get_lens_model(lenses: list[LensParams], z_source: float):
    """Build (and cache) the multi-plane LensModel for a given z_source."""
    key = (
        tuple((l.model, l.redshift, l.theta_E, l.gamma1, l.gamma2,
               l.center_x, l.center_y, l.e1, l.e2, l.gamma) for l in lenses),
        z_source,
    )
    cache = _lens_model_cache
    if key in cache:
        return cache[key]

    model_list, redshift_list, kwargs_list = expand_lenses(lenses)
    if not model_list:
        model_list, redshift_list = ["SIS"], [0.0]
        kwargs_list = [{"theta_E": 1e-9, "center_x": 0.0, "center_y": 0.0}]

    from lenstronomy.LensModel.lens_model import LensModel

    lm = LensModel(
        lens_model_list=model_list,
        lens_redshift_list=redshift_list,
        z_source=z_source,
        cosmo=None,
        multi_plane=True,
    )
    cache[key] = (lm, kwargs_list)
    return cache[key]


_lens_model_cache: dict[tuple, tuple] = {}


def _fermat_potential(lens_model, kwargs_lens, grid_x, grid_y, ref_x, ref_y):
    """Fermat potential phi = arrival_time - (reference arrival time).

    ``arrival_time`` is array-aware in lenstronomy multi-plane, so we pass the
    whole grid at once (fast) and subtract a scalar reference.
    """
    t = np.asarray(lens_model.arrival_time(grid_x, grid_y, kwargs_lens), dtype=float)
    # arrival_time flattens array input; restore the grid shape.
    if t.shape != grid_x.shape:
        t = t.reshape(grid_x.shape)
    t_ref = float(lens_model.arrival_time(ref_x, ref_y, kwargs_lens))
    return t - t_ref


def compute(config: Config, ref_source_index: int = -1) -> SimResult:
    """Evaluate the current configuration.

    ``ref_source_index`` selects which source defines the source-plane redshift
    used for the 2D fields (Fermat, time delay, critical curve, caustic).
    Negative indexes are allowed (default -1 = last source).
    """
    num_pix = int(config.num_pix)
    delta_pix = float(config.delta_pix)
    # Same convention as lenstronomy's make_grid (see _render_source_image):
    # the sky grid is symmetric about (0, 0) and spans +/- (num_pix/2 - 0.5)*delta.
    half = (num_pix / 2 - 0.5) * delta_pix
    axis = np.linspace(-half, half, num_pix)
    grid_x, grid_y = np.meshgrid(axis, axis)
    extent = (-half, half, -half, half)

    # Reference source for the scalar 2D fields.
    idx = ref_source_index if ref_source_index >= 0 else len(config.sources) + ref_source_index
    idx = max(0, min(len(config.sources) - 1, idx))
    if not config.sources:
        dummy = SourceParams()
        ref_z = dummy.redshift
    else:
        ref_z = config.sources[idx].redshift

    try:
        # Lensed image: sum each source's image at its own z_source.
        image = np.zeros((num_pix, num_pix))
        image_positions = []
        for si, source in enumerate(config.sources):
            lens_model, kwargs_lens = _get_lens_model(config.lenses, source.redshift)
            img = _render_source_image(
                lens_model, kwargs_lens, source, num_pix, delta_pix
            )
            image += img
            try:
                pos = _solve_images(lens_model, kwargs_lens, source, color_idx=si)
                image_positions.append(pos)
            except Exception:
                image_positions.append((np.array([]), np.array([]), si))

        # 2D fields at the reference source redshift.
        lens_model, kwargs_lens = _get_lens_model(config.lenses, ref_z)
        if len(config.sources) > 0:
            ref_src = config.sources[idx]
            ref_x, ref_y = ref_src.center_x, ref_src.center_y
        else:
            ref_x = ref_y = 0.0

        fermat = _fermat_potential(
            lens_model, kwargs_lens, grid_x, grid_y, ref_x, ref_y
        )
        time_delay = fermat - fermat.min()

        cc_ra, cc_dec, caustic_ra, caustic_dec = _critical_curve(
            lens_model, kwargs_lens
        )

        return SimResult(
            image=image,
            fermat=fermat,
            time_delay=time_delay,
            cc_ra=cc_ra,
            cc_dec=cc_dec,
            caustic_ra=caustic_ra,
            caustic_dec=caustic_dec,
            image_positions=image_positions,
            num_pix=num_pix,
            delta_pix=delta_pix,
            ref_z_source=ref_z,
            ok=True,
        )
    except Exception as exc:
        return SimResult(
            image=np.zeros((num_pix, num_pix)),
            fermat=np.zeros((num_pix, num_pix)),
            time_delay=np.zeros((num_pix, num_pix)),
            cc_ra=np.array([]),
            cc_dec=np.array([]),
            caustic_ra=np.array([]),
            caustic_dec=np.array([]),
            image_positions=[],
            num_pix=num_pix,
            delta_pix=delta_pix,
            ref_z_source=ref_z,
            ok=False,
            error=str(exc),
        )


def _valid_source_model(model: str) -> str:
    """Return a lenstronomy light-profile name, falling back to SERSIC_ELLIPSE."""
    return model if model in SOURCE_MODELS else "SERSIC_ELLIPSE"


def _render_source_image(lens_model, kwargs_lens, source, num_pix, delta_pix):
    from lenstronomy.Data.imaging_data import ImageData
    from lenstronomy.Data.psf import PSF
    from lenstronomy.ImSim.image_model import ImageModel
    from lenstronomy.LightModel.light_model import LightModel
    from lenstronomy.Util import util

    # Use lenstronomy's own grid convention so the sky grid is centred on the
    # lens. ``make_grid`` returns coordinates spanning +/- (num_pix/2 - 0.5) *
    # delta_pix, i.e. symmetric about (0, 0); its first values are the correct
    # ra/dec origin of pixel (0, 0). Getting this wrong puts the lens outside
    # the field of view and the image no longer looks lensed.
    x_grid, y_grid = util.make_grid(num_pix, delta_pix)
    kwargs_data = {
        "ra_at_xy_0": x_grid[0],
        "dec_at_xy_0": y_grid[0],
        "transform_pix2angle": np.array([[delta_pix, 0], [0, delta_pix]]),
        "image_data": np.zeros((num_pix, num_pix)),
    }
    data = ImageData(**kwargs_data)
    psf = PSF(psf_type="PIXEL", pixel_size=delta_pix, kernel_point_source=np.array([[1.0]]))
    light_model = LightModel(light_model_list=[_valid_source_model(source.model)])
    kwargs_light = [source.profile_kwargs()]
    image_model = ImageModel(
        data_class=data,
        psf_class=psf,
        lens_model_class=lens_model,
        source_model_class=light_model,
        kwargs_numerics={"supersampling_factor": 3},
    )
    return np.asarray(image_model.image(kwargs_lens, kwargs_light), dtype=float)


def _solve_images(lens_model, kwargs_lens, source, color_idx=0):
    from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

    solver = LensEquationSolver(lens_model)
    x, y = solver.findBrightImage(
        source.center_x, source.center_y, kwargs_lens, numImages=4
    )
    return np.asarray(x), np.asarray(y), color_idx


def _critical_curve(lens_model, kwargs_lens, window=2.5):
    from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

    ext = LensModelExtensions(lens_model)
    ra_c, dec_c, ra_ca, dec_ca = ext.critical_curve_caustics(
        kwargs_lens=kwargs_lens, compute_window=window
    )
    # Handle both single arrays, tuples of branches, and empty results.
    def flat(x):
        if isinstance(x, (list, tuple)):
            return np.concatenate([np.asarray(b, dtype=float).reshape(-1)
                                   for b in x if np.asarray(b).size]) \
                if any(np.asarray(b).size for b in x) else np.array([])
        x = np.asarray(x, dtype=float)
        return x.reshape(-1) if x.size else np.array([])

    return (flat(ra_c), flat(dec_c), flat(ra_ca), flat(dec_ca))
