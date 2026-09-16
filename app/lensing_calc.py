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

    # NFW dark-matter halo: Rs = scale radius (arcsec); alpha_Rs = deflection
    # strength in units of Rs at the scale radius. theta_E is unused by NFW.
    Rs: float = 1.0
    alpha_Rs: float = 0.5
    # Truncated isothermal (SIS_TRUNCATED): truncation radius (arcsec).
    r_trunc: float = 5.0

    # --- Deflector (lens galaxy) light -------------------------------------
    # Light emitted by the lens galaxy itself. It sits in the image plane and is
    # NOT lensed. Without it, real observations would have the deflector's light
    # absorbed into the source during a fit.
    light_model: str = "NONE"   # NONE | SERSIC_ELLIPSE | SERSIC | GAUSSIAN_ELLIPSE | GAUSSIAN
    light_amp: float = 1.0
    light_R_sersic: float = 0.8
    light_n_sersic: float = 4.0
    light_sigma: float = 0.5
    light_e1: float = 0.0
    light_e2: float = 0.0

    def has_light(self) -> bool:
        return self.light_model not in ("NONE", "", None)

    def light_kwargs(self) -> dict:
        """lenstronomy kwargs for this lens's light profile (image plane)."""
        common = {
            "amp": float(self.light_amp),
            "center_x": float(self.center_x),
            "center_y": float(self.center_y),
        }
        m = self.light_model
        if m in ("SERSIC_ELLIPSE", "SERSIC"):
            common["R_sersic"] = float(self.light_R_sersic)
            common["n_sersic"] = float(self.light_n_sersic)
            if m == "SERSIC_ELLIPSE":
                common["e1"] = float(self.light_e1)
                common["e2"] = float(self.light_e2)
        elif m in ("GAUSSIAN_ELLIPSE", "GAUSSIAN"):
            common["sigma"] = float(self.light_sigma)
            if m == "GAUSSIAN_ELLIPSE":
                common["e1"] = float(self.light_e1)
                common["e2"] = float(self.light_e2)
        return common


# Extended-source light profiles and the kwargs each one needs. All of these are
# resolved (extended) sources, not point sources.
# Pixel integration accuracy used for every rendered model image. The fitter must
# use the same value (fit_data/fitting pass it through) or the model cannot
# reproduce the data and a fit will stall above the noise floor.
SUPERSAMPLING_FACTOR = 3

SOURCE_MODELS = [
    "SERSIC_ELLIPSE", "SERSIC", "GAUSSIAN_ELLIPSE", "GAUSSIAN",
    "HERNQUIST", "CORE_SERSIC",
]

# Deflector-galaxy light profiles (image plane, unlensed). "NONE" means the lens
# galaxy emits no light in the model.
LENS_LIGHT_MODELS = ["NONE", "SERSIC_ELLIPSE", "SERSIC", "GAUSSIAN_ELLIPSE", "GAUSSIAN"]


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
    # Hernquist stellar-halo profile: Rs = scale radius (arcsec).
    Rs: float = 0.5
    # Core-Sersic (bulge with flat core): Rb = break radius (arcsec),
    # gamma = inner power-law slope.
    Rb: float = 0.2
    gamma: float = 2.0

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
        elif model == "CORE_SERSIC":
            common.update(
                R_sersic=float(self.R_sersic),
                Rb=float(self.Rb),
                n_sersic=float(self.n_sersic),
                gamma=float(self.gamma),
                e1=float(self.e1),
                e2=float(self.e2),
            )
        elif model == "HERNQUIST":
            common["Rs"] = float(self.Rs)
        else:  # GAUSSIAN / GAUSSIAN_ELLIPSE
            common["sigma"] = float(self.sigma)
            if model == "GAUSSIAN_ELLIPSE":
                common["e1"] = float(self.e1)
                common["e2"] = float(self.e2)
        return common

    def effective_radius(self) -> float:
        """A representative angular size used for the 3D source blob."""
        if self.model == "HERNQUIST":
            return float(max(self.Rs, 1e-3))
        if self.model.startswith("SERSIC") or self.model == "CORE_SERSIC":
            return float(max(self.R_sersic, 1e-3))
        return float(max(self.sigma, 1e-3))


@dataclass
class PointSourceParams:
    """A point (unresolved) source: lensed, or an unlensed star in the image plane.

    Forward rendering and fitting deliberately use **different** lenstronomy
    parameterisations (mirroring lenstronomy's own split):
      * forward is source-plane based  -> SOURCE_POSITION  (ra_source, ...),
        images + magnification solved from the lens equation;
      * fitting is image-plane based   -> LENSED_POSITION  (ra_image, ...),
        the observed image positions are the free parameters.
    ``model_config_from_result`` bridges the two with ray-shooting.
    """

    model: str = "LENSED"       # LENSED (source plane) | UNLENSED (image plane)
    source_amp: float = 1.0     # intrinsic flux (LENSED); on-sky flux = |mu| * amp
    point_amp: float = 1.0      # on-sky flux (UNLENSED star)
    center_x: float = 0.1       # LENSED: source-plane x, UNLENSED: image-plane x
    center_y: float = -0.1
    redshift: float = 1.5
    ref_source: int = -1        # >= 0: reuse source[idx] centre + redshift (AGN)

    def position_and_redshift(self, config: "Config"):
        """Resolve (ra, dec, redshift), following a source reference if set."""
        if 0 <= self.ref_source < len(config.sources):
            src = config.sources[self.ref_source]
            return (float(src.center_x), float(src.center_y), float(src.redshift))
        return (float(self.center_x), float(self.center_y), float(self.redshift))

    def lens_source_kwargs(self, config: "Config") -> tuple[str, dict, bool]:
        """(lenstronomy point-source type, kwargs_ps dict, fixed_magnification)."""
        ra, dec, _ = self.position_and_redshift(config)
        if self.model == "UNLENSED":
            return ("UNLENSED",
                    {"ra_image": [ra], "dec_image": [dec],
                     "point_amp": [float(self.point_amp)]}, False)
        return ("SOURCE_POSITION",
                {"ra_source": ra, "dec_source": dec,
                 "source_amp": float(self.source_amp)}, True)


def gaussian_psf_kernel(fwhm_arcsec: float, delta_pix: float, size: int = 0) -> np.ndarray:
    """Build a normalised 2D Gaussian PSF kernel for the given pixel scale.

    ``size`` defaults to an odd kernel spanning ~4 sigma. A ``fwhm_arcsec`` of 0
    yields a 1x1 delta kernel (no blurring).
    """
    if fwhm_arcsec is None or fwhm_arcsec <= 0:
        return np.array([[1.0]])
    sigma_pix = (fwhm_arcsec / 2.354820045) / float(delta_pix)
    if size <= 0:
        size = int(max(3, 2 * int(np.ceil(3 * sigma_pix)) + 1))
    if size % 2 == 0:
        size += 1
    c = (size - 1) / 2.0
    y, x = np.mgrid[0:size, 0:size]
    kernel = np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * sigma_pix ** 2))
    total = kernel.sum()
    return kernel / total if total > 0 else np.array([[1.0]])


@dataclass
class Config:
    """Full user configuration passed to :func:`compute`."""

    lenses: list[LensParams] = field(default_factory=lambda: [LensParams()])
    sources: list[SourceParams] = field(default_factory=lambda: [SourceParams()])
    point_sources: list[PointSourceParams] = field(default_factory=list)
    num_pix: int = 150
    delta_pix: float = 0.05
    # Convolution kernel applied to the model image. The default 1x1 kernel is a
    # delta function (no seeing); supply a real PSF to match observed data.
    psf_kernel: np.ndarray | None = None
    # Constant sky background added to the model image.
    sky_amp: float = 0.0


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
    source_x: float = 0.0           # reference source plane x (arcsec)
    source_y: float = 0.0           # reference source plane y (arcsec)
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
    "NFW": ("NFW", ["Rs", "alpha_Rs"]),        # dark-matter halo
    "SIS_TRUNCATED": ("SIS_TRUNCATED", ["theta_E", "r_trunc"]),
}

# Controls that always matter, whatever the profile: the centroid and—for a
# lens—the external shear terms.
_LENS_ALWAYS_NAMES = {"gamma1", "gamma2", "center_x", "center_y"}
_SOURCE_ALWAYS_NAMES = {"amp", "center_x", "center_y"}


def lens_param_names(model: str) -> set:
    """Slider names that are physically relevant for a lens ``model``.

    External shear (gamma1/gamma2) and the centroid apply to every profile, so
    they are always included; only the profile-specific parameters vary with the
    model (e.g. NFW shows R_s, alpha_Rs and no theta_E/ellipticity).
    """
    _, params = _MODEL_PROFILE.get(model, _MODEL_PROFILE["SIS"])
    return set(params) | _LENS_ALWAYS_NAMES


def source_param_names(model: str) -> set:
    """Slider names physically relevant for a source light ``model``."""
    if model not in SOURCE_MODELS:
        model = "SERSIC_ELLIPSE"
    if model == "CORE_SERSIC":
        base = {"amp", "R_sersic", "Rb", "n_sersic", "gamma", "e1", "e2"}
    elif model.startswith("SERSIC"):
        base = {"amp", "R_sersic", "n_sersic"}
        if model == "SERSIC_ELLIPSE":
            base |= {"e1", "e2"}
    elif model == "HERNQUIST":
        base = {"amp", "Rs"}
    else:  # GAUSSIAN / GAUSSIAN_ELLIPSE
        base = {"amp", "sigma"}
        if model == "GAUSSIAN_ELLIPSE":
            base |= {"e1", "e2"}
    return base | _SOURCE_ALWAYS_NAMES


def available_lens_models() -> list:
    """Lens models that can actually be used in this environment.

    ``PEMD`` needs the optional Fortran extension ``fastell4py``; it is only
    offered when that import succeeds, so the UI never presents a model that is
    guaranteed to fail.  The rest are pure-Python lenstronomy profiles.
    """
    models = ["SIS", "SIE", "SPEP", "NFW", "SIS_TRUNCATED"]
    try:
        from lenstronomy.LensModel.Profiles.pemd import PEMD  # noqa: F401

        PEMD()
        models.append("PEMD")
    except Exception:
        pass
    return models


# Every model this app knows about; the UI offers available_lens_models().
LENS_MODELS = ["SIS", "SIE", "SPEP", "PEMD", "NFW", "SIS_TRUNCATED"]


def lens_kwargs(lens: LensParams) -> dict:
    """Build a single lenstronomy kwargs dict for one lens profile (no shear)."""
    center = {
        "center_x": float(lens.center_x),
        "center_y": float(lens.center_y),
    }
    if lens.model == "NFW":
        return {"Rs": float(lens.Rs), "alpha_Rs": float(lens.alpha_Rs), **center}
    if lens.model == "SIS_TRUNCATED":
        return {
            "theta_E": float(lens.theta_E),
            "r_trunc": float(lens.r_trunc),
            **center,
        }
    base = {"theta_E": float(lens.theta_E), **center}
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
               l.center_x, l.center_y, l.e1, l.e2, l.gamma,
               l.Rs, l.alpha_Rs, l.r_trunc) for l in lenses),
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


def render_image(config: Config) -> np.ndarray:
    """Render ONLY the model image (lensed sources + point sources + deflector
    light + sky).

    ``compute`` additionally evaluates the Fermat/time-delay fields, the critical
    curve, the caustic and the image positions, which is ~30x more expensive. A
    fit evaluates the model thousands of times, so it must use this cheap path
    (it is also what the live fit preview draws).
    """
    num_pix = int(config.num_pix)
    delta_pix = float(config.delta_pix)
    image = np.zeros((num_pix, num_pix))
    for source in config.sources:
        lens_model, kwargs_lens = _get_lens_model(config.lenses, source.redshift)
        image += _render_source_image(
            lens_model, kwargs_lens, source, num_pix, delta_pix,
            psf_kernel=config.psf_kernel,
        )
    # Point sources: PSF-convolved spikes (lensed or unlensed).
    image += _render_point_sources(config, num_pix, delta_pix,
                                   psf_kernel=config.psf_kernel)
    # Deflector light: image plane, unlensed, added once.
    image += _render_lens_light(config, num_pix, delta_pix)
    if config.sky_amp:
        image += float(config.sky_amp)
    return image


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
        image = render_image(config)
        image_positions = []
        for si, source in enumerate(config.sources):
            lens_model, kwargs_lens = _get_lens_model(config.lenses, source.redshift)
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
            source_x=ref_x,
            source_y=ref_y,
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


def _lens_light_components(config):
    """Build (model_list, kwargs_list) for the deflector galaxy lights."""
    model_list, kwargs_list = [], []
    for lens in config.lenses:
        if lens.has_light():
            model_list.append(lens.light_model)
            kwargs_list.append(lens.light_kwargs())
    return model_list, kwargs_list


def _render_lens_light(config, num_pix, delta_pix):
    """Render the (unlensed) deflector light and sky background, PSF-convolved."""
    model_list, kwargs_list = _lens_light_components(config)
    if not model_list:
        return np.zeros((num_pix, num_pix))

    from lenstronomy.Data.imaging_data import ImageData
    from lenstronomy.Data.psf import PSF
    from lenstronomy.ImSim.image_model import ImageModel
    from lenstronomy.LightModel.light_model import LightModel
    from lenstronomy.Util import util

    x_grid, y_grid = util.make_grid(num_pix, delta_pix)
    data = ImageData(
        ra_at_xy_0=x_grid[0], dec_at_xy_0=y_grid[0],
        transform_pix2angle=np.array([[delta_pix, 0], [0, delta_pix]]),
        image_data=np.zeros((num_pix, num_pix)),
    )
    psf = PSF(psf_type="PIXEL", pixel_size=delta_pix,
              kernel_point_source=_normalised_kernel(config.psf_kernel))
    image_model = ImageModel(
        data_class=data, psf_class=psf,
        lens_model_class=None, source_model_class=None,
        lens_light_model_class=LightModel(light_model_list=model_list),
        kwargs_numerics={"supersampling_factor": SUPERSAMPLING_FACTOR},
    )
    out = image_model.image(kwargs_lens=None, kwargs_source=None,
                            kwargs_lens_light=kwargs_list)
    return np.asarray(out, dtype=float)


def _normalised_kernel(psf_kernel):
    """Return a PSF kernel that integrates to 1 (identity fallback)."""
    kernel = np.asarray(psf_kernel, dtype=float) if psf_kernel is not None \
        else np.array([[1.0]])
    if kernel.ndim != 2 or kernel.size == 0:
        return np.array([[1.0]])
    total = kernel.sum()
    if total > 0:
        return kernel / total
    return np.array([[1.0]])


def _make_image_model(lens_model, num_pix, delta_pix, psf_kernel,
                      source_light_model=None, point_source_class=None):
    """One ImageModel on the model grid (same convention as the data grids).

    ``source_light_model`` / ``point_source_class`` are optional lenstronomy
    class instances (only what is rendered is supplied).
    """
    from lenstronomy.Data.imaging_data import ImageData
    from lenstronomy.Data.psf import PSF
    from lenstronomy.ImSim.image_model import ImageModel
    from lenstronomy.Util import util

    # Use lenstronomy's own grid convention so the sky grid is centred on the
    # lens. ``make_grid`` returns coordinates spanning +/- (num_pix/2 - 0.5) *
    # delta_pix, i.e. symmetric about (0, 0); its first values are the correct
    # ra/dec origin of pixel (0, 0). Getting this wrong puts the lens outside
    # the field of view and the image no longer looks lensed.
    x_grid, y_grid = util.make_grid(num_pix, delta_pix)
    data = ImageData(
        ra_at_xy_0=x_grid[0], dec_at_xy_0=y_grid[0],
        transform_pix2angle=np.array([[delta_pix, 0], [0, delta_pix]]),
        image_data=np.zeros((num_pix, num_pix)),
    )
    # Convolve with the supplied PSF kernel; a 1x1 kernel means no blurring.
    # The kernel is normalised so it cannot rescale the image brightness: a PSF
    # must integrate to unity.
    psf = PSF(psf_type="PIXEL", pixel_size=delta_pix,
              kernel_point_source=_normalised_kernel(psf_kernel))
    return ImageModel(
        data_class=data, psf_class=psf, lens_model_class=lens_model,
        source_model_class=source_light_model,
        point_source_class=point_source_class,
        kwargs_numerics={"supersampling_factor": SUPERSAMPLING_FACTOR},
    )


def _render_source_image(lens_model, kwargs_lens, source, num_pix, delta_pix,
                         psf_kernel=None):
    from lenstronomy.LightModel.light_model import LightModel

    light_model = LightModel(light_model_list=[_valid_source_model(source.model)])
    kwargs_light = [source.profile_kwargs()]
    image_model = _make_image_model(
        lens_model, num_pix, delta_pix, psf_kernel,
        source_light_model=light_model,
    )
    return np.asarray(image_model.image(kwargs_lens, kwargs_light), dtype=float)


def _point_source_kernel(psf_kernel):
    """PSF kernel for point-source rendering.

    A bare 1x1 delta breaks lenstronomy's point-source placement: the sub-pixel
    ``ndimage.shift`` of a single-pixel kernel drops all its flux (and
    supersampling it even raises).  Promote it to a 3x3 kernel with only the
    centre set — still a pixel-delta (no blur), but numerically well-behaved.
    """
    kernel = _normalised_kernel(psf_kernel)
    if kernel.shape == (1, 1):
        out = np.zeros((3, 3), dtype=float)
        out[1, 1] = 1.0
        return out
    return kernel


def _render_point_sources(config, num_pix, delta_pix, psf_kernel=None):
    """Render the point sources: PSF-convolved spikes at the lensed image
    positions (LENSED, images solved from the source plane) or at a fixed
    image-plane position (UNLENSED star). Additive with the extended sources."""
    if not config.point_sources:
        return np.zeros((num_pix, num_pix))

    from lenstronomy.PointSource.point_source import PointSource

    pkernel = _point_source_kernel(psf_kernel)
    total = np.zeros((num_pix, num_pix))
    for point in config.point_sources:
        ptype, kw_ps, fixed_mag = point.lens_source_kwargs(config)
        z = point.position_and_redshift(config)[2]
        lens_model, kwargs_lens = _get_lens_model(config.lenses, z)
        point_source = PointSource(
            point_source_type_list=[ptype],
            lens_model=lens_model,
            fixed_magnification_list=[fixed_mag],
        )
        image_model = _make_image_model(
            lens_model, num_pix, delta_pix, pkernel,
            point_source_class=point_source,
        )
        total += np.asarray(
            image_model.image(kwargs_lens, kwargs_ps=[kw_ps],
                              source_add=False, lens_light_add=False),
            dtype=float,
        )
    return total


def _solve_images(lens_model, kwargs_lens, source, color_idx=0):
    from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

    solver = LensEquationSolver(lens_model)
    x, y = solver.findBrightImage(
        source.center_x, source.center_y, kwargs_lens, numImages=4
    )
    return np.asarray(x), np.asarray(y), color_idx


def _critical_curve(lens_model, kwargs_lens, window=None):
    from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

    if window is None:
        # Adapt the trace window to the lens scale.  A fixed ±2.5 window clips a
        # larger-θ_E critical curve into an arc that pokes out of the panel (or
        # vanishes entirely); enclosing the lens centre + ~2.5×θ_E of margin keeps
        # the whole curve.  The search-grid spacing scales with the window so the
        # compute cost stays roughly constant.
        centers = [np.hypot(float(kw.get("center_x", 0.0)),
                            float(kw.get("center_y", 0.0))) for kw in kwargs_lens]
        thetas = [abs(float(kw.get("theta_E", 0.0))) for kw in kwargs_lens]
        extent = max(centers, default=0.0) + 2.5 * max(thetas, default=0.0)
        window = max(2.5, min(extent + 1.0, 200.0))
    grid_scale = window / 250.0

    ext = LensModelExtensions(lens_model)
    ra_c, dec_c, ra_ca, dec_ca = ext.critical_curve_caustics(
        kwargs_lens=kwargs_lens, compute_window=window, grid_scale=grid_scale
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
