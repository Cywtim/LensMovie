"""Fit the modelled lens image to user-supplied data with lenstronomy.

Uses lenstronomy's own ``FittingSequence`` (PSO), so no extra packages are needed
beyond ``tqdm``. Only *unlocked* parameters participate in the fit; locked ones
are passed to lenstronomy as ``kwargs_fixed`` so they are held exactly.

The module is Qt-independent: the GUI hands in
  * a :class:`~app.lensing_calc.Config` (the current model),
  * a :class:`~app.fit_data.FitData` (the data on the model grid),
  * per-entry parameter specs ``{name: (value, lower, upper, fixed)}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import fit_data as fd
from . import lensing_calc as lc

# UI parameter name -> lenstronomy kwargs name, per component.
_LENS_LIGHT_PREFIX = "light_"

# Parameters that a lens *profile* owns (vs the separate SHEAR entry).
_SHEAR_NAMES = ("gamma1", "gamma2")


class FitError(Exception):
    """Raised when a fit cannot be set up or run."""


@dataclass
class ParamInfo:
    value: float
    lower: float
    upper: float
    fixed: bool = False


@dataclass
class FitResult:
    ok: bool
    config: Optional[lc.Config] = None      # model with the best-fit values
    model: Optional[np.ndarray] = None      # best-fit model image
    residual: Optional[np.ndarray] = None   # data - model
    chi2_before: float = float("nan")
    chi2_after: float = float("nan")
    n_free: int = 0
    ndof: int = 0
    free_names: list = field(default_factory=list)
    fixed_names: list = field(default_factory=list)
    error: str = ""
    log: list = field(default_factory=list)
    # Reduced chi² (chi2_after / ndof): the per-degree-of-freedom goodness of
    # fit, comparable across cases (≈1.0 for a good fit to known noise).
    reduced_chi2: float = float("nan")
    # Parameter chain of the best restart: one entry per swarm iteration giving
    # the global-best value of every free parameter (name -> values, aligned
    # with chain_iter) plus the estimated chi2 at that iteration.  Recorded by
    # the fit loop independently of the (throttled, optional) image preview.
    chain_iter: list = field(default_factory=list)
    chain_chi2: list = field(default_factory=list)
    chain: dict = field(default_factory=dict)

    def improved(self) -> bool:
        return self.ok and self.chi2_after < self.chi2_before


def _to_param_info(spec: dict) -> dict:
    """Normalise a card spec into {name: ParamInfo}."""
    out = {}
    for name, tup in (spec or {}).items():
        value, lower, upper, fixed = tup
        out[name] = ParamInfo(float(value), float(lower), float(upper), bool(fixed))
    return out


def _entry_lists(infos: dict, names, values: dict):
    """Split parameter ``names`` into (init, sigma, fixed, lower, upper).

    ``values`` supplies the model's current value for every name. A name with no
    UI control (not in ``infos``) cannot be fitted, so it is **fixed** at its
    current value rather than silently becoming a free parameter with no bounds.
    """
    init, sigma, fixed, lower, upper = {}, {}, {}, {}, {}
    for name in names:
        info = infos.get(name)
        if info is None:
            if name in values:
                fixed[name] = float(values[name])
            continue
        init[name] = info.value
        lower[name] = info.lower
        upper[name] = info.upper
        # Initial step: a tenth of the range is a reasonable PSO spread.
        sigma[name] = max((info.upper - info.lower) / 10.0, 1e-4)
        if info.fixed:
            fixed[name] = info.value
    return init, sigma, fixed, lower, upper


def _expand_lens_entries(config: lc.Config, lens_infos: list):
    """Build the expanded lens model list with matching init/fixed/bounds.

    Mirrors :func:`lensing_calc.expand_lenses` (a lens with shear becomes
    [profile, SHEAR]) but keeps the parameter bookkeeping aligned.
    """
    model_list, redshift_list = [], []
    init, sigma, fixed, lower, upper = [], [], [], [], []
    free_names, fixed_names = [], []

    for i, lens in enumerate(config.lenses):
        infos = lens_infos[i] if i < len(lens_infos) else {}
        profile = lc._MODEL_PROFILE.get(lens.model, lc._MODEL_PROFILE["SIS"])[0]
        uses_shear = lens.model.endswith("_SHEAR") or (
            lens.model in lc._MODEL_PROFILE
            and abs(lens.gamma1) + abs(lens.gamma2) > 1e-9
        )

        # --- profile entry (everything except the shear terms)
        profile_kwargs = lc.lens_kwargs(lens)
        pnames = [n for n in profile_kwargs if n not in _SHEAR_NAMES]
        pi, ps, pf, pl, pu = _entry_lists(infos, pnames, profile_kwargs)
        model_list.append(profile)
        redshift_list.append(lens.redshift)
        init.append(pi); sigma.append(ps); fixed.append(pf)
        lower.append(pl); upper.append(pu)
        for n in pnames:
            (fixed_names if n in pf else free_names).append(f"lens{i}.{n}")

        # --- optional SHEAR entry
        if uses_shear:
            shear_values = {"gamma1": lens.gamma1, "gamma2": lens.gamma2}
            si, ss, sf, sl, su = _entry_lists(infos, _SHEAR_NAMES, shear_values)
            # lenstronomy's SHEAR fitting parameters are
            # ['gamma1', 'gamma2', 'ra_0', 'dec_0']; the reference point must be
            # supplied or LensParam raises KeyError. We never fit it, so fix it.
            si["ra_0"] = 0.0
            si["dec_0"] = 0.0
            sf["ra_0"] = 0.0
            sf["dec_0"] = 0.0
            model_list.append("SHEAR")
            redshift_list.append(lens.redshift)
            init.append(si); sigma.append(ss); fixed.append(sf)
            lower.append(sl); upper.append(su)
            for n in _SHEAR_NAMES:
                if n in infos:
                    (fixed_names if n in sf else free_names).append(f"lens{i}.{n}")

    return (model_list, redshift_list, init, sigma, fixed, lower, upper,
            free_names, fixed_names)


def _lens_light_entries(config: lc.Config, ll_infos: list):
    """Build lens-light entries for lenses that actually emit light."""
    model_list = []
    init, sigma, fixed, lower, upper = [], [], [], [], []
    free_names, fixed_names = [], []

    for i, lens in enumerate(config.lenses):
        if not lens.has_light():
            continue
        infos = ll_infos[i] if i < len(ll_infos) else {}
        # Map UI names (light_amp, ...) onto lenstronomy kwargs names (amp, ...).
        mapped = {}
        for name, info in infos.items():
            key = name[len(_LENS_LIGHT_PREFIX):] if name.startswith(_LENS_LIGHT_PREFIX) else name
            mapped[key] = info
        names = list(lens.light_kwargs().keys())
        li, ls, lf, ll, lu = _entry_lists(mapped, names, lens.light_kwargs())
        model_list.append(lens.light_model)
        init.append(li); sigma.append(ls); fixed.append(lf)
        lower.append(ll); upper.append(lu)
        for n in names:
            (fixed_names if n in lf else free_names).append(f"lens_light{i}.{n}")

    return model_list, init, sigma, fixed, lower, upper, free_names, fixed_names


def _source_entries(config: lc.Config, src_infos: list):
    model_list = []
    init, sigma, fixed, lower, upper = [], [], [], [], []
    free_names, fixed_names = [], []

    for i, source in enumerate(config.sources):
        infos = src_infos[i] if i < len(src_infos) else {}
        names = list(source.profile_kwargs().keys())
        si, ss, sf, sl, su = _entry_lists(infos, names, source.profile_kwargs())
        model_list.append(source.model)
        init.append(si); sigma.append(ss); fixed.append(sf)
        lower.append(sl); upper.append(su)
        for n in names:
            (fixed_names if n in sf else free_names).append(f"source{i}.{n}")

    return model_list, init, sigma, fixed, lower, upper, free_names, fixed_names


def _point_source_entries(config: lc.Config, ps_infos: list):
    """Build point-source fit bookkeeping.

    The *fit* works in the image plane (LENSED_POSITION / UNLENSED), because the
    observations are the image positions — the reverse of forward rendering,
    which uses SOURCE_POSITION and solves the images from the source plane.
    Image positions are seeded by a forward lens-equation solve of the current
    config and are always free; amplitudes (flux) are linear-parity parameters
    solved by lenstronomy's linear solver unless locked by the UI.
    """
    model_list = []
    fixed_mag = []
    init, sigma, fixed, lower, upper = [], [], [], [], []
    free_names, fixed_names = [], []

    for i, point in enumerate(config.point_sources):
        infos = _to_param_info(ps_infos[i] if i < len(ps_infos) else {})
        ra, dec, z = point.position_and_redshift(config)

        if point.model == "UNLENSED":
            ra_image, dec_image = [ra], [dec]
            amp_name, amp_val = "point_amp", point.point_amp
            num = 1
        else:      # LENSED -> observed as an image-plane point source
            ra_image, dec_image = _seed_image_positions(config, ra, dec, z)
            amp_name, amp_val = "source_amp", point.source_amp
            num = len(ra_image)

        # Image positions are always sampled (with a generous search window).
        init_kw = {"ra_image": list(ra_image), "dec_image": list(dec_image)}
        sigma_kw = {"ra_image": [0.1] * num, "dec_image": [0.1] * num}
        lower_kw = {"ra_image": [-8.0] * num, "dec_image": [-8.0] * num}
        upper_kw = {"ra_image": [8.0] * num, "dec_image": [8.0] * num}
        fixed_kw = {}

        amp_info = infos.get(amp_name)
        if amp_info is not None and amp_info.fixed:
            fixed_kw[amp_name] = float(amp_val)
            fixed_names.append(f"point{i}.{amp_name}")
        else:
            init_kw[amp_name] = float(amp_val)
            if amp_info is not None:
                sigma_kw[amp_name] = max(
                    (amp_info.upper - amp_info.lower) / 10.0, 1e-4)
                lower_kw[amp_name], upper_kw[amp_name] = amp_info.lower, amp_info.upper
            else:
                sigma_kw[amp_name], lower_kw[amp_name], upper_kw[amp_name] = 0.1, 0.0, 100.0
            free_names.append(f"point{i}.{amp_name}")

        model_list.append("UNLENSED" if point.model == "UNLENSED" else "LENSED_POSITION")
        fixed_mag.append(False if point.model == "UNLENSED" else True)
        init.append(init_kw); sigma.append(sigma_kw); fixed.append(fixed_kw)
        lower.append(lower_kw); upper.append(upper_kw)
        for k in range(num):
            free_names.append(f"point{i}.ra_image[{k}]")
            free_names.append(f"point{i}.dec_image[{k}]")

    return (model_list, fixed_mag, init, sigma, fixed, lower, upper,
            free_names, fixed_names)


def _seed_image_positions(config, ra, dec, z, max_images=4, fallback=True):
    """Solve the current lens for the images of a source-plane point.

    Used to initialise a LENSED_POSITION fit from the forward model.
    """
    try:
        from lenstronomy.LensModel.Solver.lens_equation_solver import \
            LensEquationSolver

        lens_model, kwargs_lens = lc._get_lens_model(config.lenses, z)
        solver = LensEquationSolver(lens_model)
        x, y = solver.findBrightImage(ra, dec, kwargs_lens, numImages=max_images)
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(x) and np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
            return list(x), list(y)
    except Exception:
        pass
    if fallback:
        return [ra], [dec]      # unlensed fallback: a single seed at the source
    return [], []


def build_setup(
    config: lc.Config,
    data: fd.FitData,
    lens_specs: list,
    lens_light_specs: list,
    source_specs: list,
    point_source_specs: list = None,
    ref_source_index: int = -1,
):
    """Build ``(kwargs_data_joint, kwargs_model, kwargs_params, free, fixed)``."""
    if data is None:
        raise FitError("no data prepared: load an image and enable the model grid")
    if data.noise is None:
        raise FitError("no noise map: a chi-squared fit needs one (load a noise file)")
    if config.num_pix != data.num_pix:
        raise FitError(
            f"model grid {config.num_pix} != data grid {data.num_pix}"
        )

    lens_infos = [_to_param_info(s) for s in lens_specs]
    ll_infos = [_to_param_info(s) for s in lens_light_specs]
    src_infos = [_to_param_info(s) for s in source_specs]

    (lens_models, lens_zs, la, lb, lc_, ld, le,
     lens_free, lens_fixed) = _expand_lens_entries(config, lens_infos)
    (ll_models, lla, llb, llc, lld, lle,
     ll_free, ll_fixed) = _lens_light_entries(config, ll_infos)
    (src_models, sa, sb, sc, sd, se,
     src_free, src_fixed) = _source_entries(config, src_infos)

    if point_source_specs is None:
        point_source_specs = []
    (ps_models, ps_fixed_mag, pa, pb, pc, pd, pe,
     ps_free, ps_fixed) = _point_source_entries(config, point_source_specs)

    free_names = lens_free + ll_free + src_free + ps_free
    fixed_names = lens_fixed + ll_fixed + src_fixed + ps_fixed

    # Reference source redshift: FittingSequence takes a single z_source.
    idx = ref_source_index if ref_source_index >= 0 else len(config.sources) - 1
    idx = max(0, min(len(config.sources) - 1, idx))
    z_source = config.sources[idx].redshift if config.sources else 1.5

    kwargs_model = {
        "lens_model_list": lens_models,
        "lens_redshift_list": lens_zs,
        "z_source": z_source,
        "multi_plane": True,
    }
    if ll_models:
        kwargs_model["lens_light_model_list"] = ll_models
    if src_models:
        kwargs_model["source_light_model_list"] = src_models
    if ps_models:
        kwargs_model["point_source_model_list"] = ps_models
        kwargs_model["fixed_magnification_list"] = ps_fixed_mag

    kwargs_params = {
        "lens_model": [la, lb, lc_, ld, le],
        "lens_light_model": [lla, llb, llc, lld, lle],
        "source_model": [sa, sb, sc, sd, se],
    }
    if ps_models:
        kwargs_params["point_source_model"] = [pa, pb, pc, pd, pe]

    kwargs_data_joint = _build_data_joint(config, data)

    if not free_names:
        raise FitError(
            "every parameter is fixed — unlock at least one parameter (🔓) to fit"
        )

    return kwargs_data_joint, kwargs_model, kwargs_params, free_names, fixed_names


def _build_data_joint(config: lc.Config, data: fd.FitData):
    """Assemble the ``multi_band_list`` structure FittingSequence expects."""
    from lenstronomy.Util import util

    num_pix, delta = data.num_pix, data.delta_pix
    x_grid, y_grid = util.make_grid(num_pix, delta)

    kwargs_data = {
        "image_data": np.asarray(data.image, dtype=float),
        "noise_map": np.asarray(data.noise, dtype=float),
        "ra_at_xy_0": x_grid[0],
        "dec_at_xy_0": y_grid[0],
        "transform_pix2angle": np.array([[delta, 0], [0, delta]]),
        "exposure_time": 1.0,
        "background_rms": float(np.median(data.noise)),
    }
    if data.mask is not None:
        kwargs_data["likelihood_mask"] = np.asarray(data.mask, dtype=bool)

    kernel = fd.effective_psf_kernel(data)
    kwargs_psf = {
        "psf_type": "PIXEL",
        "pixel_size": delta,
        "kernel_point_source": np.asarray(kernel, dtype=float),
    }
    # Must match the renderer or the fit stalls above the noise floor.
    kwargs_numerics = {"supersampling_factor": lc.SUPERSAMPLING_FACTOR}
    return {
        "multi_band_list": [[kwargs_data, kwargs_psf, kwargs_numerics]],
        "multi_band_type": "single-band",
    }


def _ps_roundtrip(config: lc.Config, kw_lens: list,
                  kw_ps_result: list) -> list[lc.PointSourceParams]:
    """Map the fit's *image-plane* point-source kwargs back to source-plane app
    params.

    Forward rendering is SOURCE_POSITION (solve images from the source plane);
    the fit is LENSED_POSITION (refine the observed image positions).  This is
    the inverse round trip: image positions are ray-shot back to the source
    plane (``PointSource.source_position``) and the magnification-corrected
    amplitude is recovered (``source_amplitude``), so a re-render of the app
    config reproduces exactly what the fitter saw.
    """
    from lenstronomy.PointSource.point_source import PointSource

    out = []
    for i, point in enumerate(config.point_sources):
        if i >= len(kw_ps_result):
            out.append(point)
            continue
        kw = kw_ps_result[i] or {}
        z = point.position_and_redshift(config)[2]

        if point.model == "UNLENSED":
            ras = np.asarray(kw.get("ra_image") or [point.center_x], dtype=float)
            decs = np.asarray(kw.get("dec_image") or [point.center_y], dtype=float)
            amps = np.asarray(kw.get("point_amp") or [point.point_amp], dtype=float)
            out.append(lc.PointSourceParams(
                model="UNLENSED",
                point_amp=float(amps.mean()),
                source_amp=point.source_amp,
                center_x=float(ras[0]),
                center_y=float(decs[0]),
                redshift=point.redshift,
                ref_source=point.ref_source,
            ))
            continue

        # LENSED: ray-shoot the fitted image positions back through the fitted
        # lens to recover the intrinsic source-plane coordinates + flux.
        try:
            lens_model, _ = lc._get_lens_model(config.lenses, z)
            fit_ps = PointSource(
                point_source_type_list=["LENSED_POSITION"],
                lens_model=lens_model,
                fixed_magnification_list=[True],
            )
            ra_dec = np.asarray(
                fit_ps.source_position([kw], kwargs_lens=kw_lens), dtype=float
            ).reshape(-1)
            amp_src = float(np.asarray(
                fit_ps.source_amplitude([kw], kwargs_lens=kw_lens), dtype=float
            ).reshape(-1)[0])
            ra_img = np.asarray(kw.get("ra_image") or [point.center_x], dtype=float)
            dec_img = np.asarray(kw.get("dec_image") or [point.center_y], dtype=float)
            if len(ra_dec) >= 2 and np.all(np.isfinite(ra_dec)):
                out.append(lc.PointSourceParams(
                    model="LENSED",
                    source_amp=amp_src,
                    center_x=float(ra_dec[0]),
                    center_y=float(ra_dec[1]),
                    redshift=point.redshift,
                    ref_source=point.ref_source,
                ))
                continue
            ra_ok, dec_ok = bool(np.isfinite(ra_img).all()), bool(np.isfinite(dec_img).all())
        except Exception:
            ra_ok = dec_ok = False
            ra_img = np.asarray([point.center_x])
            dec_img = np.asarray([point.center_y])
        if not (ra_ok and dec_ok):
            out.append(lc.PointSourceParams(
                model="LENSED",
                source_amp=point.source_amp,
                center_x=float(ra_img[0]),
                center_y=float(dec_img[0]),
                redshift=point.redshift,
                ref_source=point.ref_source,
            ))
    return out


def model_config_from_result(config: lc.Config, kwargs_result: dict,
                             num_pix: int, ref_source_index: int = -1) -> lc.Config:
    """Rebuild an app Config from lenstronomy's best-fit kwargs."""
    kw_lens = kwargs_result.get("kwargs_lens") or []
    kw_ll = kwargs_result.get("kwargs_lens_light") or []
    kw_src = kwargs_result.get("kwargs_source") or []

    lenses = []
    li = 0          # index into the expanded lens kwargs list
    lli = 0
    for lens in config.lenses:
        profile_names = [n for n in lc.lens_kwargs(lens) if n not in _SHEAR_NAMES]
        kw = kw_lens[li] if li < len(kw_lens) else {}
        li += 1
        uses_shear = lens.model.endswith("_SHEAR") or (
            lens.model in lc._MODEL_PROFILE
            and abs(lens.gamma1) + abs(lens.gamma2) > 1e-9
        )
        gamma1, gamma2 = lens.gamma1, lens.gamma2
        if uses_shear and li < len(kw_lens):
            sh = kw_lens[li]
            li += 1
            gamma1 = float(sh.get("gamma1", gamma1))
            gamma2 = float(sh.get("gamma2", gamma2))

        light_model = lens.light_model
        light_kw = kw_ll[lli] if light_model not in ("NONE", "", None) and lli < len(kw_ll) else {}
        if light_model not in ("NONE", "", None):
            lli += 1

        lenses.append(lc.LensParams(
            model=lens.model,
            theta_E=float(kw.get("theta_E", lens.theta_E)),
            gamma1=gamma1,
            gamma2=gamma2,
            center_x=float(kw.get("center_x", lens.center_x)),
            center_y=float(kw.get("center_y", lens.center_y)),
            e1=float(kw.get("e1", lens.e1)),
            e2=float(kw.get("e2", lens.e2)),
            gamma=float(kw.get("gamma", lens.gamma)),
            redshift=lens.redshift,
            light_model=light_model,
            light_amp=float(light_kw.get("amp", lens.light_amp)),
            light_R_sersic=float(light_kw.get("R_sersic", lens.light_R_sersic)),
            light_n_sersic=float(light_kw.get("n_sersic", lens.light_n_sersic)),
            light_sigma=float(light_kw.get("sigma", lens.light_sigma)),
            light_e1=float(light_kw.get("e1", lens.light_e1)),
            light_e2=float(light_kw.get("e2", lens.light_e2)),
        ))

    sources = []
    for i, source in enumerate(config.sources):
        kw = kw_src[i] if i < len(kw_src) else {}
        sources.append(lc.SourceParams(
            model=source.model,
            amp=float(kw.get("amp", source.amp)),
            R_sersic=float(kw.get("R_sersic", source.R_sersic)),
            n_sersic=float(kw.get("n_sersic", source.n_sersic)),
            sigma=float(kw.get("sigma", source.sigma)),
            e1=float(kw.get("e1", source.e1)),
            e2=float(kw.get("e2", source.e2)),
            center_x=float(kw.get("center_x", source.center_x)),
            center_y=float(kw.get("center_y", source.center_y)),
            redshift=source.redshift,
        ))

    point_sources = _ps_roundtrip(
        config, kw_lens, kwargs_result.get("kwargs_ps") or [])

    return lc.Config(
        lenses=lenses, sources=sources, point_sources=point_sources,
        num_pix=num_pix, delta_pix=config.delta_pix,
        psf_kernel=config.psf_kernel, sky_amp=config.sky_amp,
    )


def _chain_values(config: lc.Config, kw: dict, free_names: list) -> dict:
    """Extract a numeric value per free parameter from one lenstronomy kwargs
    result (the swarm's global best at a given iteration).

    ``free_names`` come from the ``_*_entries`` builders; ``kw`` is the
    bijective kwargs mapping of the current position.  Lens / source / lens-light
    values are read through :func:`model_config_from_result`'s round-trip to the
    app Config (exact for those); point-source *image positions*
    (``pointN.ra_image[k]`` / ``dec_image[k]``) live only in the image-plane
    kwargs, so those are read from ``kwargs_ps`` directly.  Unresolvable names
    yield NaN rather than raising, so one bad key cannot kill the chain.
    """
    kw_ll = kw.get("kwargs_lens_light") or []
    kw_ps = kw.get("kwargs_ps") or []

    def light_lens(j: int):
        """The j-th lens that actually emits light (matches _lens_light_entries)."""
        n = -1
        for lens in config.lenses:
            if lens.has_light():
                n += 1
                if n == j:
                    return lens
        return None

    # kwargs (fitted) name -> LensParams field, for lens light.
    _LIGHT_FIELD = {"amp": "light_amp", "R_sersic": "light_R_sersic",
                    "n_sersic": "light_n_sersic", "sigma": "light_sigma",
                    "e1": "light_e1", "e2": "light_e2"}

    values = {}
    for name in free_names:
        values[name] = float("nan")
        try:
            if name.startswith("lens_light"):
                j = int(name[len("lens_light"):name.index(".")])
                key = name[name.index(".") + 1:]
                arr = kw_ll[j] if j < len(kw_ll) else None
                if arr is not None and key in arr:
                    values[name] = float(arr[key])
                else:
                    lens = light_lens(j)
                    if lens is not None:
                        values[name] = float(getattr(lens, _LIGHT_FIELD.get(key, key)))
            elif name.startswith("lens"):
                i = int(name[4:name.index(".")])
                key = name[name.index(".") + 1:]
                values[name] = float(getattr(config.lenses[i], key))
            elif name.startswith("source"):
                i = int(name[6:name.index(".")])
                key = name[name.index(".") + 1:]
                values[name] = float(getattr(config.sources[i], key))
            elif name.startswith("point"):
                spec = name[name.index(".") + 1:]
                i = int(name[5:name.index(".")])
                if spec.startswith("ra_image") or spec.startswith("dec_image"):
                    head = spec.split("[", 1)[0]
                    k = int(spec[spec.index("[") + 1:spec.index("]")])
                    arr = (kw_ps[i] or {}).get(head) or []
                    if k < len(arr):
                        values[name] = float(arr[k])
                else:
                    values[name] = float(getattr(config.point_sources[i], spec))
        except Exception:
            pass       # keep NaN for this name
    return values


def _run_swarm_with_preview(fs, config, data, ref_source_index, *,
                            n_particles, n_iterations, sigma_scale,
                            attempt, attempts, say, preview, preview_interval,
                            free_names):
    """Drive lenstronomy's PSO one iteration at a time, reporting progress.

    ``FittingSequence.fit_sequence([['PSO', ...]])`` runs the whole swarm in one
    blocking call with no hook, so the swarm is driven directly through
    ``ParticleSwarmOptimizer.sample()`` (the very generator that
    ``FittingSequence.pso`` consumes). The starting bounds are built exactly as
    ``FittingSequence.pso`` builds them.

    Every iteration the global-best position is also recorded into the returned
    ``samples`` list (iteration, chi2 estimate, per-free-parameter values), so a
    *parameter chain* is available for export even when image previews are off.
    Returns ``(best-fit kwargs dict, samples)``.
    """
    import time

    param_class = fs.param_class
    um = fs._updateManager
    init_pos = np.asarray(param_class.kwargs2args(**um.parameter_state), dtype=float)
    sigma = np.asarray(param_class.kwargs2args(**um.sigma_kwargs), dtype=float)
    lo_lim = np.asarray(param_class.kwargs2args(**um.lower_kwargs), dtype=float)
    hi_lim = np.asarray(param_class.kwargs2args(**um.upper_kwargs), dtype=float)

    lower_start = np.maximum(init_pos - sigma * sigma_scale, lo_lim)
    upper_start = np.minimum(init_pos + sigma * sigma_scale, hi_lim)

    from lenstronomy.Sampling.Samplers.pso import ParticleSwarmOptimizer

    swarm = ParticleSwarmOptimizer(
        fs.likelihoodModule.logL, list(lower_start), list(upper_start),
        particle_count=int(n_particles),
    )
    # lenstronomy's initial swarm is purely uniform in the box — the starting
    # position is NOT a candidate, so in many dimensions the random particles can
    # all be worse than a sharp, near-optimal start and the "best" would regress.
    # Canonically ``Sampler.pso`` injects the start as the swarm's *global best*
    # (``set_global_best(init_pos, ...)``), which guarantees the reported result
    # is never worse than the current slider values.  LensMovie drives the raw
    # ``ParticleSwarmOptimizer`` directly, so replicate that seeding, AND pin one
    # real particle at ``init_pos`` so the swarm also explores outward from it.
    try:
        logl0 = float(fs.likelihoodModule.logL(list(init_pos)))
        swarm.set_global_best([float(v) for v in init_pos], [0.0] * len(init_pos), logl0)
    except Exception:
        pass
    try:
        swarm.swarm[0].position = [float(v) for v in init_pos]
        swarm.swarm[0].velocity = [0.0] * len(init_pos)
    except Exception:
        pass

    best_pos = init_pos
    last_preview = 0.0
    offset = None      # maps lenstronomy's logL onto our chi2 scale
    samples = []       # (iteration, chi2_est, {free_name: value})
    for it, _ in enumerate(swarm.sample(max_iter=int(n_iterations), verbose=False)):
        best_pos = swarm.global_best.position

        # Chain bookkeeping first (cheap, needs no rendering): convert the
        # current global best back to kwargs, then to the app Config so every
        # free parameter can be read by name.
        kw_i = param_class.args2kwargs(best_pos, bijective=True)
        cfg_i = model_config_from_result(config, kw_i, data.num_pix,
                                         ref_source_index)
        logl = float(swarm.global_best.fitness)
        if offset is None:
            try:
                offset = fd.chi2(lc.render_image(cfg_i), data) + 2.0 * logl
            except Exception:
                offset = 0.0
        chi2_i = -2.0 * logl + offset
        vals = _chain_values(cfg_i, kw_i, free_names)
        samples.append((it + 1, chi2_i, vals))

        now = time.time()
        if preview is not None and (now - last_preview) >= float(preview_interval):
            last_preview = now
            try:
                image_i = lc.render_image(cfg_i)
                preview(it + 1, int(n_iterations), chi2_i, image_i)
            except Exception:
                pass      # a preview must never break the fit

    say(f"restart {attempt + 1}/{attempts}: swarm finished "
        f"({int(n_iterations)} iterations)")

    return param_class.args2kwargs(best_pos, bijective=True), samples


def run_pso(
    config: lc.Config,
    data: fd.FitData,
    lens_specs: list,
    lens_light_specs: list,
    source_specs: list,
    point_source_specs: list = None,
    n_particles: int = 30,
    n_iterations: int = 100,
    n_restarts: int = 2,
    polish: bool = True,
    sigma_scale: float = 4.0,
    ref_source_index: int = -1,
    progress=None,
    preview=None,
    preview_interval: float = 0.35,
) -> FitResult:
    """Run a PSO fit with lenstronomy's own FittingSequence.

    Only unlocked parameters are varied. PSO is stochastic, so the search is
    restarted ``n_restarts`` times, each run is polished with SIMPLEX
    (Nelder-Mead) when ``polish`` is set, and the lowest-chi-squared solution is
    kept.

    ``preview`` (if given) is called as ``preview(iteration, total, chi2, image)``
    while the swarm runs, so the caller can show the fit converging. It is
    throttled to at most one call per ``preview_interval`` seconds so the extra
    rendering cannot dominate the fit's runtime.

    Returns a :class:`FitResult` carrying the best-fit
    :class:`~app.lensing_calc.Config`, the model image, the residual and the
    chi-squared before/after.
    """
    log = []

    def say(msg):
        log.append(msg)
        if progress is not None:
            try:
                progress(msg)
            except Exception:
                pass

    try:
        kwargs_data_joint, kwargs_model, kwargs_params, free_names, fixed_names = build_setup(
            config, data, lens_specs, lens_light_specs, source_specs,
            point_source_specs, ref_source_index,
        )
    except FitError as exc:
        return FitResult(ok=False, error=str(exc), log=log)

    say(f"{len(free_names)} free parameter(s), {len(fixed_names)} fixed")
    say("free: " + ", ".join(free_names))

    # chi-squared of the starting model
    model_before = lc.compute(config).image
    chi2_before = fd.chi2(model_before, data)
    say(f"chi2 (initial) = {chi2_before:.4g}")

    from lenstronomy.Workflow.fitting_sequence import FittingSequence

    best = None          # (chi2, config, kwargs_result, samples)
    last_error = ""
    attempts = max(1, int(n_restarts))
    for attempt in range(attempts):
        try:
            fs = FittingSequence(
                kwargs_data_joint, kwargs_model, {},
                {"image_likelihood": True, "check_bounds": True},
                kwargs_params, verbose=False,
            )
            kw_res, samples = _run_swarm_with_preview(
                fs, config, data, ref_source_index,
                n_particles=int(n_particles), n_iterations=int(n_iterations),
                sigma_scale=float(sigma_scale), attempt=attempt,
                attempts=attempts, say=say,
                preview=preview, preview_interval=preview_interval,
                free_names=free_names,
            )
            # Swarm-stage solution is always a candidate: with the ``init_pos``
            # particle seeded inside the loop it can never be worse than the
            # start, and we keep it if refinement does not help.
            cfg_swarm = model_config_from_result(config, kw_res, data.num_pix,
                                                 ref_source_index)
            chi2_swarm = fd.chi2(lc.compute(cfg_swarm).image, data)
            cfg_i, kw_i = cfg_swarm, kw_res
            chi2_i = chi2_swarm
            if polish:
                # Refine the swarm's best solution; this is what makes the fit
                # reliably converge rather than depending on PSO luck. The swarm
                # result is pushed into the sequence state first.
                fs.update_state(kw_res)
                fs.fit_sequence([["SIMPLEX", {
                    "n_iterations": int(n_iterations),
                    "method": "Nelder-Mead",
                }]])
                kw_simplex = fs.best_fit()
                cfg_simplex = model_config_from_result(config, kw_simplex,
                                                       data.num_pix,
                                                       ref_source_index)
                chi2_simplex = fd.chi2(lc.compute(cfg_simplex).image, data)
                # SIMPLEX must not hurt: keep whichever solution scores better.
                if chi2_simplex < chi2_swarm:
                    cfg_i, kw_i, chi2_i = cfg_simplex, kw_simplex, chi2_simplex
            say(f"restart {attempt + 1}/{attempts}: chi2 = {chi2_i:.4g}")
            if best is None or chi2_i < best[0]:
                best = (chi2_i, cfg_i, kw_res, samples)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            say(f"restart {attempt + 1}/{attempts} failed: {last_error}")

    # Hard floor: a fit must never report a result *worse* than the starting
    # model (the user's current slider values).  The swarms are random, so on an
    # unlucky draw even the seeded init particle can be beaten by regression; if
    # the best restart is still worse than where we began, keep the start itself.
    if best is not None and best[0] > chi2_before:
        best = (chi2_before, config, None, best[3])

    if best is None:
        return FitResult(ok=False, error=last_error or "fit failed", log=log,
                         chi2_before=chi2_before)

    chi2_after, new_config, _, samples = best
    model_after = lc.compute(new_config).image
    say(f"chi2 (best fit) = {chi2_after:.4g}")

    # Fan the best restart's per-iteration (iteration, chi2, values) samples out
    # into columnar chain data for the parameter-chain export.
    chain_iter = [s[0] for s in samples]
    chain_chi2 = [s[1] for s in samples]
    chain = {name: [s[2].get(name, float("nan")) for s in samples]
             for name in free_names}

    ndof = data.usable_pixels() - len(free_names)
    return FitResult(
        ok=True,
        config=new_config,
        model=model_after,
        residual=np.asarray(data.image) - model_after,
        chi2_before=chi2_before,
        chi2_after=chi2_after,
        n_free=len(free_names),
        ndof=max(ndof, 0),
        reduced_chi2=chi2_after / max(ndof, 1),
        free_names=free_names,
        fixed_names=fixed_names,
        log=log,
        chain_iter=chain_iter,
        chain_chi2=chain_chi2,
        chain=chain,
    )
