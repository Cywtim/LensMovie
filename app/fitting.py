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


def build_setup(
    config: lc.Config,
    data: fd.FitData,
    lens_specs: list,
    lens_light_specs: list,
    source_specs: list,
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

    free_names = lens_free + ll_free + src_free
    fixed_names = lens_fixed + ll_fixed + src_fixed

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

    kwargs_params = {
        "lens_model": [la, lb, lc_, ld, le],
        "lens_light_model": [lla, llb, llc, lld, lle],
        "source_model": [sa, sb, sc, sd, se],
    }

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

    return lc.Config(
        lenses=lenses, sources=sources,
        num_pix=num_pix, delta_pix=config.delta_pix,
        psf_kernel=config.psf_kernel, sky_amp=config.sky_amp,
    )


def run_pso(
    config: lc.Config,
    data: fd.FitData,
    lens_specs: list,
    lens_light_specs: list,
    source_specs: list,
    n_particles: int = 30,
    n_iterations: int = 100,
    n_restarts: int = 2,
    polish: bool = True,
    sigma_scale: float = 4.0,
    ref_source_index: int = -1,
    progress=None,
) -> FitResult:
    """Run a PSO fit with lenstronomy's own FittingSequence.

    Only unlocked parameters are varied. PSO is stochastic, so the search is
    restarted ``n_restarts`` times, each run is polished with SIMPLEX
    (Nelder-Mead) when ``polish`` is set, and the lowest-chi-squared solution is
    kept.

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
            config, data, lens_specs, lens_light_specs, source_specs, ref_source_index,
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

    best = None          # (chi2, config, kwargs_result)
    last_error = ""
    attempts = max(1, int(n_restarts))
    for attempt in range(attempts):
        try:
            fs = FittingSequence(
                kwargs_data_joint, kwargs_model, {},
                {"image_likelihood": True, "check_bounds": True},
                kwargs_params, verbose=False,
            )
            sequence = [["PSO", {
                "sigma_scale": float(sigma_scale),
                "n_particles": int(n_particles),
                "n_iterations": int(n_iterations),
            }]]
            if polish:
                # Refine the swarm's best solution; this is what makes the fit
                # reliably converge rather than depending on PSO luck.
                sequence.append(["SIMPLEX", {
                    "n_iterations": int(n_iterations),
                    "method": "Nelder-Mead",
                }])
            fs.fit_sequence(sequence)
            kw_res = fs.best_fit()
            cfg_i = model_config_from_result(config, kw_res, data.num_pix,
                                             ref_source_index)
            chi2_i = fd.chi2(lc.compute(cfg_i).image, data)
            say(f"restart {attempt + 1}/{attempts}: chi2 = {chi2_i:.4g}")
            if best is None or chi2_i < best[0]:
                best = (chi2_i, cfg_i, kw_res)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            say(f"restart {attempt + 1}/{attempts} failed: {last_error}")

    if best is None:
        return FitResult(ok=False, error=last_error or "fit failed", log=log,
                         chi2_before=chi2_before)

    chi2_after, new_config, _ = best
    model_after = lc.compute(new_config).image
    say(f"chi2 (best fit) = {chi2_after:.4g}")

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
        free_names=free_names,
        fixed_names=fixed_names,
        log=log,
    )
