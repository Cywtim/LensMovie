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


def _run_swarm_with_preview(fs, config, data, ref_source_index, *,
                            n_particles, n_iterations, sigma_scale,
                            attempt, attempts, say, preview, preview_interval):
    """Drive lenstronomy's PSO one iteration at a time, reporting progress.

    ``FittingSequence.fit_sequence([['PSO', ...]])`` runs the whole swarm in one
    blocking call with no hook, so the swarm is driven directly through
    ``ParticleSwarmOptimizer.sample()`` (the very generator that
    ``FittingSequence.pso`` consumes). The starting bounds are built exactly as
    ``FittingSequence.pso`` builds them. Returns the best-fit kwargs dict.
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

    best_pos = init_pos
    last_preview = 0.0
    offset = None      # maps lenstronomy's logL onto our chi2 scale
    for it, _ in enumerate(swarm.sample(max_iter=int(n_iterations), verbose=False)):
        best_pos = swarm.global_best.position

        now = time.time()
        if preview is not None and (now - last_preview) >= float(preview_interval):
            last_preview = now
            try:
                kw_i = param_class.args2kwargs(best_pos, bijective=True)
                cfg_i = model_config_from_result(config, kw_i, data.num_pix,
                                                 ref_source_index)
                image_i = lc.render_image(cfg_i)

                # Report the fitter's OWN objective rather than recomputing it:
                # the swarm's global best is monotonic by construction, whereas a
                # recomputed chi2 (different noise/mask handling inside
                # lenstronomy's likelihood) can wobble. logL = -chi2/2 up to a
                # constant, calibrated on the first preview so the numbers are
                # comparable with the final chi2.
                logl = float(swarm.global_best.fitness)
                if offset is None:
                    offset = fd.chi2(image_i, data) + 2.0 * logl
                chi2_i = -2.0 * logl + offset
                preview(it + 1, int(n_iterations), chi2_i, image_i)
            except Exception:
                pass      # a preview must never break the fit

    say(f"restart {attempt + 1}/{attempts}: swarm finished "
        f"({int(n_iterations)} iterations)")

    return param_class.args2kwargs(best_pos, bijective=True)


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
            kw_res = _run_swarm_with_preview(
                fs, config, data, ref_source_index,
                n_particles=int(n_particles), n_iterations=int(n_iterations),
                sigma_scale=float(sigma_scale), attempt=attempt,
                attempts=attempts, say=say,
                preview=preview, preview_interval=preview_interval,
            )
            if polish:
                # Refine the swarm's best solution; this is what makes the fit
                # reliably converge rather than depending on PSO luck. The swarm
                # result is pushed into the sequence state first.
                fs.update_state(kw_res)
                fs.fit_sequence([["SIMPLEX", {
                    "n_iterations": int(n_iterations),
                    "method": "Nelder-Mead",
                }]])
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
