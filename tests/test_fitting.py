"""Tests for the PSO fitting layer (lenstronomy FittingSequence)."""

import numpy as np
import pytest

from app import fit_data as fd
from app import fitting as ft
from app import lensing_calc as lc


# --------------------------------------------------------------- helpers
def _truth_config():
    return lc.Config(
        lenses=[lc.LensParams(model="SIS", theta_E=1.10, gamma1=0.04, gamma2=-0.02,
                              light_model="SERSIC_ELLIPSE", light_amp=0.6,
                              light_R_sersic=0.9, light_n_sersic=4.0,
                              light_e1=0.15, light_e2=0.05)],
        sources=[lc.SourceParams(model="SERSIC_ELLIPSE", amp=1.0, R_sersic=0.12,
                                 n_sersic=3.0, e1=0.10, e2=-0.10,
                                 center_x=0.08, center_y=-0.06, redshift=1.5)],
        num_pix=60, delta_pix=0.05,
    )


def _specs(config, free_lens=(), free_src=()):
    """Build card specs with everything fixed except the requested names."""
    lens = {n: (getattr(config.lenses[0], n),
                0.5 if n == "theta_E" else (-0.3 if n.startswith("gamma") else
                                            (-2.0 if n.startswith("center") else -0.8)),
                2.0 if n == "theta_E" else (0.3 if n.startswith("gamma") else
                                            (2.0 if n.startswith("center") else 0.8)),
                n not in free_lens)
            for n in ("theta_E", "gamma1", "gamma2", "e1", "e2", "gamma",
                      "center_x", "center_y")}
    llight = {"light_amp": (0.6, 0.0, 5.0, True),
              "light_R_sersic": (0.9, 0.05, 3.0, True),
              "light_n_sersic": (4.0, 0.5, 8.0, True),
              "light_sigma": (0.5, 0.05, 3.0, True),
              "light_e1": (0.15, -0.8, 0.8, True),
              "light_e2": (0.05, -0.8, 0.8, True)}
    src = {"amp": (1.0, 0.05, 5.0, True), "R_sersic": (0.12, 0.01, 1.0, True),
           "sigma": (0.12, 0.01, 1.0, True), "n_sersic": (3.0, 0.5, 8.0, True),
           "e1": (0.10, -0.8, 0.8, True), "e2": (-0.10, -0.8, 0.8, True),
           "center_x": (0.08, -2.0, 2.0, "center_x" not in free_src),
           "center_y": (-0.06, -2.0, 2.0, "center_y" not in free_src)}
    return [lens], [llight], [src]


def _data_for(config, sigma=0.002, seed=3):
    mock = lc.compute(config).image
    rng = np.random.RandomState(seed)
    return fd.prepare_fit_data(
        mock + rng.normal(0, sigma, mock.shape),
        source_delta_pix=config.delta_pix, model_num_pix=config.num_pix,
        model_delta_pix=config.delta_pix, noise=np.full(mock.shape, sigma),
    )


# ---------------------------------------------------------- setup building
def test_build_setup_structure():
    cfg = _truth_config()
    data = _data_for(cfg)
    lens_s, ll_s, src_s = _specs(cfg, free_lens=("theta_E",))

    kdj, km, kp, free, fixed = ft.build_setup(cfg, data, lens_s, ll_s, src_s)

    assert "multi_band_list" in kdj
    band = kdj["multi_band_list"][0]
    assert len(band) == 3                     # [data, psf, numerics]
    assert band[0]["image_data"].shape == (cfg.num_pix, cfg.num_pix)
    assert "noise_map" in band[0]

    assert km["lens_model_list"] == ["SIS", "SHEAR"]
    assert km["lens_light_model_list"] == ["SERSIC_ELLIPSE"]
    assert km["source_light_model_list"] == ["SERSIC_ELLIPSE"]
    assert km["multi_plane"] is True
    assert "z_source" in km

    for key in ("lens_model", "lens_light_model", "source_model"):
        assert key in kp and len(kp[key]) == 5     # init, sigma, fixed, lo, up

    assert free == ["lens0.theta_E"]
    assert any(n.startswith("lens0.gamma") for n in fixed)


def test_only_unlocked_parameters_are_free():
    cfg = _truth_config()
    data = _data_for(cfg)
    lens_s, ll_s, src_s = _specs(cfg, free_lens=("theta_E",), free_src=("center_x",))
    _, _, _, free, fixed = ft.build_setup(cfg, data, lens_s, ll_s, src_s)
    assert set(free) == {"lens0.theta_E", "source0.center_x"}
    assert "lens0.gamma1" in fixed and "source0.center_y" in fixed


def test_locked_values_are_placed_in_kwargs_fixed():
    cfg = _truth_config()
    data = _data_for(cfg)
    lens_s, ll_s, src_s = _specs(cfg, free_lens=("theta_E",))
    kdj, km, kp, _, _ = ft.build_setup(cfg, data, lens_s, ll_s, src_s)

    lens_fixed = kp["lens_model"][2]
    # the locked shear goes in fixed, not free
    assert lens_fixed[1]["gamma1"] == pytest.approx(0.04)
    assert lens_fixed[0]["center_x"] == pytest.approx(0.0)
    # the free theta_E must NOT be in fixed
    assert "theta_E" not in lens_fixed[0]


def test_params_without_ui_controls_are_fixed_not_free():
    """Regression: a parameter with no slider must be fixed, not silently free."""
    cfg = _truth_config()
    data = _data_for(cfg)
    # the lens-light spec has no light_center_x / light_center_y sliders
    lens_s, ll_s, src_s = _specs(cfg, free_lens=("theta_E",))
    _, _, kp, free, _ = ft.build_setup(cfg, data, lens_s, ll_s, src_s)

    ll_fixed = kp["lens_light_model"][2][0]     # fixed dict of the first light entry
    assert "center_x" in ll_fixed and "center_y" in ll_fixed
    assert not any(n.startswith("lens_light") for n in free)


def test_shear_entry_supplies_reference_point():
    """Regression: lenstronomy's SHEAR needs ra_0/dec_0 or LensParam KeyErrors."""
    cfg = _truth_config()          # has non-zero shear -> SIS + SHEAR
    data = _data_for(cfg)
    lens_s, ll_s, src_s = _specs(cfg, free_lens=("theta_E",))
    _, km, kp, _, _ = ft.build_setup(cfg, data, lens_s, ll_s, src_s)
    assert km["lens_model_list"][1] == "SHEAR"
    shear_fixed = kp["lens_model"][2][1]
    assert shear_fixed.get("ra_0") == 0.0
    assert shear_fixed.get("dec_0") == 0.0


def test_setup_errors_are_reported():
    cfg = _truth_config()
    lens_s, ll_s, src_s = _specs(cfg)
    # no data
    with pytest.raises(ft.FitError):
        ft.build_setup(cfg, None, lens_s, ll_s, src_s)
    # data without noise
    data = fd.prepare_fit_data(lc.compute(cfg).image, source_delta_pix=0.05,
                               model_num_pix=cfg.num_pix, model_delta_pix=0.05)
    with pytest.raises(ft.FitError):
        ft.build_setup(cfg, data, lens_s, ll_s, src_s)
    # grid mismatch
    d2 = _data_for(cfg)
    cfg2 = lc.Config(lenses=cfg.lenses, sources=cfg.sources,
                     num_pix=cfg.num_pix + 10, delta_pix=cfg.delta_pix)
    with pytest.raises(ft.FitError):
        ft.build_setup(cfg2, d2, lens_s, ll_s, src_s)


def test_all_fixed_raises():
    cfg = _truth_config()
    data = _data_for(cfg)
    lens_s, ll_s, src_s = _specs(cfg)      # nothing free
    with pytest.raises(ft.FitError):
        ft.build_setup(cfg, data, lens_s, ll_s, src_s)


# ---------------------------------------------------------- model rebuild
def test_model_config_from_result_roundtrip():
    cfg = _truth_config()
    kw = {
        "kwargs_lens": [{"theta_E": 1.23, "center_x": 0.01, "center_y": 0.02},
                        {"gamma1": 0.05, "gamma2": -0.03}],
        "kwargs_lens_light": [{"amp": 0.7, "R_sersic": 1.0, "n_sersic": 4.0,
                               "e1": 0.1, "e2": 0.0, "center_x": 0.0, "center_y": 0.0}],
        "kwargs_source": [{"amp": 1.1, "R_sersic": 0.15, "n_sersic": 3.0,
                           "e1": 0.1, "e2": -0.1, "center_x": 0.09, "center_y": -0.05}],
    }
    new = ft.model_config_from_result(cfg, kw, cfg.num_pix)
    assert new.lenses[0].theta_E == pytest.approx(1.23)
    assert new.lenses[0].gamma1 == pytest.approx(0.05)
    assert new.lenses[0].gamma2 == pytest.approx(-0.03)
    assert new.lenses[0].light_amp == pytest.approx(0.7)
    assert new.sources[0].center_x == pytest.approx(0.09)
    # untouched fields survive
    assert new.lenses[0].light_model == "SERSIC_ELLIPSE"
    assert new.sources[0].redshift == cfg.sources[0].redshift


# ---------------------------------------------------------------- fitting
@pytest.mark.slow
def test_pso_recovers_a_single_free_parameter():
    """A one-parameter fit must land on the injected theta_E."""
    truth = _truth_config()
    data = _data_for(truth)

    start = lc.Config(
        lenses=[lc.LensParams(model="SIS", theta_E=0.85, gamma1=0.04, gamma2=-0.02,
                              light_model="SERSIC_ELLIPSE", light_amp=0.6,
                              light_R_sersic=0.9, light_n_sersic=4.0,
                              light_e1=0.15, light_e2=0.05)],
        sources=truth.sources, num_pix=truth.num_pix, delta_pix=truth.delta_pix,
    )
    lens_s, ll_s, src_s = _specs(start, free_lens=("theta_E",))
    res = ft.run_pso(start, data, lens_s, ll_s, src_s,
                     n_particles=25, n_iterations=80, n_restarts=2, polish=True)

    assert res.ok, res.error
    assert res.n_free == 1
    assert abs(res.config.lenses[0].theta_E - 1.10) < 0.03
    assert res.chi2_after < res.chi2_before
    assert res.improved()
    assert res.model is not None and res.residual is not None
    assert res.model.shape == data.image.shape
    # locked shear untouched
    assert res.config.lenses[0].gamma1 == pytest.approx(0.04)
    assert res.config.lenses[0].light_amp == pytest.approx(0.6)


# ------------------------------------------------------------------ previews
def test_render_image_matches_compute_and_is_image_only():
    """render_image must equal compute().image; it is the cheap path a fit uses."""
    cfg = _truth_config()
    assert np.allclose(lc.render_image(cfg), lc.compute(cfg).image)
    assert np.shape(lc.render_image(cfg)) == (cfg.num_pix, cfg.num_pix)


@pytest.mark.slow
def test_run_pso_emits_converging_previews():
    """The fit must report intermediate models so the user can watch it."""
    # PSO is fully driven by numpy's global RNG: fix the seed so this regression
    # test is deterministic (unseeded draws made it flaky ~1/3 of the time).
    np.random.seed(0)
    truth = _truth_config()
    data = _data_for(truth)
    start = lc.Config(
        lenses=[lc.LensParams(model="SIS", theta_E=0.85, gamma1=0.04, gamma2=-0.02,
                              light_model="SERSIC_ELLIPSE", light_amp=0.6,
                              light_R_sersic=0.9, light_n_sersic=4.0,
                              light_e1=0.15, light_e2=0.05)],
        sources=truth.sources, num_pix=truth.num_pix, delta_pix=truth.delta_pix,
    )
    lens_s, ll_s, src_s = _specs(start, free_lens=("theta_E",))

    seen = []
    res = ft.run_pso(
        start, data, lens_s, ll_s, src_s,
        n_particles=25, n_iterations=80, n_restarts=1, polish=True,
        # a tiny interval so the test sees plenty of previews
        preview=lambda it, total, chi2, img: seen.append((it, total, chi2, img)),
        preview_interval=0.0,
    )

    assert res.ok, res.error
    assert len(seen) >= 5, f"expected several previews, got {len(seen)}"
    for it, total, chi2, img in seen:
        assert 1 <= it <= total
        assert np.isfinite(chi2) and chi2 > 0
        assert img.shape == data.image.shape     # a renderable model each time
        assert np.isfinite(img).all()
    # The swarm's global best can only improve, so the previewed chi2 must be
    # monotonically non-increasing. (Strict decrease is not guaranteed: the
    # initial random swarm may already contain a near-perfect particle.)
    chi2s = [c for _, _, c, _ in seen]
    assert all(b <= a + 1e-6 for a, b in zip(chi2s, chi2s[1:])), chi2s
    # The fit itself must be an improvement on the starting model, and the final
    # answer must be at least as good as anything previewed.
    assert res.chi2_after < res.chi2_before
    assert res.chi2_after <= min(chi2s) + 1e-6

    # The parameter chain is recorded independently of the (throttled) previews:
    # one sample per swarm iteration, every free parameter present, chi2 of the
    # global best monotonically non-increasing. (The swarm may early-stop on
    # convergence, so only a minimum length is guaranteed.)
    assert len(res.chain_iter) >= 5, "expected at least several chain samples"
    assert len(res.chain_chi2) == len(res.chain_iter)
    assert "lens0.theta_E" in res.chain
    assert len(res.chain["lens0.theta_E"]) == len(res.chain_iter)
    assert all(np.isfinite(c) for c in res.chain_chi2)
    assert all(b <= a + 1e-6 for a, b in
               zip(res.chain_chi2, res.chain_chi2[1:]))
    # reduced chi² is the per-DoF goodness of fit (uses the app's ndof scale)
    assert np.isfinite(res.reduced_chi2)
    assert res.reduced_chi2 == pytest.approx(res.chi2_after / res.ndof)


def test_run_pso_early_stop_skips_remaining_restarts():
    """early_stop_reduced>0 halts once a restart hits the target reduced chi²."""
    np.random.seed(0)
    truth = _truth_config()
    data = _data_for(truth)
    start = lc.Config(
        lenses=[lc.LensParams(model="SIS", theta_E=0.85, gamma1=0.04, gamma2=-0.02,
                              light_model="SERSIC_ELLIPSE", light_amp=0.6,
                              light_R_sersic=0.9, light_n_sersic=4.0,
                              light_e1=0.15, light_e2=0.05)],
        sources=truth.sources, num_pix=truth.num_pix, delta_pix=truth.delta_pix,
    )
    lens_s, ll_s, src_s = _specs(start, free_lens=("theta_E",))

    res = ft.run_pso(
        start, data, lens_s, ll_s, src_s,
        n_particles=20, n_iterations=60, n_restarts=3, polish=True,
        early_stop_reduced=1.5,
    )
    assert res.ok, res.error
    n_restarts_run = sum(1 for m in res.log
                         if m.startswith("restart ") and ": chi2 = " in m)
    assert any("skipping remaining restarts" in m for m in res.log), res.log
    assert n_restarts_run < 3, f"expected some restarts skipped, ran {n_restarts_run}"
    # the reached solution is at least as good as the (reduced) target
    assert res.chi2_after <= 1.5 * max(res.ndof, 1)
    assert res.chi2_after < res.chi2_before

    # control: without early stop every restart runs
    res2 = ft.run_pso(
        start, data, lens_s, ll_s, src_s,
        n_particles=20, n_iterations=60, n_restarts=3, polish=True,
        early_stop_reduced=0.0,
    )
    assert res2.ok
    n2 = sum(1 for m in res2.log
             if m.startswith("restart ") and ": chi2 = " in m)
    assert n2 == 3, f"expected all 3 restarts without early stop, ran {n2}"


@pytest.mark.slow
def test_preview_failure_does_not_break_the_fit():
    """A broken preview callback must be swallowed, not abort the fit."""
    truth = _truth_config()
    data = _data_for(truth)
    start = lc.Config(
        lenses=[lc.LensParams(model="SIS", theta_E=0.85, gamma1=0.04, gamma2=-0.02,
                              light_model="SERSIC_ELLIPSE", light_amp=0.6,
                              light_R_sersic=0.9, light_n_sersic=4.0,
                              light_e1=0.15, light_e2=0.05)],
        sources=truth.sources, num_pix=truth.num_pix, delta_pix=truth.delta_pix,
    )
    lens_s, ll_s, src_s = _specs(start, free_lens=("theta_E",))

    def boom(*a, **k):
        raise RuntimeError("preview exploded")

    res = ft.run_pso(start, data, lens_s, ll_s, src_s,
                     n_particles=20, n_iterations=40, n_restarts=1,
                     preview=boom, preview_interval=0.0)
    assert res.ok, res.error
    assert res.chi2_after < res.chi2_before


@pytest.mark.slow
def test_preview_interval_limits_the_number_of_renders():
    """A larger interval must produce strictly fewer preview renders."""
    truth = _truth_config()
    data = _data_for(truth)
    start = lc.Config(
        lenses=[lc.LensParams(model="SIS", theta_E=0.85, gamma1=0.04, gamma2=-0.02,
                              light_model="SERSIC_ELLIPSE", light_amp=0.6,
                              light_R_sersic=0.9, light_n_sersic=4.0,
                              light_e1=0.15, light_e2=0.05)],
        sources=truth.sources, num_pix=truth.num_pix, delta_pix=truth.delta_pix,
    )
    lens_s, ll_s, src_s = _specs(start, free_lens=("theta_E",))

    def count(interval):
        n = [0]
        ft.run_pso(start, data, lens_s, ll_s, src_s,
                   n_particles=25, n_iterations=80, n_restarts=1, polish=False,
                   preview=lambda *a: n.__setitem__(0, n[0] + 1),
                   preview_interval=interval)
        return n[0]

    dense = count(0.0)        # every iteration
    sparse = count(1.0)       # at most ~1/s
    assert dense > sparse, f"dense={dense} sparse={sparse}"
    assert sparse >= 1        # still shows progress


@pytest.mark.slow
def test_previews_off_costs_nothing_and_skips_the_callback():
    """With preview disabled the fit must not call the renderer at all."""
    truth = _truth_config()
    data = _data_for(truth)
    start = lc.Config(
        lenses=[lc.LensParams(model="SIS", theta_E=0.85, gamma1=0.04, gamma2=-0.02,
                              light_model="SERSIC_ELLIPSE", light_amp=0.6,
                              light_R_sersic=0.9, light_n_sersic=4.0,
                              light_e1=0.15, light_e2=0.05)],
        sources=truth.sources, num_pix=truth.num_pix, delta_pix=truth.delta_pix,
    )
    lens_s, ll_s, src_s = _specs(start, free_lens=("theta_E",))
    res = ft.run_pso(start, data, lens_s, ll_s, src_s,
                     n_particles=20, n_iterations=50, n_restarts=1,
                     preview=None)          # what the worker passes when unchecked
    assert res.ok, res.error
    assert res.chi2_after < res.chi2_before


# ---------------------------------------------------------------- point sources

def _tiny_data(config):
    """A minimal FitData on the config grid (needed only by build_setup)."""
    num, delta = config.num_pix, config.delta_pix
    return fd.FitData(image=np.zeros((num, num)),
                      noise=np.ones((num, num)),
                      delta_pix=delta)


def test_point_source_entries_seed_image_plane():
    """A LENSED point source becomes a LENSED_POSITION (image-plane) fit entry
    seeded by the forward solve; UNLENSED stays a single on-sky point."""
    cfg = lc.Config(num_pix=40, lenses=[lc.LensParams(theta_E=1.0)],
                    point_sources=[lc.PointSourceParams(model="LENSED",
                                                        center_x=0.1, center_y=-0.1)])
    specs = [{"source_amp": (0.7, 0.02, 5.0, False)}]
    models, fm, init, sigma, fixed, lower, upper, free, fixed_names = \
        ft._point_source_entries(cfg, specs)
    assert models == ["LENSED_POSITION"] and fm == [True]
    ra_img = list(init[0]["ra_image"])
    lc_ = lc
    lm, kwl = lc_._get_lens_model(cfg.lenses, 1.5)
    from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
    x, y = LensEquationSolver(lm).findBrightImage(0.1, -0.1, kwl, numImages=4)
    assert len(ra_img) == len(x) and len(ra_img) >= 2
    # seeded image planes match the forward solve (order may differ)
    exp = sorted(zip(np.round(x, 4), np.round(y, 4)))
    got = sorted(zip(np.round(ra_img, 4), np.round(init[0]["dec_image"], 4)))
    for (gx, gy), (ex, ey) in zip(got, exp):
        assert gx == pytest.approx(ex, abs=1e-3)
        assert gy == pytest.approx(ey, abs=1e-3)
    assert "point0.source_amp" in free   # amplitude is linear/free by default
    assert "point0.ra_image[0]" in free

    # UNLENSED: single fixed on-sky point, point_amp entry
    cfg2 = lc.Config(num_pix=40, point_sources=[lc.PointSourceParams(
        model="UNLENSED", point_amp=0.5, center_x=0.3, center_y=-0.2)])
    models2, fm2, init2, *_ = ft._point_source_entries(cfg2, [{}])
    assert models2 == ["UNLENSED"] and fm2 == [False]
    assert list(init2[0]["ra_image"]) == [0.3]
    assert list(init2[0]["dec_image"]) == [-0.2]
    assert init2[0]["point_amp"] == 0.5


def test_point_source_roundtrip_recovers_source_plane():
    """The image-plane fit result must map back to the app's source-plane model
    such that re-rendering reproduces the fitted image positions."""
    cfg = lc.Config(num_pix=40, lenses=[lc.LensParams(theta_E=1.0)],
                    point_sources=[lc.PointSourceParams(model="LENSED",
                                                        source_amp=0.7,
                                                        center_x=0.1, center_y=-0.1)])
    lm, kwl = lc._get_lens_model(cfg.lenses, 1.5)
    kwargs_result = {
        "kwargs_lens": kwl,
        "kwargs_ps": [{"ra_image": [0.807, -0.607],
                       "dec_image": [-0.807, 0.607],
                       "source_amp": 0.71}],
    }
    out = ft.model_config_from_result(cfg, kwargs_result, 40)
    ps = out.point_sources[0]
    assert ps.model == "LENSED"
    assert ps.center_x == pytest.approx(0.1, abs=2e-3)
    assert ps.center_y == pytest.approx(-0.1, abs=2e-3)
    assert ps.source_amp == pytest.approx(0.71, rel=1e-2)

    # forward-render the recovered config -> same two images
    lm2, kwl2 = lc._get_lens_model(out.lenses, ps.position_and_redshift(out)[2])
    from lenstronomy.PointSource.point_source import PointSource
    rt = PointSource(point_source_type_list=["SOURCE_POSITION"],
                     lens_model=lm2, fixed_magnification_list=[True])
    ra_i, dec_i, _ = rt.point_source_list([ps.lens_source_kwargs(out)[1]],
                                          kwargs_lens=kwl2)
    assert sorted(np.round(ra_i, 2)) == pytest.approx([-0.61, 0.81], abs=2e-2)
    assert sorted(np.round(dec_i, 2)) == pytest.approx([-0.81, 0.61], abs=2e-2)


def test_build_setup_includes_point_sources():
    cfg = lc.Config(num_pix=20, lenses=[lc.LensParams()],
                    point_sources=[lc.PointSourceParams(model="LENSED")])
    kw_data, kw_model, kw_params, free_names, fixed_names = ft.build_setup(
        cfg, _tiny_data(cfg), [], [], [], [{}])
    assert kw_model["point_source_model_list"] == ["LENSED_POSITION"]
    assert kw_model["fixed_magnification_list"] == [True]
    assert "point_source_model" in kw_params
    assert any(n.startswith("point0.") for n in free_names)

    # no point sources -> keys absent, old behaviour unchanged
    cfg_none = lc.Config(num_pix=20, lenses=[lc.LensParams()])
    kw_data, kw_model, kw_params, *_ = ft.build_setup(
        cfg_none, _tiny_data(cfg_none), [{"theta_E": (1.0, 0.1, 5.0, False)}],
        [], [], [])
    assert "point_source_model_list" not in kw_model
    assert "point_source_model" not in kw_params
