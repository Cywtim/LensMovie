"""Export helpers: parameter-chain value resolution + file writers."""

import csv

import numpy as np
import pytest

from app import export_fit as exp
from app import fit_data as fd
from app import fitting as ft
from app import lensing_calc as lc


def _roundtripped(kw, free_names, num_pix=40):
    """A config built from ``kw`` via the usual fit round-trip, so lens/source
    values reflect the fitted kwargs rather than the starting config."""
    cfg = lc.Config(
        lenses=[lc.LensParams(model="SIE", theta_E=1.0, e1=0.1, e2=0.02,
                              gamma1=0.05, gamma2=0.01,
                              light_model="SERSIC_ELLIPSE", light_amp=0.7,
                              light_R_sersic=0.9, light_n_sersic=4.0)],
        sources=[lc.SourceParams(amp=1.2, R_sersic=0.3, n_sersic=2.0,
                                 center_x=0.2, center_y=-0.1)],
        point_sources=[lc.PointSourceParams(model="LENSED", source_amp=0.5,
                                            center_x=-0.1, center_y=0.1)],
    )
    cfg_i = ft.model_config_from_result(cfg, kw, num_pix)
    return ft._chain_values(cfg_i, kw, free_names)


def test_chain_resolver_reads_all_component_types():
    """Every free-name shape resolves to the fitted numeric value."""
    kw = {
        "kwargs_lens": [
            {"theta_E": 1.1, "e1": 0.11, "e2": 0.03,
             "center_x": 0.0, "center_y": 0.0},
            {"gamma1": 0.055, "gamma2": 0.015},     # separate SHEAR entry
        ],
        "kwargs_lens_light": [
            {"amp": 0.71, "R_sersic": 0.91, "n_sersic": 4.1,
             "center_x": 0.0, "center_y": 0.0},
        ],
        "kwargs_source": [
            {"amp": 1.21, "R_sersic": 0.31, "n_sersic": 2.1,
             "center_x": 0.21, "center_y": -0.11},
        ],
        "kwargs_ps": [
            {"ra_image": [0.8, -0.6], "dec_image": [-0.8, 0.6],
             "source_amp": 0.5},
        ],
    }
    free_names = [
        "lens0.theta_E", "lens0.e1", "lens0.gamma1", "lens0.gamma2",
        "lens_light0.amp", "lens_light0.R_sersic",
        "source0.amp", "source0.center_x", "source0.center_y",
        "point0.source_amp", "point0.ra_image[0]", "point0.dec_image[0]",
    ]
    vals = _roundtripped(kw, free_names)
    assert vals["lens0.theta_E"] == pytest.approx(1.1)
    assert vals["lens0.e1"] == pytest.approx(0.11)
    assert vals["lens0.gamma1"] == pytest.approx(0.055)   # shear entry
    assert vals["lens0.gamma2"] == pytest.approx(0.015)
    assert vals["lens_light0.amp"] == pytest.approx(0.71)
    assert vals["lens_light0.R_sersic"] == pytest.approx(0.91)
    assert vals["source0.amp"] == pytest.approx(1.21)
    assert vals["source0.center_x"] == pytest.approx(0.21)
    assert vals["source0.center_y"] == pytest.approx(-0.11)
    assert vals["point0.source_amp"] == pytest.approx(0.5, rel=1e-2)
    assert vals["point0.ra_image[0]"] == pytest.approx(0.8)
    assert vals["point0.dec_image[0]"] == pytest.approx(-0.8)


def test_chain_resolver_nans_on_bad_names():
    vals = _roundtripped({"kwargs_ps": [{"ra_image": []}]},
                         ["lens9.missing", "point0.ra_image[3]"])
    assert np.isnan(vals["lens9.missing"])
    assert np.isnan(vals["point0.ra_image[3]"])


def _result_with_chain():
    return ft.FitResult(
        ok=True,
        model=np.ones((4, 4)), residual=np.zeros((4, 4)),
        chi2_before=90.0, chi2_after=2.5, n_free=2, ndof=10,
        reduced_chi2=2.5 / 10,
        chain_iter=[1, 2, 3], chain_chi2=[9.0, 4.5, 2.5],
        chain={"lens0.theta_E": [0.6, 0.63, 0.64],
               "lens_light0.amp": [0.50, 0.51, 0.52]},
        free_names=["lens0.theta_E", "lens_light0.amp"],
    )


def test_save_chain_csv_roundtrip(tmp_path):
    res = _result_with_chain()
    p = exp.save_chain_csv(str(tmp_path / "c.csv"), res)
    with open(p, newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["iteration", "chi2", "lens0.theta_E", "lens_light0.amp"]
    assert rows[1][:3] == ["1", "9", "0.6"]
    assert rows[3][:3] == ["3", "2.5", "0.64"]
    assert len(rows) == 4                       # header + 3 iterations


def test_save_report_and_trajectory_pngs(tmp_path):
    data = fd.FitData(image=np.arange(16, dtype=float).reshape(4, 4),
                      noise=np.ones((4, 4)), delta_pix=0.1)
    res = _result_with_chain()
    report = exp.save_fit_report_png(str(tmp_path / "r.png"), data, res)
    traj = exp.save_chain_png(str(tmp_path / "t.png"), res)
    assert tmp_path.joinpath("r.png").stat().st_size > 1000
    assert tmp_path.joinpath("t.png").stat().st_size > 1000
    assert report.endswith(".png") and traj.endswith(".png")


def test_save_all_writes_three_files(tmp_path):
    data = fd.FitData(image=np.zeros((4, 4)), noise=np.ones((4, 4)),
                      delta_pix=0.1)
    res = _result_with_chain()
    out = exp.save_all(str(tmp_path / "fit"), data, res)
    assert set(out) == {"report", "chain_csv", "chain_png"}
    for p in out.values():
        assert tmp_path.joinpath(p).name and tmp_path.joinpath(p).stat().st_size > 0


def test_report_falls_back_to_three_panels_without_noise(tmp_path):
    """Without a noise map the χ² panel has no σ to divide by, so the report
    degrades gracefully back to data | model | residual."""
    data = fd.FitData(image=np.arange(16, dtype=float).reshape(4, 4),
                      delta_pix=0.1)                       # no noise map
    res = _result_with_chain()
    p = exp.save_fit_report_png(str(tmp_path / "r_nonoise.png"), data, res)
    assert tmp_path.joinpath("r_nonoise.png").stat().st_size > 1000


def test_report_includes_chi2_panel_and_diagnostic(tmp_path, monkeypatch):
    """With a noise map the report grows to 4 images (including the per-pixel
    χ² panel) and the title carries the noise-consistency diagnostic."""
    import matplotlib.pyplot as plt

    real_subplots = plt.subplots
    captured = {}

    def spy_subplots(nrows, ncols, *a, **k):
        fig, axs = real_subplots(nrows, ncols, *a, **k)
        captured["nimages"] = len(axs)
        orig = fig.suptitle
        fig.suptitle = lambda t, *aa, **kk: (captured.__setitem__("title", t)
                                             or orig(t, *aa, **kk))
        return fig, axs

    monkeypatch.setattr("matplotlib.pyplot.subplots", spy_subplots)

    data = fd.FitData(image=np.arange(16, dtype=float).reshape(4, 4),
                      noise=np.ones((4, 4)), delta_pix=0.1)
    model = np.ones((4, 4))
    res = ft.FitResult(ok=True, model=model, residual=data.image - model,
                       chi2_before=90.0, chi2_after=160.0, n_free=2, ndof=10,
                       reduced_chi2=4.0)
    exp.save_fit_report_png(str(tmp_path / "r4.png"), data, res)

    assert captured["nimages"] == 4
    assert "χ²" in captured["title"] and "PSF" in captured["title"] or "noise" \
        in captured["title"]
    assert tmp_path.joinpath("r4.png").stat().st_size > 1000


def test_build_chain_figure_layout():
    """build_chain_figure yields χ² on top and one subplot per free parameter —
    the figure the PNG exporter and the GUI preview both render from."""
    res = _result_with_chain()
    fig = exp.build_chain_figure(res)
    try:
        assert len(fig.axes) == 3                     # 1 χ² + 2 free params
        assert len(fig.axes[0].lines) == 1            # the χ² trajectory
        assert len(fig.axes[1].lines) == 1 and len(fig.axes[2].lines) == 1
        fig.canvas.draw()
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
