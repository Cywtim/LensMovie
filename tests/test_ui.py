"""Offscreen smoke tests for the Qt UI (no real display needed)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_lenses_panel_produces_config(qapp):
    from app.controls import LensesPanel

    panel = LensesPanel()
    lenses = panel.lens_list()
    assert len(lenses) == 1
    assert lenses[0].model in ("SIS", "SIE", "PEMD")
    assert lenses[0].redshift > 0
    panel.deleteLater()


def test_sources_panel_produces_config(qapp):
    from app.controls import SourcesPanel

    panel = SourcesPanel()
    sources = panel.source_list()
    assert len(sources) == 1
    assert sources[0].redshift > 0
    panel.deleteLater()


def test_add_remove_lens_src_keeps_at_least_one(qapp):
    from app.controls import LensesPanel, SourcesPanel

    lp = LensesPanel()
    sp = SourcesPanel()
    lp._add_card(lp._cards[0].__class__())
    lp._add_card(lp._cards[0].__class__())
    sp._add_card(sp._cards[0].__class__())
    assert len(lp.lens_list()) == 3
    assert len(sp.source_list()) == 2

    for card in list(lp._cards):
        lp._remove_card(card)
    assert len(lp.lens_list()) == 1
    for card in list(sp._cards):
        sp._remove_card(card)
    assert len(sp.source_list()) == 1
    lp.deleteLater()
    sp.deleteLater()


def test_display_bar_defaults(qapp):
    from app.controls import DisplayBar

    bar = DisplayBar()
    d = bar.display()
    assert d["num_pix"] == 150
    assert d["colormap"] in ("magma", "viridis", "plasma", "inferno", "gray", "turbo")
    assert d["stretch"] in ("log", "linear")
    bar.deleteLater()


def test_main_window_constructs_and_renders(qapp):
    """Full window should build (with the 2x3 grid) and render once."""
    from app.main_window import MainWindow

    win = MainWindow()
    assert win._render_ok is True
    assert win.fermat_canvas is not None
    assert win.image_canvas is not None
    assert win.delay_canvas is not None
    assert win.curves_canvas is not None
    assert win.lenses_panel is not None
    assert win.sources_panel is not None
    win.close()
    win.deleteLater()


def test_canvases_expand_to_fill_cell(qapp):
    """Matplotlib canvases must expand to fill their grid cell on resize."""
    from PyQt5.QtWidgets import QSizePolicy

    from app.main_window import MainWindow

    win = MainWindow()
    for c in (win.fermat_canvas, win.image_canvas, win.delay_canvas, win.curves_canvas):
        sp = c.sizePolicy()
        assert sp.horizontalPolicy() in (QSizePolicy.Ignored, QSizePolicy.Expanding)
        assert sp.verticalPolicy() in (QSizePolicy.Ignored, QSizePolicy.Expanding)
    win.close()
    win.deleteLater()


def test_3d_toggle_exists_and_defaults_on(qapp):
    from app.controls import DisplayBar

    bar = DisplayBar()
    assert bar.three_d_enabled() is True
    bar._three_d.setChecked(False)
    assert bar.three_d_enabled() is False
    bar.deleteLater()


def test_3d_toggle_keeps_black_background_and_skips_rebuild(qapp):
    """Turning 3D off hides only the rendering canvas: the black area stays."""
    from app.main_window import MainWindow

    win = MainWindow()
    if win.scene3d is None:
        pytest.skip("3D scene unavailable in this environment")

    win.show()
    qapp.processEvents()
    assert win.display_bar.three_d_enabled() is True
    assert win._3d_native.isVisible() is True

    # Turn off: only the GL canvas is hidden; the black background area remains
    # (the layout must not reflow) and the 2D render still succeeds.
    win.display_bar._three_d.setChecked(False)
    qapp.processEvents()
    assert win._3d_native.isVisible() is False
    assert win._3d_wrap.isVisible() is True
    assert win._3d_wrap.height() == win._3d_height
    win._rerender()
    assert win._render_ok is True

    # Turn back on.
    win.display_bar._three_d.setChecked(True)
    qapp.processEvents()
    assert win._3d_native.isVisible() is True
    win._rerender()
    assert win._render_ok is True
    win.close()
    win.deleteLater()


def test_3d_shares_top_row_with_square_external_panel(qapp):
    """Top row = stretchable 3D (fixed height) + a fixed square external panel."""
    from PyQt5.QtWidgets import QSizePolicy

    from app.main_window import MainWindow

    win = MainWindow()
    if win.scene3d is None:
        pytest.skip("3D scene unavailable in this environment")
    win.resize(1500, 1050)
    win.show()
    qapp.processEvents()

    native = win._3d_native
    assert native.sizePolicy().horizontalPolicy() == QSizePolicy.Expanding
    assert native.sizePolicy().verticalPolicy() == QSizePolicy.Fixed
    assert native.height() == win._3d_height

    # The external panel is square and sits at the right of the same row.
    assert win.ext_panel.width() == win.ext_panel.height() == win._3d_height
    # Both are in the top row: compare positions of sibling widgets (the 3D
    # native lives inside its wrapper, so use the wrapper).
    assert win.ext_panel.x() > win._3d_wrap.x()
    assert win.ext_panel.y() == win._3d_wrap.y()
    assert win.ext_panel.height() == win._3d_wrap.height()
    # The 3D takes the remaining width (window minus the square panel).
    assert native.width() > 0.6 * win.width()
    assert win.external_canvas is not None
    win.close()
    win.deleteLater()


# --------------------------------------------------------------- fix / lock
def test_fix_locks_value_and_disables_slider(qapp):
    from app.controls import LensesPanel

    panel = LensesPanel()
    card = panel._cards[0]
    row = card.sliders["theta_E"]
    original = row.value()

    assert row.is_fixed() is False
    row.set_fixed(True)                      # fixing may be programmatic
    assert row.is_fixed() is True
    assert row._slider.isEnabled() is False  # user cannot drag it
    assert row._lock_btn.text() == "🔒"

    # A fixed parameter keeps its value against any programmatic change, even a
    # direct setValue on the underlying slider widget.
    row.set_value(original + 1.0)
    assert row.value() == original
    row._slider.setValue(row._to_int(original + 1.5))
    assert row.value() == original
    assert card.to_params().theta_E == original

    panel.deleteLater()


def test_unfix_requires_user_action(qapp):
    from app.controls import LensesPanel

    panel = LensesPanel()
    row = panel._cards[0].sliders["theta_E"]
    row.set_fixed(True)

    # Programmatic unfix must be refused.
    with pytest.raises(PermissionError):
        row.set_fixed(False)
    assert row.is_fixed() is True

    # The user's click (user=True) releases it.
    row.set_fixed(False, user=True)
    assert row.is_fixed() is False
    assert row._slider.isEnabled() is True
    assert row._lock_btn.text() == "🔓"

    # Value can be changed again once released.
    row.set_value(2.0)
    assert abs(row.value() - 2.0) < 0.01
    panel.deleteLater()


def test_lock_button_click_unfixes(qapp):
    """Clicking the lock button is the sanctioned way to unfix."""
    from app.controls import LensesPanel

    panel = LensesPanel()
    row = panel._cards[0].sliders["theta_E"]
    row._lock_btn.setChecked(True)           # simulates a user click
    assert row.is_fixed() is True
    row._lock_btn.setChecked(False)          # user clicks again to unfix
    assert row.is_fixed() is False
    panel.deleteLater()


def test_fixed_params_reported_for_lenses_and_sources(qapp):
    from app.controls import LensesPanel, SourcesPanel

    lp = LensesPanel()
    sp = SourcesPanel()
    lp._cards[0].fix_param("theta_E")
    lp._cards[0].fix_param("gamma1")
    sp._cards[0].fix_param("R_sersic")

    assert lp.fixed_params()[0] == {"theta_E", "gamma1"}
    assert sp.fixed_params()[0] == {"R_sersic"}

    # unfix_all needs a user action too
    with pytest.raises(PermissionError):
        lp.unfix_all()
    lp.unfix_all(user=True)
    assert lp.fixed_params()[0] == set()
    lp.deleteLater()
    sp.deleteLater()


def test_lock_survives_a_rerender(qapp):
    """Rendering must not clear a user's locks."""
    from app.main_window import MainWindow

    win = MainWindow()
    win.lenses_panel._cards[0].fix_param("theta_E")
    before = win.lenses_panel._cards[0].sliders["theta_E"].value()
    win._rerender()
    win._rerender()
    assert win._render_ok is True
    row = win.lenses_panel._cards[0].sliders["theta_E"]
    assert row.is_fixed() is True
    assert row.value() == before
    win.close()
    win.deleteLater()


# ------------------------------------------------- pixel scale / PSF / data
def test_display_bar_exposes_pixel_scale_and_psf(qapp):
    from app.controls import DisplayBar

    bar = DisplayBar()
    d = bar.display()
    assert d["delta_pix"] == 0.05
    assert d["psf_fwhm"] == 0.0
    bar._delta_pix.setValue(0.12)
    bar._psf_fwhm.setValue(0.6)
    d = bar.display()
    assert abs(d["delta_pix"] - 0.12) < 1e-9
    assert abs(d["psf_fwhm"] - 0.6) < 1e-9
    bar.deleteLater()


def test_config_uses_ui_pixel_scale_and_psf(qapp):
    from app.main_window import MainWindow

    win = MainWindow()
    win.display_bar._delta_pix.setValue(0.08)
    cfg = win._build_config()
    assert abs(cfg.delta_pix - 0.08) < 1e-9
    # FWHM 0 -> delta PSF
    assert np.asarray(win.current_psf_kernel()).shape == (1, 1)
    win.display_bar._psf_fwhm.setValue(0.5)
    assert np.asarray(win.current_psf_kernel()).shape[0] > 1
    win._rerender()
    assert win._render_ok is True
    win.close()
    win.deleteLater()


def test_prepare_fit_data_puts_external_image_on_model_grid(qapp):
    from app.main_window import MainWindow

    win = MainWindow()
    win.display_bar._numpix.setValue(80)
    win.display_bar._delta_pix.setValue(0.05)
    win._external_array = np.random.RandomState(0).rand(200, 200)
    win._noise_array = np.full((200, 200), 0.05)
    win._ext_mode.setCurrentText("on model grid")
    win._rerender()

    fd = win._fit_data
    assert fd is not None
    assert fd.image.shape == (80, 80)
    assert fd.noise.shape == (80, 80)
    # The data grid matches the model grid, ready for a fit.
    assert win._build_config().num_pix == 80
    win.close()
    win.deleteLater()


def test_prepare_fit_data_returns_none_without_image(qapp):
    from app.main_window import MainWindow

    win = MainWindow()
    assert win.prepare_fit_data() is None
    win.close()
    win.deleteLater()


def test_load_aux_noise_mask_psf(qapp, tmp_path, monkeypatch):
    """Noise/mask/PSF files load, and a PSF kernel is normalised."""
    from app import main_window as mw

    win = mw.MainWindow()
    shape = (40, 40)
    noise = np.full(shape, 0.07)
    mask = np.ones(shape, bool)
    mask[5, 5] = False
    kernel = np.ones((5, 5)) * 3.0            # deliberately not normalised
    np.save(tmp_path / "noise.npy", noise)
    np.save(tmp_path / "mask.npy", mask.astype(float))
    np.save(tmp_path / "psf.npy", kernel)

    for kind, fname in (("noise", "noise.npy"), ("mask", "mask.npy"),
                        ("psf", "psf.npy")):
        monkeypatch.setattr(mw.QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: (str(tmp_path / fname), "")))
        win._load_aux(kind)

    assert win._noise_array is not None and win._noise_array.shape == shape
    assert win._mask_array is not None
    assert win._mask_array[5, 5] == False  # noqa: E712
    # The PSF kernel must be normalised so it cannot rescale brightness.
    assert abs(np.asarray(win.current_psf_kernel()).sum() - 1.0) < 1e-9
    win._rerender()
    assert win._render_ok is True
    win.close()
    win.deleteLater()


# --------------------------------------------------- lens light / sky in the UI
def test_lens_card_exposes_light_model_and_disables_sliders(qapp):
    from app.controls import LensesPanel

    panel = LensesPanel()
    card = panel._cards[0]
    items = [card._light_combo.itemText(i) for i in range(card._light_combo.count())]
    assert items == ["NONE", "SERSIC_ELLIPSE", "SERSIC", "GAUSSIAN_ELLIPSE", "GAUSSIAN"]
    # Default: no deflector light, so its sliders are disabled.
    assert card.light_model() == "NONE"
    assert card.sliders["light_amp"].isEnabled() is False
    # Turning light on enables the sliders and reaches the params.
    card._light_combo.setCurrentText("SERSIC_ELLIPSE")
    assert card.sliders["light_amp"].isEnabled() is True
    card.sliders["light_amp"].set_value(0.4)
    p = card.to_params()
    assert p.light_model == "SERSIC_ELLIPSE"
    assert abs(p.light_amp - 0.4) < 0.01
    panel.deleteLater()


def test_lens_light_reaches_the_rendered_model(qapp):
    from app.main_window import MainWindow

    win = MainWindow()
    card = win.lenses_panel._cards[0]
    before = win._build_config()
    assert before.lenses[0].has_light() is False

    card._light_combo.setCurrentText("SERSIC_ELLIPSE")
    card.sliders["light_amp"].set_value(0.5)
    win._rerender()
    cfg = win._build_config()
    assert cfg.lenses[0].has_light() is True
    assert win._render_ok is True

    from app import lensing_calc as lc
    assert not np.allclose(lc.compute(before).image, lc.compute(cfg).image)
    win.close()
    win.deleteLater()


def test_sky_control_reaches_config(qapp):
    from app.main_window import MainWindow

    win = MainWindow()
    win.display_bar._sky.setValue(0.004)
    assert abs(win._build_config().sky_amp - 0.004) < 1e-9
    win._rerender()
    assert win._render_ok is True
    win.close()
    win.deleteLater()


# ------------------------------------------------------------ fitting in the GUI
def test_fit_bar_settings_and_running_state(qapp):
    from app.controls import FitBar

    bar = FitBar()
    s = bar.settings()
    assert {"n_particles", "n_iterations", "n_restarts"} <= set(s)
    assert bar._fit_btn.isEnabled() is True
    bar.set_running(True)
    assert bar._fit_btn.isEnabled() is False
    assert bar._cancel_btn.isEnabled() is True
    bar.set_running(False)
    assert bar._fit_btn.isEnabled() is True
    bar.deleteLater()


def test_fit_refuses_without_data(qapp):
    from app.main_window import MainWindow

    win = MainWindow()
    win._start_fit()          # no image loaded
    assert "load an image" in win.fit_bar._status.text().lower()
    win.close()
    win.deleteLater()


def test_fit_refuses_without_noise(qapp):
    from app.main_window import MainWindow

    win = MainWindow()
    win._external_array = np.random.RandomState(0).rand(60, 60)
    win._start_fit()          # image but no noise map
    assert "noise" in win.fit_bar._status.text().lower()
    win.close()
    win.deleteLater()


def test_external_display_modes_without_fit(qapp):
    from app.main_window import MainWindow

    win = MainWindow()
    win._external_array = np.random.RandomState(0).rand(60, 60)
    for mode in ("data", "on model grid", "best-fit model", "residual"):
        win._ext_mode.setCurrentText(mode)
        win._rerender()
        assert win._render_ok is True     # missing fit must not crash the view
    win.close()
    win.deleteLater()


def test_apply_fitted_config_writes_back_but_respects_locks(qapp):
    """Fitted values reach the sliders; fixed ones are not overwritten."""
    from app import lensing_calc as lc
    from app.main_window import MainWindow

    win = MainWindow()
    card = win.lenses_panel._cards[0]
    card.sliders["theta_E"].set_value(0.9)
    card.sliders["gamma1"].set_value(0.01)
    card.sliders["gamma1"].set_fixed(True)          # lock gamma1

    fitted = lc.Config(
        lenses=[lc.LensParams(model="SIS", theta_E=1.42, gamma1=0.25,
                              light_model="NONE")],
        sources=win.sources_panel.source_list(),
        num_pix=150, delta_pix=0.05,
    )
    win._apply_fitted_config(fitted)

    # Sliders are quantised (1000 steps across their range), so allow one step.
    assert abs(card.sliders["theta_E"].value() - 1.42) < 0.005   # free -> updated
    assert abs(card.sliders["gamma1"].value() - 0.01) < 0.002    # locked -> kept
    win.close()
    win.deleteLater()


@pytest.mark.slow
def test_gui_fit_closed_loop(qapp):
    """Full loop: load data, lock all but one parameter, fit, write back."""
    from PyQt5.QtCore import QEventLoop, QTimer

    from app import lensing_calc as lc
    from app.main_window import MainWindow

    win = MainWindow()
    truth = lc.Config(
        lenses=[lc.LensParams(model="SIS", theta_E=1.10,
                              light_model="SERSIC_ELLIPSE", light_amp=0.6,
                              light_R_sersic=0.9, light_n_sersic=4.0,
                              light_e1=0.15, light_e2=0.05)],
        sources=[lc.SourceParams(amp=1.0, R_sersic=0.12, n_sersic=3.0,
                                 e1=0.1, e2=-0.1, center_x=0.08, center_y=-0.06)],
        num_pix=60, delta_pix=0.05,
    )
    mock = lc.compute(truth).image
    sigma = 0.002
    win._external_array = mock + np.random.RandomState(3).normal(0, sigma, mock.shape)
    win._noise_array = np.full(mock.shape, sigma)

    win.display_bar._numpix.setValue(60)
    win.display_bar._delta_pix.setValue(0.05)
    card = win.lenses_panel._cards[0]
    card._light_combo.setCurrentText("SERSIC_ELLIPSE")
    for name, val in (("light_amp", 0.6), ("light_R_sersic", 0.9),
                      ("light_n_sersic", 4.0), ("light_e1", 0.15), ("light_e2", 0.05)):
        card.sliders[name].set_value(val)
    card.sliders["theta_E"].set_value(0.85)
    scard = win.sources_panel._cards[0]
    for name, val in (("amp", 1.0), ("R_sersic", 0.12), ("n_sersic", 3.0),
                      ("e1", 0.1), ("e2", -0.1), ("center_x", 0.08), ("center_y", -0.06)):
        scard.sliders[name].set_value(val)

    for name, row in card.sliders.items():
        if name != "theta_E":
            row.set_fixed(True)
    for row in scard.sliders.values():
        row.set_fixed(True)

    win.fit_bar._particles.setValue(25)
    win.fit_bar._iterations.setValue(80)
    win.fit_bar._restarts.setValue(2)

    loop = QEventLoop()
    win._start_fit()
    win._fit_worker.finished_ok.connect(lambda *a: QTimer.singleShot(20, loop.quit))
    win._fit_worker.failed.connect(lambda *a: QTimer.singleShot(20, loop.quit))
    QTimer.singleShot(120000, loop.quit)
    loop.exec_()
    qapp.processEvents()

    assert win._fit_result is not None, win.fit_bar._status.text()
    r = win._fit_result
    assert r.free_names == ["lens0.theta_E"]
    assert abs(card.sliders["theta_E"].value() - 1.10) < 0.05
    assert card.sliders["gamma1"].is_fixed() is True
    assert abs(card.sliders["light_amp"].value() - 0.6) < 0.05
    assert r.chi2_after < r.chi2_before
    win.close()
    win.deleteLater()


# ------------------------------------------------- panel x-axis alignment
def test_image_and_curves_panels_share_the_same_x_axis(qapp):
    """The Lens image and the Critical-curve panel sit in the same column, so a
    given sky coordinate must land at the same canvas x in both.

    Regression: the curves panel used to auto-scale to its own curve extents
    (a much smaller window than the image's field of view) and the image panel's
    colourbar stole axes width via tight_layout, so the two never lined up.
    """
    from app.main_window import MainWindow

    win = MainWindow()
    win.resize(1600, 1080)
    win.show()
    qapp.processEvents()
    win._rerender()
    qapp.processEvents()

    ic, cc = win.image_canvas, win.curves_canvas
    ic.figure.canvas.draw()
    cc.figure.canvas.draw()

    # Same field of view.
    assert np.allclose(ic._ax.get_xlim(), cc._ax.get_xlim())
    assert np.allclose(ic._ax.get_ylim(), cc._ax.get_ylim())
    # Identical *requested* axes rectangle (before the square-aspect adjustment
    # trims a hair when the two widgets differ by a pixel in height).
    assert np.allclose(ic._ax.get_position(original=True).bounds,
                       cc._ax.get_position(original=True).bounds, atol=1e-9)

    # The same sky x maps to (very nearly) the same canvas x.
    def canvas_x(canvas, sky):
        return float(canvas._ax.transData.transform((sky, 0.0))[0])

    for sky in (-3.7, -1.0, 0.0, 1.0, 3.7):
        assert abs(canvas_x(ic, sky) - canvas_x(cc, sky)) < 1.0
    win.close()
    win.deleteLater()


def test_all_panels_share_one_axes_rectangle(qapp):
    """A colourbar must not change a panel's axes geometry."""
    from app.main_window import MainWindow

    win = MainWindow()
    # A colourbar must not shift/shrink the main axes: the requested rectangle is
    # the same for every canvas, whether or not it has one.
    ref = win.image_canvas._ax.get_position(original=True).bounds
    for c in (win.fermat_canvas, win.image_canvas,
              win.delay_canvas, win.curves_canvas):
        assert np.allclose(c._ax.get_position(original=True).bounds, ref, atol=1e-9)
    win.close()
    win.deleteLater()


def test_curves_panel_uses_the_grid_field_of_view(qapp):
    """update_curves must adopt the model grid's FOV, not the curve extents."""
    from app.plotting import CurvesCanvas

    cv = CurvesCanvas()
    # a tiny critical curve inside a large field
    theta = np.linspace(0, 2 * np.pi, 50)
    r = 0.3
    cv.update_curves(r * np.cos(theta), r * np.sin(theta),
                     r * np.cos(theta), r * np.sin(theta),
                     num_pix=100, delta_pix=0.05)
    lo, hi = cv._ax.get_xlim()
    assert np.isclose(lo, -2.475) and np.isclose(hi, 2.475)   # grid FOV, not ±0.3
    cv.deleteLater()
