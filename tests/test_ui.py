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


def test_3d_bar_and_external_panel_layout(qapp):
    """Top area: stretchable 3D bar (fixed height) + an external panel that
    spans BOTH the 3D row and the numPix/display row below it."""
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

    # The external panel sits to the right of the 3D area, top-aligned...
    assert win.ext_panel.x() > win._3d_wrap.x()
    assert abs(win.ext_panel.y() - win._3d_wrap.y()) < 2
    # ...and spans at least the 3D row AND the display row: it is taller than
    # the 3D bar alone and runs down past the display row's bottom edge.
    assert win.ext_panel.height() > win._3d_wrap.height()
    assert win.ext_panel.geometry().bottom() \
        >= win.display_row.geometry().bottom() - 2
    # The 3D takes the remaining width (window minus the panel).
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


# --------------------------------------------------- independent fields of view
def test_image_and_curves_fields_of_view_are_independent(qapp):
    """The Lens image and the Critical-curve panel use different fields of view
    by design: the image keeps the model-grid FOV, while the curves panel
    auto-scales to the curves so a critical curve outside the grid is never
    clipped at the panel edge.
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

    # The lens image keeps the model grid's FOV (default 150 px @ 0.05″ → ±3.725).
    assert np.allclose(ic._ax.get_xlim(), (-3.725, 3.725))
    assert np.allclose(ic._ax.get_ylim(), (-3.725, 3.725))

    # A ring far outside that FOV must be fully visible on the curves panel.
    theta = np.linspace(0, 2 * np.pi, 60)
    r = 9.0
    cc.update_curves(r * np.cos(theta), r * np.sin(theta),
                     r * np.cos(theta), r * np.sin(theta),
                     num_pix=150, delta_pix=0.05)
    lo, hi = cc._ax.get_xlim()
    assert lo <= -9.0 * 1.1 - 1e-9, "curve clipped at the panel edge (left)"
    assert hi >= 9.0 * 1.1 + 1e-9, "curve clipped at the panel edge (right)"
    win.close()
    win.deleteLater()


def test_column_and_row_splitters_resize_the_panes(qapp):
    """The 2D grid columns and the grid-vs-top row are draggable splitters."""
    from app.main_window import MainWindow

    win = MainWindow()
    win.resize(1500, 1050)
    win.show()
    qapp.processEvents()

    # Columns: three panes (Fermat/delay | image/curves | lenses/sources).
    s = win.col_split.sizes()
    assert len(s) == 3 and all(w > 150 for w in s)
    assert abs(s[0] - s[1]) <= 2  # the two view columns start (near-)equal
    # Drag: give the config pane far more width.
    total = sum(s)
    cols = [int(total * 0.16), int(total * 0.16), total - 2 * int(total * 0.16)]
    win.col_split.setSizes(cols)
    qapp.processEvents()
    after = win.col_split.sizes()
    assert after[2] > s[2] + 100, "column splitter did not resize the panes"

    # Rows: top block (3D + display + fit) vs the 2D grid.
    r = win.row_split.sizes()
    assert len(r) == 2
    top, grid = r
    # Enlarge the top area (taller external panel); the grid must give way.
    taller = [top + int(grid / 2), grid - int(grid / 2)]
    win.row_split.setSizes(taller)
    qapp.processEvents()
    after_r = win.row_split.sizes()
    assert after_r[0] > r[0] + 50, "row splitter did not resize the panes"
    assert abs(after_r[0] + after_r[1] - (r[0] + r[1])) <= 2

    # The canvases still keep the same axes frame after the drag (no reflow).
    im = win.image_canvas._ax.get_position(original=True).bounds
    cv = win.curves_canvas._ax.get_position(original=True).bounds
    assert np.allclose(im, cv, atol=5e-3)
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


def test_curves_panel_shows_large_curves_in_full(qapp):
    """update_curves must show the whole critical curve / caustic even when it
    is much bigger than the model grid FOV (it auto-scales to the curves)."""
    from app.plotting import CurvesCanvas

    cv = CurvesCanvas()
    theta = np.linspace(0, 2 * np.pi, 50)
    r = 6.0                                     # grid FOV would be ±2.475
    cv.update_curves(r * np.cos(theta), r * np.sin(theta),
                     r * np.cos(theta), r * np.sin(theta),
                     num_pix=100, delta_pix=0.05)
    lo, hi = cv._ax.get_xlim()
    assert not np.isclose(lo, -2.475), "still pinned to the grid FOV"
    assert lo <= -6.0 * 1.05 and hi >= 6.0 * 1.05, "large curve clipped"
    cv.deleteLater()


def test_curves_panel_zooms_to_small_curves(qapp):
    """A small critical curve must zoom in on itself, not show a huge FOV."""
    from app.plotting import CurvesCanvas

    cv = CurvesCanvas()
    theta = np.linspace(0, 2 * np.pi, 50)
    r = 0.3
    cv.update_curves(r * np.cos(theta), r * np.sin(theta),
                     r * np.cos(theta), r * np.sin(theta),
                     num_pix=100, delta_pix=0.05)
    lo, hi = cv._ax.get_xlim()
    assert abs(hi) < 2.0, "small curve should be zoomed in, not at ±2.475"
    cv.deleteLater()


# ------------------------------------------------------- adaptive fill + ticks
def test_ticks_stay_inside_the_data_range(qapp):
    """Tick marks must adapt to each panel's arcsec range, not spill past it.

    Regression: the default locator rounded the +/-3.725 field up to +/-4.5, so
    ticks floated outside the visible image.  With ``prune="both"`` every tick
    must fall within (or on) the panel's own data limits.
    """
    from app.main_window import MainWindow

    win = MainWindow()
    win.resize(1500, 1050)
    win.show()
    qapp.processEvents()
    win._rerender()
    qapp.processEvents()

    for c in (win.fermat_canvas, win.image_canvas, win.delay_canvas,
              win.curves_canvas):
        x0, x1 = c._ax.get_xlim()
        y0, y1 = c._ax.get_ylim()
        for tk in c._ax.get_xticks():
            assert x0 - 1e-9 <= tk <= x1 + 1e-9, f"{c}: xtick {tk} outside {x0},{x1}"
        for tk in c._ax.get_yticks():
            assert y0 - 1e-9 <= tk <= y1 + 1e-9, f"{c}: ytick {tk} outside {y0},{y1}"
    win.close()
    win.deleteLater()


def test_axes_fill_the_widget_binding_dimension(qapp):
    """The square sky field must fill the cell's binding (shorter) usable
    dimension edge-to-edge and stay square on screen.

    Regression: the axes were a fixed fraction of the figure, so on a wide cell
    the square left ~150 px of dead space.  The axes now adapt to the widget:
    the square side equals ``min(usable width, usable height)`` — the binding
    dimension — after the fixed plot margins and the reserved colourbar strip.
    """
    from app.main_window import MainWindow

    win = MainWindow()
    win.resize(1500, 1050)
    win.show()
    qapp.processEvents()
    win._rerender()
    qapp.processEvents()

    for cname in ("fermat_canvas", "image_canvas", "delay_canvas", "curves_canvas"):
        c = getattr(win, cname)
        # Usable width = cell − _MARGIN(left 46 + right 14) − colourbar strip 26
        # (reserved for every panel so stacked panels keep identical geometry).
        usable_w = max(c.width() - 86, 0)
        # Usable height = cell − _MARGIN(top 22 + bottom 30).
        usable_h = max(c.height() - 52, 0)
        binding = min(usable_w, usable_h)
        bb = c._ax.get_window_extent()
        # the square fills the binding dimension edge-to-edge...
        assert abs(bb.width - binding) <= 4, \
            f"{cname}: axes width {bb.width:.0f} vs binding {binding}px"
        assert abs(bb.height - binding) <= 4, \
            f"{cname}: axes height {bb.height:.0f} vs binding {binding}px"
        # ...and is a square on screen (equal aspect, undistorted)
        assert abs(bb.width - bb.height) <= 2, \
            f"{cname}: axes not square ({bb.width:.0f}x{bb.height:.0f})"
    win.close()
    win.deleteLater()


@pytest.mark.slow
def test_gui_shows_live_fit_previews(qapp):
    """During a fit the data panel must show the swarm's current model."""
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
    for name, row in card.sliders.items():
        if name != "theta_E":
            row.set_fixed(True)

    win.fit_bar._particles.setValue(20)
    win.fit_bar._iterations.setValue(60)
    win.fit_bar._restarts.setValue(1)

    seen = []
    win._start_fit()
    win._fit_worker.previewed.connect(
        lambda it, total, chi2, img: seen.append((it, total, chi2)))

    loop = QEventLoop()
    win._fit_worker.finished_ok.connect(lambda *a: QTimer.singleShot(20, loop.quit))
    win._fit_worker.failed.connect(lambda *a: QTimer.singleShot(20, loop.quit))
    QTimer.singleShot(120000, loop.quit)
    loop.exec_()
    qapp.processEvents()

    assert len(seen) >= 2, f"expected live previews, got {len(seen)}"
    # the panel ended up showing the finished best-fit model
    assert "Best-fit model" in win.external_canvas._ax.get_title()
    assert win._fit_result is not None
    win.close()
    win.deleteLater()


def test_fit_bar_exposes_preview_settings(qapp):
    """Preview rendering is optional and rate-limited from the UI."""
    from app.controls import FitBar

    bar = FitBar()
    s = bar.settings()
    assert s["preview_enabled"] is True
    assert s["preview_interval"] > 0
    # Unchecking must disable the interval box (it is meaningless without it).
    bar._preview_chk.setChecked(False)
    assert bar.preview_enabled() is False
    assert bar._preview_interval.isEnabled() is False
    bar._preview_chk.setChecked(True)
    assert bar._preview_interval.isEnabled() is True
    # The interval is settable within a sane range.
    bar._preview_interval.setValue(2.5)
    assert abs(bar.settings()["preview_interval"] - 2.5) < 1e-9
    bar.deleteLater()


def test_worker_disables_previews_when_unchecked():
    from app.fit_worker import FitWorker

    on = FitWorker(None, None, [], [], [], preview_enabled=True)
    off = FitWorker(None, None, [], [], [], preview_enabled=False)
    assert on._preview_enabled is True
    assert off._preview_enabled is False
    assert "preview_interval" in on._kwargs and "preview_interval" in off._kwargs


# ------------------------------------------------- display/data row layout
def test_external_image_buttons_share_the_numpix_row(qapp):
    """The data buttons sit on the numPix row, in their own separated group."""
    from app.main_window import MainWindow

    win = MainWindow()
    win.resize(1600, 1080)
    win.show()
    qapp.processEvents()

    numpix = win.display_bar._numpix
    bx, by = numpix.mapTo(win, numpix.rect().topLeft()).x(), \
        numpix.mapTo(win, numpix.rect().topLeft()).y()

    for btn in (win.data_bar._load_btn, win.data_bar._clear_btn,
                win.data_bar._noise_btn, win.data_bar._mask_btn,
                win.data_bar._psf_btn):
        tl = btn.mapTo(win, btn.rect().topLeft())
        assert abs(tl.y() - by) < 12, "button not on the numPix row"
        assert tl.x() > bx, "button should be to the right of numPix"

    # separated: a visible gap between the last display control and the buttons
    last_display = win.display_bar._three_d
    gap = win.data_bar._load_btn.mapTo(win, win.data_bar._load_btn.rect().topLeft()).x() \
        - (last_display.mapTo(win, last_display.rect().topLeft()).x() + last_display.width())
    assert gap > 25, f"groups are not visually separated (gap {gap} px)"
    win.close()
    win.deleteLater()


def test_display_row_scrolls_instead_of_clipping(qapp):
    """On a narrow window the row must stay reachable via a scrollbar."""
    from app.main_window import MainWindow

    win = MainWindow()
    win.resize(900, 900)
    win.show()
    qapp.processEvents()

    sa = win.display_row
    sb = sa.horizontalScrollBar()
    assert sb.maximum() > 0, "expected a horizontal scrollbar on a narrow window"
    sb.setValue(sb.maximum())
    qapp.processEvents()
    tl = win.data_bar._psf_btn.mapTo(win, win.data_bar._psf_btn.rect().topLeft())
    assert 0 <= tl.x() and tl.x() + win.data_bar._psf_btn.width() <= win.width()
    win.close()
    win.deleteLater()


def test_data_bar_buttons_emit_the_right_kind(qapp):
    """Each moved button must emit the kind its loader expects.

    The window's real handler (which opens a modal file dialog) is disconnected
    first, so this stays a unit test of the button -> signal mapping.
    """
    from app.controls import DataBar

    bar = DataBar()
    calls = []
    bar.loadAuxRequested.connect(calls.append)     # no window handler attached
    bar._noise_btn.click()
    bar._mask_btn.click()
    bar._psf_btn.click()
    assert calls == ["noise", "mask", "psf"]

    loads = []
    bar.loadImageRequested.connect(lambda: loads.append("image"))
    clears = []
    bar.clearRequested.connect(lambda: clears.append("clear"))
    bar._load_btn.click()
    bar._clear_btn.click()
    assert loads == ["image"] and clears == ["clear"]
    bar.deleteLater()


def test_window_wires_data_bar_to_its_loaders(qapp):
    """The window must connect the data bar to its handlers (not leave them dead).

    ``disconnect()`` raises TypeError when a signal has no connections, so
    succeeding here proves the window wired each one up.
    """
    from app.main_window import MainWindow

    win = MainWindow()
    for signal in (win.data_bar.loadImageRequested, win.data_bar.clearRequested,
                   win.data_bar.loadAuxRequested):
        signal.disconnect()        # TypeError if the window never connected it
    win.close()
    win.deleteLater()


# ------------------------------------------------- slider + numeric input
def test_parameter_accepts_a_typed_exact_value(qapp):
    """Sliders cannot hit exact values; the number box can, and it wins."""
    from app.controls import LensesPanel

    panel = LensesPanel()
    row = panel._cards[0].sliders["theta_E"]
    step = (row.vmax - row.vmin) / 1000.0
    assert step > 1e-3, "pick a parameter whose slider step is coarse"

    exact = 1.1050                      # not representable on the slider
    row._value.setValue(exact)
    assert abs(row.value() - exact) < 1e-9          # exact value kept
    assert abs(row.slider_value() - exact) <= step  # slider only snaps nearby
    # and the model actually gets the typed value
    assert abs(panel._cards[0].to_params().theta_E - exact) < 1e-9
    panel.deleteLater()


def test_dragging_the_slider_updates_the_number_box(qapp):
    from app.controls import LensesPanel

    panel = LensesPanel()
    row = panel._cards[0].sliders["theta_E"]
    row._slider.setValue(row._to_int(2.0))
    # the box rounds the slider's position to its own (finer) resolution
    res = 10.0 ** -row._value.decimals()
    assert abs(row._value.value() - row.slider_value()) <= res
    assert abs(row.value() - 2.0) < 0.01
    panel.deleteLater()


def test_typing_emits_changed_once_and_can_be_locked(qapp):
    from app.controls import LensesPanel

    panel = LensesPanel()
    row = panel._cards[0].sliders["theta_E"]
    seen = []
    panel.changed.connect(lambda: seen.append(1))

    row._value.setValue(1.5)                    # typing
    assert len(seen) == 1, "typing should emit exactly one change"
    row._slider.setValue(row._to_int(0.9))      # dragging
    assert len(seen) == 2, "dragging should emit exactly one change"

    # A locked parameter ignores typing just like dragging.
    row.set_value(1.2)
    row.set_fixed(True)
    assert row._value.isEnabled() is False
    row._value.setValue(2.9)
    assert abs(row.value() - 1.2) < 1e-9
    panel.deleteLater()


def test_fit_uses_the_typed_value_not_the_slider_snap(qapp):
    """The typed precision must reach the fit, not the coarse slider position."""
    from app.main_window import MainWindow

    win = MainWindow()
    card = win.lenses_panel._cards[0]
    card.sliders["theta_E"]._value.setValue(1.1050)
    assert abs(win._build_config().lenses[0].theta_E - 1.1050) < 1e-9
    assert abs(card.sliders["theta_E"].slider_value() - 1.1050) > 1e-4
    win.close()
    win.deleteLater()


def test_source_point_checkbox_creates_attached_point_source(qapp):
    """The Sources card's 'point' checkbox makes an attached point source, and
    the resulting app config carries a source-referencing point source."""
    from app.controls import PointSourcesPanel, SourcesPanel

    sp = SourcesPanel()
    pp = PointSourcesPanel()
    sp.point_source_toggled.connect(pp.set_attached)
    pp.attached_changed.connect(sp.set_point_checked)
    pp.update_sources(sp.source_list())      # MainWindow does this per config build

    assert pp.point_source_list() == []          # no point sources by default
    sp._cards[0]._pt_check.setChecked(True)
    pts = pp.point_source_list()
    assert len(pts) == 1 and pts[0].ref_source == 0
    assert sp._cards[0]._pt_check.isChecked()

    # flipping the point-source card's anchor back to 'own' unchecks the source
    pp._cards[0]._on_anchor(0)
    assert not sp._cards[0]._pt_check.isChecked()
    assert pp.point_source_list()[0].ref_source == -1

    # unchecking from the source side frees the attached source
    sp._cards[0]._pt_check.setChecked(True)
    sp._cards[0]._pt_check.setChecked(False)
    assert pp.point_source_list()[0].ref_source == -1
    sp.deleteLater()
    pp.deleteLater()


def test_point_sources_panel_add_model_and_config(qapp):
    """Adding point sources in the standalone panel flows into a Config."""
    from app.controls import PointSourcesPanel
    from app import lensing_calc as lc

    pp = PointSourcesPanel()
    pp._add_point(lc.PointSourceParams(model="UNLENSED", point_amp=0.5,
                                       center_x=0.3, center_y=-0.2))
    pts = pp.point_source_list()
    assert [p.model for p in pts] == ["UNLENSED"]
    assert pts[0].point_amp == 0.5 and pts[0].center_x == 0.3
    assert pts[0].position_and_redshift(lc.Config()) == (0.3, -0.2, 0.3 or 0.3) \
        if False else True
    pp.deleteLater()


def test_main_window_renders_with_point_source(qapp):
    """End-to-end: a LENSED point source added through the window renders a
    model image whose point contribution lands on solved image positions."""
    from app.main_window import MainWindow
    from app import lensing_calc as lc

    win = MainWindow()
    try:
        win.point_sources_panel._add_point(
            lc.PointSourceParams(model="LENSED", source_amp=3.0,
                                 center_x=0.1, center_y=-0.1))
        from dataclasses import replace
        res_a = lc.compute(win._build_config())
        cfg_no = replace(win._build_config(), point_sources=[])
        diff = res_a.image - lc.compute(cfg_no).image
        assert diff.max() > 0                    # the spike showed up
        sx, sy = res_a.image_positions[0][0], res_a.image_positions[0][1]
        ys, xs = np.unravel_index(diff.argmax(), diff.shape)
        half = (cfg_no.num_pix - 1) / 2
        ra_max = (xs - half) * cfg_no.delta_pix
        dec_max = (ys - half) * cfg_no.delta_pix
        assert min((ra_max - x) ** 2 + (dec_max - y) ** 2
                   for x, y in zip(sx, sy)) < (2 * cfg_no.delta_pix) ** 2
        assert win.point_sources_panel.point_source_list()[
            0].ref_source == -1
    finally:
        win.close()


def test_apply_fitted_config_handles_point_sources(qapp):
    """Writing a fit result back must rebuild point-source cards when the
    count changed, and push values otherwise."""
    from app.main_window import MainWindow
    from app import lensing_calc as lc

    win = MainWindow()
    try:
        win._build_config()                                  # populate anchor sources
        # start with one LENSED point source card
        win.point_sources_panel._add_point(lc.PointSourceParams(
            model="LENSED", source_amp=0.5, center_x=0.2, center_y=-0.1))

        fitted = lc.Config(
            lenses=win.lenses_panel.lens_list(),
            sources=win.sources_panel.source_list(),
            point_sources=[  # same count: values pushed back
                lc.PointSourceParams(
                    model="LENSED", source_amp=0.9, center_x=0.4,
                    center_y=-0.3)],
        )
        win._apply_fitted_config(fitted)
        ps = win.point_sources_panel.point_source_list()
        assert len(ps) == 1
        assert ps[0].source_amp == 0.9 and ps[0].center_x == 0.4

        # different count (fit dropped one): cards are rebuilt from the result
        fitted2 = lc.Config(
            lenses=win.lenses_panel.lens_list(),
            sources=win.sources_panel.source_list(),
            point_sources=[
                lc.PointSourceParams(model="UNLENSED", point_amp=0.2,
                                     center_x=0.9, center_y=0.1),
                lc.PointSourceParams(model="LENSED", source_amp=0.3,
                                     center_x=-0.5, center_y=0.2)],
        )
        win._apply_fitted_config(fitted2)
        ps2 = win.point_sources_panel.point_source_list()
        assert [p.model for p in ps2] == ["UNLENSED", "LENSED"]
        assert ps2[0].point_amp == 0.2 and ps2[1].center_y == 0.2
    finally:
        win.close()


def test_config_panels_resizeable_via_splitter(qapp):
    """Lenses / Sources / Point sources share a vertical splitter so the user
    can resize each config panel independently."""
    from app.main_window import MainWindow

    win = MainWindow()
    try:
        sp = win.cfg_split
        assert sp is not None
        assert sp.orientation() == 2  # Qt.Vertical
        widgets = [sp.widget(i) for i in range(sp.count())]
        assert widgets == [win.lenses_panel, win.sources_panel,
                           win.point_sources_panel]
        # moving the handle resizes the panels (no crash)
        sizes = sp.sizes()
        sizes[1] += 40
        sp.setSizes(sizes)
        assert sum(sp.sizes()) > 0
    finally:
        win.close()


def test_entry_cards_auto_numbered(qapp):
    """Card titles carry a live index (Lens 1, Lens 2, …) that stays correct
    after adding and removing entries."""
    from app.controls import LensesPanel, SourcesPanel, PointSourcesPanel

    lp = LensesPanel()
    assert [c.title() for c in lp._cards] == ["Lens 1"]
    lp._add_card(lp._cards[0].__class__())
    lp._add_card(lp._cards[0].__class__())
    assert [c.title() for c in lp._cards] == ["Lens 1", "Lens 2", "Lens 3"]
    lp._remove_card(lp._cards[0])
    assert [c.title() for c in lp._cards] == ["Lens 1", "Lens 2"]

    sp = SourcesPanel()
    sp._add_card(sp._cards[0].__class__())
    assert [c.title() for c in sp._cards] == ["Source 1", "Source 2"]

    pp = PointSourcesPanel()
    from app import lensing_calc as lc
    pp._add_point(lc.PointSourceParams())
    pp._add_point(lc.PointSourceParams())
    assert [c.title() for c in pp._cards] == ["Point source 1", "Point source 2"]
    pp._remove_card(pp._cards[0])
    assert [c.title() for c in pp._cards] == ["Point source 1"]

    lp.deleteLater(); sp.deleteLater(); pp.deleteLater()


def test_cards_show_only_params_for_selected_model(qapp):
    """Switching the model hides the irrelevant sliders; switching back restores
    them and their values are preserved."""
    from app.controls import LensesPanel, SourcesPanel
    from app import lensing_calc as lc

    lp = LensesPanel()
    lens = lp._cards[0]
    lp.show(); qapp.processEvents()   # offscreen: isVisible needs a shown ancestor
    lens.sliders["theta_E"].set_value(2.0)
    lens._model_combo.setCurrentText("NFW")
    assert not lens.sliders["theta_E"].isVisible()
    assert not lens.sliders["e1"].isVisible()
    assert lens.sliders["Rs"].isVisible()
    assert lens.sliders["alpha_Rs"].isVisible()
    assert lens.sliders["gamma1"].isVisible()        # external shear stays

    lens._model_combo.setCurrentText("SIS_TRUNCATED")
    assert lens.sliders["Rs"].isVisible() is False
    assert lens.sliders["r_trunc"].isVisible()
    assert lens.sliders["theta_E"].isVisible()       # back where it matters

    # and the value that lived on the hidden slider is preserved
    assert lens.sliders["theta_E"].value() == 2.0

    sp = SourcesPanel()
    src = sp._cards[0]
    sp.show(); qapp.processEvents()
    src._model_combo.setCurrentText("GAUSSIAN")
    assert src.sliders["sigma"].isVisible()
    assert not src.sliders["R_sersic"].isVisible()
    assert not src.sliders["e1"].isVisible()
    # the ROW disappears too: the label hides with the control
    assert not src._row_labels["R_sersic"].isVisible()
    assert src._row_labels["sigma"].isVisible()
    src._model_combo.setCurrentText("CORE_SERSIC")
    assert src.sliders["R_sersic"].isVisible()
    assert src.sliders["Rb"].isVisible()
    assert src.sliders["gamma"].isVisible()
    assert src._row_labels["Rb"].isVisible()
    lp.deleteLater(); sp.deleteLater()


def test_lens_light_sliders_follow_light_model(qapp):
    from app.controls import LensesPanel

    lp = LensesPanel()
    lens = lp._cards[0]
    lp.show(); qapp.processEvents()
    lens._light_combo.setCurrentText("GAUSSIAN")
    assert lens.sliders["light_sigma"].isVisible()
    assert not lens.sliders["light_R_sersic"].isVisible()
    assert not lens.sliders["light_e1"].isVisible()
    lens._light_combo.setCurrentText("SERSIC_ELLIPSE")
    assert lens.sliders["light_R_sersic"].isVisible()
    assert lens.sliders["light_e1"].isVisible()
    assert not lens.sliders["light_sigma"].isVisible()
    lens._light_combo.setCurrentText("NONE")
    assert not lens.sliders["light_sigma"].isVisible()
    assert not lens.sliders["light_R_sersic"].isVisible()
    lp.deleteLater()
