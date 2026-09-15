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
    win._on_grid_chk.setChecked(True)
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
