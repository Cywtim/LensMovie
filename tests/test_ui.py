"""Offscreen smoke tests for the Qt UI (no real display needed)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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
