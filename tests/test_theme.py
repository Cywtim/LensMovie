"""Theme layer: the QSS/palette are the app's single skin, and applying the
theme must not break the window's build/render."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import re

import pytest
from PyQt5.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_qss_resolves_every_palette_token():
    """No ``@name@`` placeholder may survive substitution.

    The palette is the single source of colours, so a forgotten token should
    fail loudly rather than silently rendering a raw ``@name@``.
    """
    from app import theme

    sheet = theme.qss()
    assert "@" not in sheet, "unresolved palette token leaked into the QSS"
    assert "QGroupBox" in sheet
    # a palette colour actually made it into the stylesheet
    assert theme.PALETTE["accent"].lower() in sheet.lower()


def test_qss_uses_only_known_palette_tokens():
    """theme.qss may only reference colours that exist in PALETTE."""
    from app import theme

    text = theme._QSS_PATH.read_text(encoding="utf-8")
    used = set(re.findall(r"@([a-z_0-9]+)@", text))
    missing = used - set(theme.PALETTE)
    assert not missing, f"theme.qss references palette tokens missing from PALETTE: {missing}"


def test_apply_theme_styles_and_keeps_window_render(qapp):
    """Applying the theme must not break constructing/rendering the window."""
    import matplotlib

    from app import theme
    from app.main_window import MainWindow

    before_rc = dict(matplotlib.rcParams)
    previous_sheet = qapp.styleSheet()
    try:
        theme.apply_theme(qapp)
        assert qapp.styleSheet() == theme.qss()
        assert qapp.styleSheet()  # non-empty, so styling is actually applied

        win = MainWindow()            # builds and renders under the theme
        assert win._render_ok is True
        win.close()
        win.deleteLater()
    finally:
        qapp.setStyleSheet(previous_sheet)
        matplotlib.rcParams.update(before_rc)


def test_plotting_canvases_use_theme_colours(qapp):
    """2D canvases must pull their colours from the central theme, not hard-code."""
    from app import theme
    from app.plotting import FieldCanvas, ImageCanvas

    f = FieldCanvas("Fermat potential (relative)")
    assert f._figure.get_facecolor()[:3] == tuple(
        int(theme.PALETTE["panel"].lstrip("#")[i:i + 2], 16) / 255
        for i in (0, 2, 4)
    )
    assert f._ax.get_facecolor()[:3] == tuple(
        int(theme.PALETTE["input"].lstrip("#")[i:i + 2], 16) / 255
        for i in (0, 2, 4)
    )
    img = ImageCanvas()
    assert img._IMG_COLORS == theme.MARKER_COLORS
    f.deleteLater()
    img.deleteLater()


def test_marker_colours_are_distinct():
    from app import theme

    rows = [c.lstrip("#").upper() for c in theme.MARKER_COLORS]
    assert len(set(rows)) == len(rows), "duplicate marker colours reduce legibility"
