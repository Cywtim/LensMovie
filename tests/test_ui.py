"""Offscreen smoke tests for the Qt UI (no real display needed)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_config_panel_produces_valid_config(qapp):
    from app.controls import ConfigPanel

    panel = ConfigPanel()
    cfg = panel.as_config()
    assert len(cfg.lenses) == 1
    assert len(cfg.sources) == 1
    assert cfg.lenses[0].model in ("SIS", "SIE", "PEMD")
    assert cfg.sources[0].redshift > 0
    assert cfg.num_pix == 150
    panel.deleteLater()


def test_add_remove_lens_and_source(qapp):
    from app.controls import ConfigPanel

    panel = ConfigPanel()
    n_lenses = len(panel._lenses)
    n_sources = len(panel._sources)

    panel._add_lens()
    panel._add_lens()
    panel._add_source()
    assert len(panel._lenses) == n_lenses + 2
    assert len(panel._sources) == n_sources + 1

    # Removing down to one lens / one source is blocked.
    for card in list(panel._lenses):
        panel._remove_entry(card)
    assert len(panel._lenses) == 1
    for card in list(panel._sources):
        panel._remove_entry(card)
    assert len(panel._sources) == 1
    panel.deleteLater()


def test_main_window_constructs_and_renders(qapp):
    """The full window should build and trigger one successful 2D render."""
    from app.main_window import MainWindow

    win = MainWindow()
    assert win._render_ok is True
    win.close()
    win.deleteLater()
