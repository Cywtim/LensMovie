"""Offscreen smoke tests for the Qt UI (no real display needed).

Run with `QT_QPA_PLATFORM=offscreen` (set inside conftest or here).
Constructing widgets with the offscreen platform verifies layout/signal wiring
without opening a window.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

import pytest


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_parameter_panel_defaults(qapp):
    from app.controls import ParameterPanel

    panel = ParameterPanel()
    d = panel.as_dict()
    assert d["lens"]["theta_E"] == 1.0
    assert d["source"]["R_sersic"] == 0.1
    assert d["display"]["num_pix"] == 150
    panel.deleteLater()


def test_main_window_constructs(qapp):
    """The full window should build and trigger one successful render."""
    from app.main_window import MainWindow

    win = MainWindow()
    assert win._render_ok is True
    win.close()
    win.deleteLater()


def test_slider_change_emits_params(qapp):
    from app.controls import ParameterPanel

    panel = ParameterPanel()
    received = []

    def on_change(params):
        received.append(params)

    panel.parametersChanged.connect(on_change)
    # Move the slider to a genuinely different value so valueChanged fires.
    target = 2.0
    row = panel._sliders["theta_E"]
    row._slider.setValue(row._to_int(target))
    assert received, "expected a parametersChanged emission"
    assert abs(received[-1]["lens"]["theta_E"] - target) < 0.01
    panel.deleteLater()


if __name__ == "__main__":
    # Allow running this file directly (e.g. an IDE "Run" button on the file),
    # which otherwise would neither find `app` nor actually run any test.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(pytest.main([__file__, "-v"]))
