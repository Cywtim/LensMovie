"""Plotting helpers: the source-plane contour geometry + the curves canvas."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from app import lensing_calc as lc
from app.plotting import source_contour


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# -------------------------------------------------------------- source_contour
def test_source_contour_round_and_centred():
    """A circular source (e1=e2=0) maps to a circle on the source plane."""
    s = lc.SourceParams(center_x=0.1, center_y=-0.2, R_sersic=0.3)
    ra, dec = source_contour(s, n=200)
    assert len(ra) == 200
    # every point sits on an ellipse centred at (center_x, center_y)
    rr = np.hypot(ra - 0.1, dec + 0.2)
    r = float(s.effective_radius()) * 2.0
    assert np.allclose(rr, r, atol=1e-4)


def test_source_contour_ellipse_follows_ellipticity():
    """Non-zero e1 flattens the contour along the expected axis: one axis keeps
    the full 2r, the perpendicular one shrinks by (1 - e)."""
    s = lc.SourceParams(e1=-0.4, e2=0.0, R_sersic=0.25)
    ra, dec = source_contour(s, n=200)
    r = float(s.effective_radius()) * 2.0
    # e = |e1| = 0.4; phi = pi/2 puts the major axis along dec
    assert dec.ptp() > ra.ptp()                     # flattened along the other axis
    assert np.isclose(dec.ptp() / 2.0, r, atol=1e-4)            # major = 2r
    assert np.isclose(ra.ptp() / 2.0, r * (1 - 0.4), atol=1e-4)  # minor = 2r(1-e)


def test_source_contour_tiny_and_huge_clamped():
    """A minuscule source cannot collapse to nothing; a huge one is capped."""
    tiny = lc.SourceParams(R_sersic=1e-5)
    ra, dec = source_contour(tiny, n=64)
    assert np.hypot(ra - tiny.center_x, dec - tiny.center_y).max() >= 0.02 - 1e-6

    huge = lc.SourceParams(R_sersic=10.0)
    ra, dec = source_contour(huge, n=64)
    assert np.hypot(ra - huge.center_x, dec - huge.center_y).max() <= 1.0 + 1e-6


# -------------------------------------------------------------- curves canvas
def test_curves_canvas_draws_source_outlines(qapp):
    """update_curves with outlines adds one gold extent line per source and
    includes it in the auto-scaled window."""
    from app.plotting import CurvesCanvas

    canvas = CurvesCanvas()
    src = lc.SourceParams(center_x=0.3, center_y=-0.1, R_sersic=0.3)
    ra, dec = source_contour(src)
    canvas.update_curves([], [], [], [], source_outlines=[(ra, dec, 0)],
                         source_positions=[(np.array([0.3]), np.array([-0.1]), 0)])
    assert len(canvas._outline_lines) == 1
    x, y = canvas._outline_lines[0].get_data()
    assert len(x) == len(ra)
    lo, hi = canvas._ax.get_xlim()
    r = float(src.effective_radius()) * 2.0
    assert hi > 0.3 + r                  # contour (ra up to center+r) in window
    assert lo < -0.1 - r                 # ... and the whole ellipse visible
    canvas._style_axes("x")
    canvas.close()
