"""Tests for the lenstronomy 2D simulation module."""

import numpy as np
import pytest

from app import sim2d


def test_render_returns_grid():
    img = sim2d.render(num_pix=100, delta_pix=0.05)
    assert img.shape == (100, 100)
    assert np.isfinite(img).all()


def test_render_has_bright_content():
    """A lensed image should contain resolved structure (max >> median)."""
    img = sim2d.render(num_pix=120, delta_pix=0.05)
    assert img.max() > img.mean()


def test_default_kwargs_keys():
    lens = sim2d.default_lens_kwargs()
    src = sim2d.default_source_kwargs()
    assert {"theta_E", "gamma1", "gamma2", "center_x", "center_y"} <= set(lens)
    assert {"amp", "R_sersic", "e1", "e2", "center_x", "center_y"} <= set(src)


def test_build_args_structure():
    kwargs_lens, kwargs_light = sim2d.build_args(
        sim2d.default_lens_kwargs(), sim2d.default_source_kwargs()
    )
    assert len(kwargs_lens) == 2
    assert len(kwargs_light) == 1
    assert "theta_E" in kwargs_lens[0]
    assert "R_sersic" in kwargs_light[0]


def test_parameter_change_alters_image():
    """Moving the source well off-axis should change the lensed image."""
    src1 = sim2d.default_source_kwargs()
    src2 = dict(src1, center_x=-1.2, center_y=0.8)
    img1 = sim2d.render(source=src1, num_pix=100, delta_pix=0.05)
    img2 = sim2d.render(source=src2, num_pix=100, delta_pix=0.05)
    assert not np.allclose(img1, img2)


if __name__ == "__main__":
    # Allow running this file directly (IDE "Run" button) — needs the project
    # root on sys.path to import app.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(pytest.main([__file__, "-v"]))
