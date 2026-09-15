"""Tests for the lenstronomy physics core (:mod:`app.lensing_calc`)."""

import numpy as np
import pytest

from app import lensing_calc as lc


def test_default_single_lens_compute():
    cfg = lc.Config(num_pix=100)
    res = lc.compute(cfg)
    assert res.ok, res.error
    assert res.image.shape == (100, 100)
    assert np.isfinite(res.image).all()
    assert np.isfinite(res.fermat).all()
    assert np.isfinite(res.time_delay).all()
    assert res.image.max() > 0


def test_multiplane_multisource_compute():
    lenses = [
        lc.LensParams(model="SIE", theta_E=1.0, e1=0.1, e2=-0.2, redshift=0.3),
        lc.LensParams(model="SIS", theta_E=0.6, center_x=0.2, center_y=-0.1, redshift=0.7),
    ]
    sources = [
        lc.SourceParams(center_x=0.1, center_y=-0.1, redshift=1.0),
        lc.SourceParams(center_x=-0.2, center_y=0.15, amp=0.7, redshift=2.0),
    ]
    cfg = lc.Config(lenses=lenses, sources=sources, num_pix=90)
    res = lc.compute(cfg)
    assert res.ok, res.error
    assert np.isfinite(res.image).all()
    assert res.image.max() > 0
    # Two sources -> two (possibly populated) image-position entries.
    assert len(res.image_positions) == 2


def test_expand_lenses_with_shear():
    lens = lc.LensParams(model="SIS", theta_E=1.0, gamma1=0.05, gamma2=0.0, redshift=0.5)
    models, zs, kwargs = lc.expand_lenses([lens])
    assert models == ["SIS", "SHEAR"]
    assert zs == [0.5, 0.5]
    assert "gamma1" in kwargs[1]


def test_lens_kwargs_by_model():
    assert "e1" not in lc.lens_kwargs(lc.LensParams(model="SIS"))
    assert "e1" in lc.lens_kwargs(lc.LensParams(model="SIE"))
    assert "gamma" in lc.lens_kwargs(lc.LensParams(model="PEMD", gamma=2.0))


def test_all_extended_source_models_compute():
    for model in lc.SOURCE_MODELS:
        cfg = lc.Config(sources=[lc.SourceParams(model=model, R_sersic=0.2, sigma=0.2)],
                        num_pix=70)
        res = lc.compute(cfg)
        assert res.ok, f"{model}: {res.error}"
        assert res.image.max() > 0


def test_source_profile_kwargs_per_model():
    assert "R_sersic" in lc.SourceParams(model="SERSIC_ELLIPSE").profile_kwargs()
    assert "n_sersic" in lc.SourceParams(model="SERSIC").profile_kwargs()
    assert "sigma" in lc.SourceParams(model="GAUSSIAN").profile_kwargs()
    ge = lc.SourceParams(model="GAUSSIAN_ELLIPSE").profile_kwargs()
    assert "sigma" in ge and "e1" in ge
    # A Sersic profile must not carry Gaussian-only keys and vice versa.
    assert "sigma" not in lc.SourceParams(model="SERSIC_ELLIPSE").profile_kwargs()
    assert "R_sersic" not in lc.SourceParams(model="GAUSSIAN").profile_kwargs()


def test_extended_source_is_resolved_not_pointlike():
    """A large source must illuminate more of the image plane than a tiny one."""
    big = lc.compute(lc.Config(sources=[lc.SourceParams(R_sersic=0.4)], num_pix=90))
    small = lc.compute(lc.Config(sources=[lc.SourceParams(R_sersic=0.01)], num_pix=90))
    big_px = int((big.image > big.image.max() * 1e-3).sum())
    small_px = int((small.image > small.image.max() * 1e-3).sum())
    assert big_px > small_px


def test_source_effective_radius():
    assert lc.SourceParams(model="SERSIC_ELLIPSE", R_sersic=0.3).effective_radius() == 0.3
    assert lc.SourceParams(model="GAUSSIAN", sigma=0.15).effective_radius() == 0.15


def test_lensed_image_is_centred_and_ring_shaped():
    """Regression: the sky grid must be centred on the lens.

    A source well inside theta_E must produce a centred Einstein ring, not a
    blob pushed to the edge of the frame (which is what a wrong
    ra/dec-at-pixel-0 origin produces).
    """
    cfg = lc.Config(num_pix=150, delta_pix=0.05)   # SIS theta_E=1.0
    res = lc.compute(cfg)
    assert res.ok, res.error

    img = res.image
    n = img.shape[0]
    mask = img > img.max() * 0.05
    ys, xs = np.nonzero(mask)
    assert len(xs) > 0

    # Emission must be centred on the lens, not at a frame edge.
    assert abs(xs.mean() - (n - 1) / 2) < 0.15 * n
    assert abs(ys.mean() - (n - 1) / 2) < 0.15 * n

    # Radial profile should peak away from the centre (a ring), at ~theta_E.
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2)
    prof = np.array([img[(rr >= r) & (rr < r + 3)].mean() for r in range(0, 60, 3)])
    peak_px = int(np.argmax(prof) * 3)
    assert peak_px >= 9, f"expected a ring, radial peak at {peak_px}px"
    # ring radius should be near theta_E in pixels (1.0 arcsec / 0.05)
    assert abs(peak_px - 20) < 10, f"ring at {peak_px}px, expected ~20px"
