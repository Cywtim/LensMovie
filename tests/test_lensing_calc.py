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
