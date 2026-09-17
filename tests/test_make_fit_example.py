"""The fit-example generator produces loadable, internally-consistent files."""

import sys

import numpy as np
import pytest

sys.path.insert(0, "tools")

import make_fit_example as mfe  # noqa: E402


def test_generate_writes_consistent_example(tmp_path, monkeypatch):
    monkeypatch.setattr(mfe, "OUT", tmp_path)
    mfe._generate()

    data = np.load(tmp_path / "data.npy")
    model = np.load(tmp_path / "model_truth.npy")
    noise = np.load(tmp_path / "noise.npy")
    kernel = np.load(tmp_path / "psf_kernel.npy")

    assert data.shape == (mfe.NUM_PIX, mfe.NUM_PIX)
    assert model.shape == data.shape
    assert noise.shape == data.shape
    assert (noise == mfe.NOISE_SIGMA).all()          # constant sigma map
    assert np.isfinite(kernel).all() and abs(kernel.sum() - 1.0) < 1e-6
    # data = truth + the declared noise (exactly reproduceable via the seed)
    generated = mfe.truth_config()
    assert np.allclose(mfe.lc.compute(generated).image, model, atol=1e-9)
    resid = data - model
    assert abs(resid.std() - mfe.NOISE_SIGMA) < 0.005 * mfe.NOISE_SIGMA

    # every promised file is present
    for name in ("data.npy", "model_truth.npy", "noise.npy",
                 "psf_kernel.npy", "truth.json", "overview.png", "README.md"):
        assert (tmp_path / name).exists(), f"missing {name}"
    cfg = mfe.config_to_json(generated)
    assert cfg["lens"]["model"] == "SIE"
    assert cfg["source"]["center_x/y"] == pytest.approx([0.18, 0.08])


def test_clean_model_plus_noise_map_reproduces_data(tmp_path):
    """With the noise-map semantics, loading the CLEAN model as the image and the
    sigma map as noise must reproduce the pre-made observation exactly.  The
    controller's draw seed equals the example generator's, so
    ``model_truth + noise.npy == data.npy`` pixel-for-pixel."""
    from app.controller import LensMovieController

    ctrl = LensMovieController()
    ctrl.external_array = np.load(mfe.OUT / "model_truth.npy")
    ctrl.noise_array = np.load(mfe.OUT / "noise.npy")
    observed = np.asarray(ctrl.observation)
    data = np.load(mfe.OUT / "data.npy")
    assert observed.shape == data.shape
    assert np.allclose(observed, data, atol=1e-9), "seeded draw must match example"
    # deterministic: repeated reads give the identical array
    assert ctrl.observation is observed
    # and swapping the noise map gives a *different* draw
    ctrl.noise_array = np.full(data.shape, mfe.NOISE_SIGMA * 2.0)
    assert not np.allclose(ctrl.observation, observed)
