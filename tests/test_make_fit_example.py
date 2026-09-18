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


def test_make_fit_example_lens_noise_decoration(tmp_path):
    """The sigma map decorates the Lens image model panel: loading it must not
    alter ``data.npy``; the panel draw has the right sigma and is deterministic."""
    from types import SimpleNamespace
    from app.main_window import MainWindow

    # _lens_image_with_noise only reads self.controller.noise_array, so exercise
    # the real method on a stub without constructing MainWindow's vispy scene.
    make = SimpleNamespace(controller=SimpleNamespace(noise_array=None))
    method = MainWindow._lens_image_with_noise

    model = np.load(mfe.OUT / "model_truth.npy")
    sig = np.load(mfe.OUT / "noise.npy")
    sigma = float(sig[0, 0])
    # without a sigma map the lens image is the clean model
    assert np.allclose(method(make, model, mfe.NUM_PIX, 0.05), model)
    make.controller.noise_array = sig
    noisy = np.asarray(method(make, model, mfe.NUM_PIX, 0.05))
    assert abs((noisy - model).std() - sigma) < 0.02 * sigma
    assert np.allclose(method(make, model, mfe.NUM_PIX, 0.05), noisy)
