"""Tests for the PSF support and the fit-data preparation layer."""

import numpy as np
import pytest

from app import fit_data as fd
from app import lensing_calc as lc


# ------------------------------------------------------------------ PSF
def test_gaussian_psf_kernel_normalised():
    k = lc.gaussian_psf_kernel(0.3, 0.05)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1] and k.shape[0] % 2 == 1
    assert abs(k.sum() - 1.0) < 1e-9


def test_gaussian_psf_kernel_zero_fwhm_is_delta():
    k = lc.gaussian_psf_kernel(0.0, 0.05)
    assert k.shape == (1, 1)
    assert k[0, 0] == 1.0


def test_larger_fwhm_blurs_more():
    cfg = lc.Config(num_pix=110)
    sharp = lc.compute(cfg)
    blurred = lc.compute(lc.Config(num_pix=110, psf_kernel=lc.gaussian_psf_kernel(0.6, 0.05)))
    n_sharp = int((sharp.image > sharp.image.max() * 0.05).sum())
    n_blur = int((blurred.image > blurred.image.max() * 0.05).sum())
    assert n_blur > n_sharp
    assert blurred.image.max() < sharp.image.max()


def test_psf_kernel_is_accepted_and_default_is_delta():
    assert lc.Config().psf_kernel is None
    res = lc.compute(lc.Config(num_pix=80, psf_kernel=np.array([[1.0]])))
    assert res.ok


# --------------------------------------------------------- fit-data prep
def test_same_grid_skips_resampling():
    img = np.arange(100, dtype=float).reshape(10, 10)
    d = fd.prepare_fit_data(img, source_delta_pix=0.05, model_num_pix=10,
                            model_delta_pix=0.05)
    assert d.resampled is False
    assert np.allclose(d.image, img)
    assert d.num_pix == 10


def test_resampling_changes_pixel_scale():
    big = np.zeros((40, 40))
    big[18:22, 18:22] = 1.0
    d = fd.prepare_fit_data(big, source_delta_pix=0.05, model_num_pix=20,
                            model_delta_pix=0.10)
    assert d.resampled is True
    assert d.image.shape == (20, 20)
    cy, cx = np.unravel_index(np.argmax(d.image), d.image.shape)
    assert abs(cy - 9.5) <= 2 and abs(cx - 9.5) <= 2   # stays centred


def test_center_offset_moves_the_feature_to_the_centre():
    off = np.zeros((40, 40))
    off[8:12, 28:32] = 1.0            # blob up-and-right of the data centre
    d = fd.prepare_fit_data(
        off, source_delta_pix=0.05, model_num_pix=20, model_delta_pix=0.05,
        center_offset=(0.5, 0.5),     # lens centre is at +0.5", +0.5"
    )
    cy, cx = np.unravel_index(np.argmax(d.image), d.image.shape)
    assert abs(cy - 9.5) <= 2 and abs(cx - 9.5) <= 2


def test_noise_and_mask_are_carried_and_resampled():
    img = np.random.RandomState(0).rand(240, 240)
    noise = np.full((240, 240), 0.05)
    mask = np.ones((240, 240), bool)
    mask[100:140, 100:140] = False
    d = fd.prepare_fit_data(img, source_delta_pix=0.05, model_num_pix=100,
                            model_delta_pix=0.05, noise=noise, mask=mask)
    assert d.noise.shape == d.image.shape
    assert d.mask.shape == d.image.shape
    # The masked block must survive the resampling.
    assert d.usable_pixels() < d.image.size
    assert d.usable_pixels() == 100 * 100 - 40 * 40


def test_non_positive_noise_pixels_are_excluded():
    img = np.ones((10, 10))
    noise = np.full((10, 10), 0.1)
    noise[0, 0] = -1.0
    d = fd.prepare_fit_data(img, source_delta_pix=0.05, model_num_pix=10,
                            model_delta_pix=0.05, noise=noise)
    assert d.mask is not None
    assert d.mask[0, 0] == False  # noqa: E712
    assert d.usable_pixels() == 99


def test_psf_kernel_is_normalised_and_kernel_beats_fwhm():
    k = np.ones((3, 3)) * 4.0
    d = fd.prepare_fit_data(np.ones((10, 10)), source_delta_pix=0.05,
                            model_num_pix=10, model_delta_pix=0.05,
                            psf_kernel=k, psf_fwhm=0.7)
    assert abs(d.psf_kernel.sum() - 1.0) < 1e-9
    # A loaded kernel takes precedence over the FWHM.
    assert np.allclose(fd.effective_psf_kernel(d), d.psf_kernel)


def test_effective_psf_from_fwhm_when_no_kernel():
    d = fd.prepare_fit_data(np.ones((10, 10)), source_delta_pix=0.05,
                            model_num_pix=10, model_delta_pix=0.05, psf_fwhm=0.4)
    k = fd.effective_psf_kernel(d)
    assert k.ndim == 2 and k.shape[0] > 1
    assert abs(k.sum() - 1.0) < 1e-9


def test_chi2_uses_noise_and_mask():
    img = np.ones((10, 10))
    noise = np.full((10, 10), 0.5)
    mask = np.ones((10, 10), bool)
    mask[0, 0] = False
    d = fd.prepare_fit_data(img, source_delta_pix=0.05, model_num_pix=10,
                            model_delta_pix=0.05, noise=noise, mask=mask)
    assert fd.chi2(img, d) == 0.0
    # shifting by 1 sigma over 99 usable pixels
    assert abs(fd.chi2(img + 0.5, d) - 99.0) < 1e-9


def test_chi2_without_noise_raises():
    d = fd.prepare_fit_data(np.ones((10, 10)), source_delta_pix=0.05,
                            model_num_pix=10, model_delta_pix=0.05)
    with pytest.raises(fd.DataPrepError):
        fd.chi2(np.ones((10, 10)), d)


def test_invalid_inputs_raise():
    img = np.ones((10, 10))
    with pytest.raises(fd.DataPrepError):
        fd.prepare_fit_data(img, source_delta_pix=0.0, model_num_pix=10,
                            model_delta_pix=0.05)
    with pytest.raises(fd.DataPrepError):
        fd.prepare_fit_data(img, source_delta_pix=0.05, model_num_pix=10,
                            model_delta_pix=0.0)
    with pytest.raises(fd.DataPrepError):
        fd.prepare_fit_data(img, source_delta_pix=0.05, model_num_pix=10,
                            model_delta_pix=0.05, noise=np.zeros((3, 3)))


def test_model_image_shape_matches_model_grid():
    """The prepared data grid and the model image must agree for a fit."""
    d = fd.prepare_fit_data(np.random.rand(80, 80), source_delta_pix=0.08,
                            model_num_pix=60, model_delta_pix=0.08)
    res = lc.compute(lc.Config(num_pix=60, delta_pix=0.08))
    assert res.image.shape == d.image.shape


def test_unnormalised_psf_kernel_cannot_rescale_brightness():
    """A PSF must integrate to 1: a scaled kernel must not change photometry."""
    cfg = lc.Config(num_pix=90)
    base = lc.compute(cfg)
    k = lc.gaussian_psf_kernel(0.3, cfg.delta_pix)
    scaled = lc.compute(lc.Config(num_pix=90, psf_kernel=k * 7.0))
    assert np.isclose(scaled.image.max(), lc.compute(
        lc.Config(num_pix=90, psf_kernel=k)).image.max(), rtol=1e-6)
    assert scaled.image.max() < base.image.max()   # blurred, but not rescaled


# ------------------------------------------------------- psf error map (quadrature)
def test_psf_error_map_carried_and_resampled():
    img = np.arange(100.0).reshape(10, 10)
    noise = np.full((10, 10), 0.1)
    psf = np.full((10, 10), 0.3)
    d = fd.prepare_fit_data(img, source_delta_pix=0.05, model_num_pix=10,
                            model_delta_pix=0.05, noise=noise, psf_error=psf)
    assert d.psf_error is not None
    assert np.allclose(d.psf_error, 0.3)
    assert fd.effective_noise(d) is not None
    assert np.allclose(fd.effective_noise(d), np.sqrt(0.1 ** 2 + 0.3 ** 2))


def test_effective_noise_without_psf_error_is_noise():
    img = np.ones((8, 8))
    noise = np.full((8, 8), 0.05)
    d = fd.prepare_fit_data(img, source_delta_pix=0.05, model_num_pix=8,
                            model_delta_pix=0.05, noise=noise)
    assert np.allclose(fd.effective_noise(d), noise)


def test_chi2_uses_effective_noise_including_psf_error():
    """lenstronomy adds model/PSF uncertainty to the variance; chi² must use
    σ_eff = sqrt(σ² + σ_psf²), so a PSF error map lowers (divides out) χ²."""
    data = np.zeros((4, 4))
    model = np.zeros((4, 4))
    model[1, 1] = 1.0
    d_no_bg = fd.FitData(image=data, delta_pix=0.05)
    d = fd.FitData(image=data, delta_pix=0.05, noise=np.full((4, 4), 0.1))
    d_psf = fd.FitData(image=data, delta_pix=0.05, noise=np.full((4, 4), 0.1),
                       psf_error=np.full((4, 4), 0.1))
    chi2_no = fd.chi2(model, d)
    chi2_psf = fd.chi2(model, d_psf)
    assert np.isfinite(chi2_no) and np.isfinite(chi2_psf)
    # same residual, wider effective sigma -> smaller chi², by exactly 0.5x
    assert chi2_psf == pytest.approx(chi2_no * 0.5)
    # without any noise map chi² is still refused
    with pytest.raises(fd.DataPrepError):
        fd.chi2(model, d_no_bg)


def test_psf_error_invalid_pixels_zeroed():
    img = np.ones((6, 6))
    psf = np.full((6, 6), 0.2)
    psf[0, 0] = -1.0
    psf[1, 1] = np.nan
    d = fd.prepare_fit_data(img, source_delta_pix=0.05, model_num_pix=6,
                            model_delta_pix=0.05, noise=np.full((6, 6), 0.1),
                            psf_error=psf)
    assert d.psf_error[0, 0] == 0.0 and np.isfinite(d.psf_error[1, 1])
    assert d.psf_error[1, 1] == 0.0
    assert np.allclose(d.psf_error[2:, 2:], 0.2)
