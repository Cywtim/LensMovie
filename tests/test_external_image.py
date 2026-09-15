"""Tests for loading externally supplied lensed-image matrices."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from app.external_image import ImageLoadError, load_image_file


@pytest.fixture(scope="module")
def sample_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("extimgs")
    a = np.random.RandomState(0).rand(40, 55) * 10.0
    np.save(d / "a.npy", a)
    np.savez(d / "a.npz", im=a)
    np.savetxt(d / "a.csv", a, delimiter=",")
    np.savetxt(d / "a.txt", a)

    from astropy.io import fits

    fits.PrimaryHDU(a).writeto(d / "a.fits", overwrite=True)
    fits.PrimaryHDU(np.random.rand(4, 20, 20)).writeto(d / "cube.fits", overwrite=True)

    from scipy.io import savemat

    savemat(d / "a.mat", {"img": a})

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.imsave(d / "a.png", np.random.RandomState(1).rand(30, 30, 3))
    return d, a


def test_load_npy(sample_dir):
    d, a = sample_dir
    arr, desc = load_image_file(str(d / "a.npy"))
    assert arr.shape == (40, 55)
    assert np.allclose(arr, a)
    assert "a.npy" in desc and "40x55" in desc


def test_load_npz(sample_dir):
    d, a = sample_dir
    arr, desc = load_image_file(str(d / "a.npz"))
    assert arr.shape == (40, 55)
    assert np.allclose(arr, a)


def test_load_fits(sample_dir):
    d, a = sample_dir
    arr, desc = load_image_file(str(d / "a.fits"))
    assert arr.shape == (40, 55)
    assert np.allclose(arr, a)
    assert "HDU" in desc


def test_load_fits_cube_takes_first_plane(sample_dir):
    d, _ = sample_dir
    arr, desc = load_image_file(str(d / "cube.fits"))
    assert arr.shape == (20, 20)          # first plane, not a (4, 20) slice
    assert "first plane" in desc


def test_load_mat(sample_dir):
    d, a = sample_dir
    arr, _ = load_image_file(str(d / "a.mat"))
    assert arr.shape == (40, 55)
    assert np.allclose(arr, a)


def test_load_text_formats(sample_dir):
    d, a = sample_dir
    for name in ("a.csv", "a.txt"):
        arr, _ = load_image_file(str(d / name))
        assert arr.shape == (40, 55)
        assert np.allclose(arr, a)


def test_load_colour_image_becomes_luminance(sample_dir):
    d, _ = sample_dir
    arr, desc = load_image_file(str(d / "a.png"))
    assert arr.ndim == 2
    assert arr.shape == (30, 30)
    assert "luminance" in desc


def test_missing_file_raises():
    with pytest.raises(ImageLoadError):
        load_image_file("/tmp/definitely-not-here-12345.npy")


def test_non_image_file_raises(tmp_path):
    p = tmp_path / "words.txt"
    p.write_text("hello world\nthis is not a matrix\n")
    with pytest.raises(ImageLoadError):
        load_image_file(str(p))
