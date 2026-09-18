"""LensMovieController: the program side of the app lives here, independently
of any widget.  These tests prove the controller can hold state, build fit data
and drive a fit without a single QWidget being constructed."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication


@pytest.fixture(scope="module")
def ctrl():
    from app.controller import LensMovieController

    c = LensMovieController()
    yield c
    c.deleteLater()


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_controller_holds_external_state_without_widgets(ctrl):
    arr = np.ones((8, 8))
    ctrl.external_array = arr
    assert ctrl.external_array is arr
    ctrl.noise_array = np.full((8, 8), 0.1)
    ctrl.psf_kernel = np.ones((3, 3)) / 9.0
    assert ctrl.current_psf_kernel({}).sum() == pytest.approx(1.0)
    ctrl.clear_external()
    assert ctrl.external_array is None
    assert ctrl.fit_data is None


def test_controller_psf_kernel_from_fwhm(ctrl):
    ctrl.psf_kernel = None
    delta = np.asarray(ctrl.current_psf_kernel({"psf_fwhm": 0.0, "delta_pix": 0.05}))
    assert delta.shape == (1, 1)
    blur = np.asarray(ctrl.current_psf_kernel({"psf_fwhm": 0.6, "delta_pix": 0.05}))
    assert blur.shape[0] > 1
    assert blur.sum() == pytest.approx(1.0)


def test_controller_prepares_fit_data(ctrl):
    ctrl.external_array = np.random.RandomState(0).rand(40, 40)
    ctrl.noise_array = np.full((40, 40), 0.05)
    display = {"num_pix": 30, "delta_pix": 0.05, "psf_fwhm": 0.0}
    fd = ctrl.prepare_fit_data(display)
    assert fd is not None
    assert fd.image.shape == (30, 30)
    assert ctrl.fit_data is fd


def test_controller_psf_error_flows_into_fit_data(ctrl, tmp_path):
    from app import fit_data as fdm

    ctrl.external_array = np.random.RandomState(1).rand(32, 32)
    ctrl.noise_array = np.full((32, 32), 0.05)
    np.save(tmp_path / "psf_err.npy", np.full((32, 32), 0.02))
    _, desc = ctrl.load_aux("psf_error", str(tmp_path / "psf_err.npy"))
    assert ctrl.psf_error_array is not None
    display = {"num_pix": 32, "delta_pix": 0.05, "psf_fwhm": 0.0}
    fd = ctrl.prepare_fit_data(display)
    assert fd.psf_error is not None
    assert np.allclose(fd.psf_error, 0.02)
    eff = fdm.effective_noise(fd)
    assert np.allclose(eff, np.sqrt(0.05 ** 2 + 0.02 ** 2))


def test_controller_prepare_fit_data_none_without_image(ctrl):
    ctrl.external_array = None
    assert ctrl.prepare_fit_data({}) is None


def test_controller_load_aux_normalises_psf(tmp_path):
    from app.controller import LensMovieController

    c = LensMovieController()
    kernel = np.ones((5, 5)) * 3.0  # deliberately not normalised
    np.save(tmp_path / "psf.npy", kernel)
    _, desc = c.load_aux("psf", str(tmp_path / "psf.npy"))
    assert "5x5" in desc
    assert c.psf_kernel.sum() == pytest.approx(1.0)
    c.deleteLater()


def test_window_delegates_state_to_controller(qapp):
    """The presenter must not duplicate state: it reads/writes the controller."""
    from app.main_window import MainWindow

    win = MainWindow()
    arr = np.ones((6, 6))
    win._external_array = arr
    assert win.controller.external_array is arr
    win.controller.noise_array = np.zeros((6, 6))
    assert win._noise_array is win.controller.noise_array
    win._psf_kernel = np.ones((3, 3)) / 9.0
    assert win.controller.psf_kernel is win.controller.psf_kernel
    win.close()
    win.deleteLater()


def test_fit_worker_lives_on_the_controller(qapp):
    """The background fit worker belongs to the controller, not the widgets."""
    from app.main_window import MainWindow

    win = MainWindow()
    assert win._fit_worker is None
    assert win._fit_worker is win.controller._fit_worker
    win.close()
    win.deleteLater()
