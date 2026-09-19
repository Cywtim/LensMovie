"""Tests for the vispy 3D scene module.

Pure-geometry parts (spline ray smoothing, middle-click pan maths) run without a
GL context; anything that constructs a real ``Scene3D`` skips when no context is
available.
"""

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication


@pytest.fixture(scope="module")
def _app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_scene3d_module_importable():
    import app.scene3d as s3

    assert hasattr(s3, "Scene3D")
    assert s3.scene  # vispy scene namespace accessible
    assert s3.visuals  # visual classes accessible
    assert s3.PanTurntableCamera  # pan-capable camera subclass


# ------------------------------------------------------------------- smooth rays
def test_spline_path_smooths_a_sharp_kink():
    """A 90° polyline corner must come out of Catmull-Rom as a smooth curve
    (no segment turns more than ~30°) that still passes its endpoints."""
    from app.scene3d import _spline_path

    pts = [[6, 0, 0], [0, 0, 0], [0, 1, 0], [-6, 1, 0]]
    n_seg = 30
    path = _spline_path(pts, n_seg=n_seg)

    # dense sampling: (n_ctrl-1) segments × n_seg samples + final point
    assert len(path) >= (len(pts) - 1) * n_seg
    # endpoint interpolation
    assert np.allclose(path[0], [6, 0, 0])
    assert np.allclose(path[-1], [-6, 1, 0])
    # the bend magnitude is preserved (y rises by ~1 along the ray)
    assert np.isclose(path[-1, 1] - path[0, 1], 1.0, atol=0.05)

    seg = path[1:] - path[:-1]
    n = np.linalg.norm(seg, axis=1)
    d = seg[n > 1e-9] / n[n > 1e-9, None]
    turn = np.arccos(np.clip(np.einsum("ij,ij->i", d[:-1], d[1:]), -1, 1))
    assert np.degrees(turn.max()) < 30, "corner not smoothed"


def test_ray_path_in_a_lens_bends_smoothly(_app):
    """A real single-lens light ray is a smooth path (not a sharp kink at the
    lens plane) that still deflects toward the lens centre."""
    import app.scene3d as s3
    from app import lensing_calc as lc

    try:
        scene = s3.Scene3D(size=(200, 120))
    except Exception as exc:
        pytest.skip(f"no GL context: {exc}")
    try:
        cfg = lc.Config()                       # SIS θ_E=1 at z=0.5, source at z=1.5
        z_max = max([l.redshift for l in cfg.lenses] + [0.3])
        path = scene._ray_path(list(cfg.lenses), z_max,
                               sy=0.0, sz=0.0, y0=0.0, z0=0.2)

        # dense (4 control points → 67 samples); 3rd-lens count scales it
        assert len(path) >= 60
        # endpoints pinned to the source / observer planes
        assert np.isclose(path[0, 0], scene._L)
        assert np.isclose(path[-1, 0], -scene._L)

        # no sharp corner anywhere along the ray
        seg = path[1:] - path[:-1]
        n = np.linalg.norm(seg, axis=1)
        d = seg[n > 1e-6] / n[n > 1e-6, None]
        turn = np.arccos(np.clip(np.einsum("ij,ij->i", d[:-1], d[1:]), -1, 1))
        assert np.degrees(turn.max()) < 20, "ray still kinks at the lens plane"

        # the ray bends toward the lens centre in (z): from +0.2 toward/through 0
        # (starting oﬀ-axis, the θ_E-sized deflection carries it past the centre)
        assert path[-1, 2] < path[0, 2] > 0
        # y is unchanged — the lens sits at y=0 and the ray started on y=0
        assert np.isclose(path[-1, 1], path[0, 1])
    finally:
        scene.close()


# --------------------------------------------------------- middle-button panning
def test_pan_camera_translates_center_on_a_horizontal_drag():
    """A pure-horizontal middle drag must move the camera centre along the
    line-of-sight (x) axis — this is the 'horizontal drag' interaction."""
    from app.scene3d import PanTurntableCamera

    cam = PanTurntableCamera(fov=45, center=(0, 0, 0), elevation=14, azimuth=0)
    # Don't cascade into GL/transform work that needs a real viewbox.
    cam.view_changed = lambda: None

    class _ViewBox:
        size = np.array([800.0, 600.0])

    cam._viewbox = _ViewBox()
    cam._scale_factor = 9.5

    before = np.array(cam.center)
    cam._pan_mouse((200, 300), (500, 300))        # straight right
    delta = np.array(cam.center) - before

    assert delta[0] != 0, "horizontal drag did not pan the camera"
    # for a purely horizontal drag the x (line-of-sight) component dominates
    assert abs(delta[0]) > abs(delta[2])


# ----------------------------------------------------------------- lens disks
def test_lens_disk_radius_follows_einstein_radius():
    """A mass disk scales with the deflector's Einstein scale: theta_E for
    isothermal/power-law profiles, alpha_Rs for an NFW halo, clamped both ends."""
    import app.scene3d as s3
    from app import lensing_calc as lc

    sis = lc.LensParams(model="SIS", theta_E=1.0)
    assert s3._lens_disk_radius(sis) == pytest.approx(1.3)     # 1.5×1.0, clamped
    weak = lc.LensParams(model="SIS", theta_E=0.02)
    assert s3._lens_disk_radius(weak) == pytest.approx(0.10)   # tiny lens floor
    nfw = lc.LensParams(model="NFW", theta_E=5.0, alpha_Rs=0.6)
    assert s3._lens_disk_radius(nfw) == pytest.approx(0.9)     # uses alpha_Rs, not θ_E


def test_lens_disk_geometry_lies_in_the_sky_plane():
    """The disk mesh sits in the y-z plane (x=0), centred on the lens, with
    valid faces and a finite ring-radius drive for the shading."""
    import app.scene3d as s3

    pos, faces, rings = s3._lens_disk_geometry(0.2, -0.1, 0.8)
    assert np.allclose(pos[:, 0], 0.0)                         # sky (y-z) plane
    assert np.isclose(pos[:, 1].max(), 0.2 + 0.8, atol=1e-4)
    assert np.isclose(pos[:, 1].min(), 0.2 - 0.8, atol=1e-4)
    assert np.isclose(pos[:, 2].max(), -0.1 + 0.8, atol=1e-4)
    assert pos[:, 1].min() >= 0.2 - 0.8 - 1e-4
    assert faces.min() >= 0 and faces.max() < len(pos)
    assert rings.min() == 0.0 and rings.max() == pytest.approx(0.8)


def test_update_scene_builds_lens_disks_on_their_planes(_app):
    """Each lens gets one translucent disk, parked at that lens's redshift-xyz."""
    import app.scene3d as s3
    from app import lensing_calc as lc

    try:
        scene = s3.Scene3D(size=(200, 120))
    except Exception as exc:
        pytest.skip(f"no GL context: {exc}")
    try:
        cfg = lc.Config(lenses=[
            lc.LensParams(model="SIS", theta_E=0.6, redshift=0.3),
            lc.LensParams(model="SIS", theta_E=0.9, redshift=0.9),
        ])
        scene.update_scene(cfg, lc.SimResult(
            image=np.zeros((10, 10)), fermat=np.zeros((10, 10)),
            time_delay=np.zeros((10, 10)), cc_ra=np.array([]), cc_dec=np.array([]),
            caustic_ra=np.array([]), caustic_dec=np.array([]),
            image_positions=[], num_pix=10, delta_pix=0.05, ref_z_source=1.5))
        assert len(scene._lens_disks) == 2
        z_max = 0.9
        for disk, lens in zip(scene._lens_disks, sorted(cfg.lenses,
                                                        key=lambda l: l.redshift)):
            x_lens = scene._x_of_redshift(lens.redshift, z_max)
            verts = disk.mesh_data.get_vertices()
            assert np.allclose(verts[:, 0], x_lens)          # on the lens plane
            rgba = disk.mesh_data.get_vertex_colors()
            assert rgba.shape[1] == 4                        # translucent RGBA halo
            assert rgba[:, 3].min() >= 0.0 and rgba[:, 3].max() <= 1.0
    finally:
        scene.close()


def test_update_scene_respects_show_mass_disks(_app):
    """show_mass_disks=False drops the lens disks while keeing rays/blobs."""
    import app.scene3d as s3
    from app import lensing_calc as lc

    try:
        scene = s3.Scene3D(size=(200, 120))
    except Exception as exc:
        pytest.skip(f"no GL context: {exc}")
    try:
        cfg = lc.Config(lenses=[lc.LensParams(model="SIS", theta_E=0.6,
                                              redshift=0.3)])
        scene.update_scene(cfg, lc.SimResult(
            image=np.zeros((10, 10)), fermat=np.zeros((10, 10)),
            time_delay=np.zeros((10, 10)), cc_ra=np.array([]), cc_dec=np.array([]),
            caustic_ra=np.array([]), caustic_dec=np.array([]),
            image_positions=[], num_pix=10, delta_pix=0.05, ref_z_source=1.5),
            show_mass_disks=True)
        assert len(scene._lens_disks) == 1
        scene.update_scene(cfg, lc.SimResult(
            image=np.zeros((10, 10)), fermat=np.zeros((10, 10)),
            time_delay=np.zeros((10, 10)), cc_ra=np.array([]), cc_dec=np.array([]),
            caustic_ra=np.array([]), caustic_dec=np.array([]),
            image_positions=[], num_pix=10, delta_pix=0.05, ref_z_source=1.5),
            show_mass_disks=False)
        assert len(scene._lens_disks) == 0       # stale disks cleared, none rebuilt
    finally:
        scene.close()
