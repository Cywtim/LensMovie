"""Vispy 3D scene: lens mass planes + light-ray schematic.

The scene is a full-width bar at the top of the window. It renders:
  * one translucent mass plane per lens, its height/colour encoding the SIS-like
    surface density, and its position along the view axis scaled by its redshift
    (so lenses at different redshifts stack in depth);
  * a set of light-ray strips per source that bend through the lens planes on
    their way from the source side to the observer side.

Construction may fail if no GL context is available; ``MainWindow`` handles that
by degrading to 2D-only.
"""

from __future__ import annotations

import numpy as np

import vispy
from vispy import scene

# Force the PyQt5 backend so the canvas embeds as a Qt widget.
vispy.use("pyqt5")

from vispy.scene import visuals  # noqa: E402

from . import lensing_calc as lc


class Scene3D:
    def __init__(self, size=(760, 320)):
        self.canvas = scene.SceneCanvas(keys="interactive", size=size, show=False,
                                        title="LensMovie 3D")
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "turntable"
        self.view.bgcolor = "#111318"

        self._half = 2.6
        self._z_obs = 2.0
        self._z_src = -2.0
        self._n_rays = 5

        self._stress = []  # stress-test attribute placeholder
        self._disks = []
        self._rays = []
        self._markers = None
        self._add_axes()
        self.update_scene(lc.Config(), _empty_result())

    # ------------------------------------------------------------------ axes
    def _add_axes(self):
        h = self._half
        axis_defs = [
            ([-h, 0, 0, h, 0, 0], (1.0, 0.4, 0.4, 0.8)),
            ([0, -h, 0, 0, h, 0], (0.4, 1.0, 0.4, 0.8)),
            ([0, 0, -self._z_src, 0, 0, self._z_obs], (0.4, 0.6, 1.0, 0.8)),
        ]
        self._axis = []
        for pts, color in axis_defs:
            self._axis.append(
                visuals.Line(pos=np.array(pts).reshape(2, 3), color=color,
                             parent=self.view.scene)
            )

    # ------------------------------------------------------------------ update
    def update_scene(self, config: lc.Config, result: lc.SimResult):
        """Rebuild the scene from the current config and its computed result."""
        for v in (self._disks, self._rays, self._markers):
            items = v if isinstance(v, list) else [v]
            for item in items:
                if item is not None:
                    try:
                        item.parent = None
                    except Exception:
                        pass
        self._disks, self._rays, self._markers = [], [], None

        z_max = max([l.redshift for l in config.lenses] + [0.3])

        # One mass plane per lens, depth offset ∝ redshift.
        for i, lens in enumerate(config.lenses):
            self._disks.append(self._make_mass_plane(lens, z_max))
        self._add_rays(config, result)
        # Reset camera centre to middle of the stack.
        self.view.camera.center = (0, 0, 0)
        self.canvas.update()

    # ------------------------------------------------------------------ mass
    def _make_mass_plane(self, lens, z_max):
        n = 64
        h = self._half
        x = np.linspace(-h, h, n)
        X, Y = np.meshgrid(x, x)
        R = np.hypot(X - lens.center_x, Y - lens.center_y)
        eps = 0.05
        dens = 1.0 / np.maximum(R, eps) ** 2
        dens = dens / dens.max()
        Z = 0.12 * dens * (lens.theta_E / 1.0)
        # Depth offset along the observation axis proportional to redshift.
        z_plane = self._z_src + (lens.redshift / z_max) * (
            self._z_obs - self._z_src
        ) - 0.6

        positions = np.stack(
            [X.ravel(), Y.ravel(), (Z + z_plane).ravel()], axis=-1
        ).astype(np.float32)
        faces = _grid_faces(n, n)

        rgb = np.zeros((n * n, 3))
        rgb[..., 2] = 0.5 + 0.5 * dens.ravel()
        rgb[..., 0] = 0.95 * dens.ravel()
        rgb[..., 1] = 0.2 * dens.ravel()
        mesh = visuals.Mesh(vertices=positions, faces=faces,
                            vertex_colors=rgb.astype(np.float32),
                            shading="smooth", parent=self.view.scene)
        mesh.set_gl_state(blend=True, depth_test=True)
        mesh.opacity = 0.7
        return mesh

    # ------------------------------------------------------------------ rays
    def _add_rays(self, config, result):
        cmap = [(1.0, 0.6, 0.2), (0.2, 1.0, 0.6), (0.4, 0.8, 1.0),
                (1.0, 0.4, 0.9), (0.9, 0.9, 0.3)]
        z_start, z_end = self._z_src, self._z_obs
        self._rays = []
        source_marker_pos = []
        for si, source in enumerate(config.sources):
            sx, sy = source.center_x, source.center_y
            offsets = np.linspace(-0.5, 0.5, self._n_rays)
            for i, off in enumerate(offsets):
                x0, y0 = sx + off, sy + 0.12 * (i - self._n_rays // 2)
                # Total deflection: use total alpha at image pos from result if
                # available, else a heuristic bend toward the mass centroid.
                dx = 0 - x0
                dy = 0 - y0
                dist = max(float(np.hypot(dx, dy)), 1e-3)
                bend = min(_total_theta_e(config.lenses), 1.5)
                x1 = x0 + (dx / dist) * bend
                y1 = y0 + (dy / dist) * bend
                pts = np.array([
                    [x0, y0, z_start],
                    [x0, y0, -0.35],
                    [x1, y1, 0.35],
                    [x1, y1, z_end],
                ])
                color = cmap[si % len(cmap)] + (1.0,)
                self._rays.append(
                    visuals.Line(pos=pts, color=color, width=2.5,
                                 connect="strip", parent=self.view.scene)
                )
            source_marker_pos.append([sx, sy, z_start])
        if source_marker_pos:
            self._markers = visuals.Markers(
                pos=np.array(source_marker_pos), face_color=(1, 1, 1, 1),
                edge_color=(0.2, 0.2, 0.2, 1), size=9, parent=self.view.scene
            )

    @property
    def native(self):
        return self.canvas.native

    def close(self):
        try:
            self.canvas.close()
        except Exception:
            pass


def _grid_faces(nx, ny):
    idx = np.arange(nx * ny).reshape(nx, ny)
    tri = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j], idx[i + 1, j + 1]
            tri.append((a, b, d))
            tri.append((a, d, c))
    return np.array(tri, dtype=np.uint32)


def _total_theta_e(lenses):
    return sum(abs(l.theta_E) for l in lenses)


def _empty_result():
    return lc.SimResult(
        image=np.zeros((10, 10)), fermat=np.zeros((10, 10)),
        time_delay=np.zeros((10, 10)), cc_ra=np.array([]), cc_dec=np.array([]),
        caustic_ra=np.array([]), caustic_dec=np.array([]),
        image_positions=[], num_pix=10, delta_pix=0.05, ref_z_source=1.5,
    )
