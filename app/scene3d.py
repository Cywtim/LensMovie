"""Vispy 3D scene: lens mass plane + light-ray schematic.

The scene renders:
  * a background lens-mass "disk" in the sky plane (x-y) whose density profile
    (proportional to r**-2 for an SIS) is shown as a height warp + color;
  * a set of light rays travelling from a source plane (far, -z) that bend at the
    lens plane (z=0) and converge toward the observer (+z), illustrating how a
    SIS defocusses/refocuses parallel light by the Einstein angle.

All geometry is driven by the same lens/source parameters used for the 2D image,
so dragging a slider updates both views together.
"""

from __future__ import annotations

import numpy as np

import vispy
from vispy import scene

# Force the PyQt5 backend so the canvas embeds as a Qt widget.
vispy.use("pyqt5")

from vispy.scene import visuals  # noqa: E402


class Scene3D:
    """A vispy SceneCanvas rendering the 3D lensing schematic.

    Exposes ``canvas.native`` (a QWidget) for embedding into PyQt5 and
    ``update_scene(lens, source)`` to re-render from parameters.
    """

    def __init__(self, size=(420, 420)):
        self.canvas = scene.SceneCanvas(
            keys="interactive",
            size=size,
            show=False,
            title="LensMovie 3D — lens mass + rays",
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "turntable"
        self.view.bgcolor = "#111318"

        # Convenient scene bounds.
        self._half_size = 2.6
        self._z_lens = 0.0
        self._z_obs = 2.0
        self._z_src = -2.0
        self._n_rays = 5

        # Visual holders (rebuilt on each update).
        self._disk = None
        self._rays = None
        self._src_markers = None
        self._axis = None

        self._add_axes()
        self.update_scene()

    # ------------------------------------------------------------------ setup
    def _add_axes(self):
        h = self._half_size
        zr = 2.2
        # Three separate single-colour axis lines (avoids Line colour-array quirks).
        axis_defs = [
            ([-h, 0, 0, h, 0, 0], (1.0, 0.4, 0.4, 0.8)),
            ([0, -h, 0, 0, h, 0], (0.4, 1.0, 0.4, 0.8)),
            ([0, 0, -zr, 0, 0, zr], (0.4, 0.6, 1.0, 0.8)),
        ]
        self._axis = []
        for pts, color in axis_defs:
            self._axis.append(
                visuals.Line(
                    pos=np.array(pts).reshape(2, 3),
                    color=color,
                    parent=self.view.scene,
                )
            )

    # ------------------------------------------------------------------ update
    def update_scene(self, lens: dict | None = None, source: dict | None = None):
        """Rebuild the 3D scene from lens/source parameters."""
        lens = lens or {"theta_E": 1.0, "center_x": 0.0, "center_y": 0.0}
        source = source or {"center_x": 0.1, "center_y": -0.1}
        theta_e = float(lens.get("theta_E", 1.0))
        cx, cy = float(lens.get("center_x", 0.0)), float(lens.get("center_y", 0.0))

        # Tear down old visuals (except axes). Handle both single visuals and lists.
        for v in (self._disk, self._rays, self._src_markers):
            items = v if isinstance(v, list) else [v]
            for item in items:
                if item is not None:
                    try:
                        item.parent = None
                    except Exception:
                        pass
        self._disk = None
        self._rays = None
        self._src_markers = None

        self._add_mass_disk(theta_e, cx, cy)
        self._add_rays(theta_e, cx, cy, source)
        self.canvas.update()

    def _add_mass_disk(self, theta_e, cx, cy):
        """A disk whose height/color encode the SIS surface-density ~ r^-2.

        Built as an explicit Mesh (vertices/faces/vertex_colors) to dodge a
        SurfacePlot color-ordering bug in vispy 0.14.
        """
        n = 90
        h = self._half_size
        x = np.linspace(-h, h, n)
        y = np.linspace(-h, h, n)
        X, Y = np.meshgrid(x, y)
        R = np.hypot(X, Y)

        # SIS projected density: sigma ~ 1/(2 r)  -> peak at center.
        eps = 1e-2
        dens = 1.0 / np.maximum(R, eps) ** 2
        dens = dens / dens.max()
        # Warp the mass plane out of the sky plane to visualize density as height.
        Z = 0.15 * dens

        # Centre the plane on the lens position.
        X = X + cx
        Y = Y + cy

        N = n
        positions = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1).astype(np.float32)

        # Triangulate a regular grid: each cell -> two triangles.
        idx = np.arange(N * N).reshape(N, N)
        tri = []
        for i in range(N - 1):
            for j in range(N - 1):
                a = idx[i, j]
                b = idx[i, j + 1]
                c = idx[i + 1, j]
                d = idx[i + 1, j + 1]
                tri.append((a, b, d))
                tri.append((a, d, c))
        faces = np.array(tri, dtype=np.uint32)

        # Per-vertex RGB colors.
        rgb = np.zeros((*dens.shape, 3))
        rgb[..., 2] = 0.55 + 0.45 * dens  # blue brightens with density
        rgb[..., 0] = 0.9 * dens
        rgb[..., 1] = 0.25 * dens
        vertex_colors = rgb.reshape(-1, 3).astype(np.float32)

        self._disk = scene.visuals.Mesh(
            vertices=positions,
            faces=faces,
            vertex_colors=vertex_colors,
            shading="smooth",
            parent=self.view.scene,
        )
        self._disk.set_gl_state(blend=True, depth_test=True)
        self._disk.opacity = 0.8

    def _add_rays(self, theta_e, cx, cy, source):
        """Light rays bending through the lens plane (SIS constant deflection)."""
        # Source-plane starting points slightly scattered around the source.
        sx = float(source.get("center_x", 0.1))
        sy = float(source.get("center_y", -0.1))

        # Choose impact parameters: rays that pass at various distances from center.
        offsets = np.linspace(-0.6, 0.6, self._n_rays)
        pos = []
        for i, off in enumerate(offsets):
            # ray passes above in x = sx + off, y = sy + small spread
            x0 = sx + off
            y0 = sy + 0.15 * (i - self._n_rays // 2)
            # deflection toward center, magnitude = theta_e (SIS), radial direction
            dx = cx - x0
            dy = cy - y0
            dist = max(np.hypot(dx, dy), 1e-6)
            ux, uy = dx / dist, dy / dist
            defl = min(theta_e, 1.6)  # cap so rays stay in view
            x1 = x0 + ux * defl
            y1 = y0 + uy * defl

            z_start = self._z_src
            z_kink = self._z_lens - 0.25
            z_kink2 = self._z_lens + 0.25
            z_end = self._z_obs

            pts = np.array(
                [
                    [x0, y0, z_start],
                    [x0, y0, z_kink],
                    [x1, y1, z_kink2],
                    [x1, y1, z_end],
                ]
            )
            pos.append(pts)

        pos = np.array(pos)  # (n_rays, 4, 3)
        cmap = [
            (1.0, 0.6, 0.2), (0.2, 1.0, 0.6), (0.4, 0.8, 1.0),
            (1.0, 0.4, 0.9), (0.9, 0.9, 0.3),
        ]
        # One contiguous strip of lines per ray, each with a single colour.
        self._rays = []
        for i in range(pos.shape[0]):
            self._rays.append(
                visuals.Line(
                    pos=pos[i],
                    color=cmap[i % len(cmap)] + (1.0,),
                    width=3,
                    connect="strip",
                    parent=self.view.scene,
                )
            )

        # Source-plane starting points (white markers).
        self._src_markers = visuals.Markers(
            pos=pos[:, 0, :],
            face_color=(1, 1, 1, 1),
            edge_color=(0.2, 0.2, 0.2, 1),
            size=10,
            parent=self.view.scene,
        )

    @property
    def native(self):
        """The underlying Qt widget to embed in a PyQt5 layout."""
        return self.canvas.native

    def close(self):
        try:
            self.canvas.close()
        except Exception:
            pass
