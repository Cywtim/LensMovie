"""Vispy 3D scene, edge-on line-of-sight schematic.

Coordinate convention (a genuine side / edge-on view):

        observer                                      source
          |                                             |
   x = -L +---------+---------+---------+---------+---------+ x = +L
    (left)     lens1   (z sorted      lens2)          lens3   (right)
               \                                            /
    light      \                                          /
    travels      \_______________________________________/   rays bend in (y,z)
    right-to-left    at each lens plane

  * the line of sight runs along **X** (horizontal and wider than tall);
  * the sky (angular) coordinates are **Y** and **Z** — each lens is drawn as a
    translucent mass disk lying in the y-z plane on its own lens plane, centred
    at (x_lens, 0, 0) plus the lens's sky offset, sized by its Einstein radius;
  * x_lens is scaled by the lens redshift between the observer (-x) and the
    source (+x), so lenses at higher redshift sit closer to the source;
  * light rays travel from the source side (+x) to the observer side (-x),
    bending in (y,z) at each lens plane — each ray is drawn as a *smooth*
    Catmull-Rom curve (rounded at every plane) rather than a sharp kink.

Construction may fail if no GL context is available; MainWindow degrades to 2D.
"""

from __future__ import annotations

import numpy as np

import vispy
from vispy import scene
from vispy.scene import cameras

vispy.use("pyqt5")

from vispy.scene import visuals  # noqa: E402

from . import lensing_calc as lc

# One colour per source, shared by that source's rays, blob and marker.
_SOURCE_COLORS = [
    (1.0, 0.6, 0.2), (0.2, 1.0, 0.6), (0.4, 0.8, 1.0),
    (1.0, 0.4, 0.9), (0.9, 0.9, 0.3),
]

# One dim colour per lens, used for its translucent mass disk on the lens plane.
_LENS_COLORS = [
    (0.62, 0.55, 0.95), (0.95, 0.62, 0.55), (0.55, 0.85, 0.78),
    (0.93, 0.82, 0.45), (0.72, 0.62, 0.5),
]


class PanTurntableCamera(cameras.TurntableCamera):
    """TurntableCamera that ALSO pans on a plain middle-button drag.

    vispy's TurntableCamera only translates the centre on **SHIFT + LMB**, so
    without a hint the scene seems 'rotate-only'.  This subclass reuses the
    parent's exact translation math for the middle button (button 4): middle-drag
    moves the scene horizontally/vertically, keeping LMB rotate, RMB/scroll zoom,
    SHIFT+LMB pan and SHIFT+RMB fov unchanged.
    """

    def _pan_mouse(self, p1, p2):
        """Translate the camera centre for a press -> current mouse movement."""
        norm = float(np.mean(self._viewbox.size))
        if norm <= 1e-6:
            return  # viewbox not sized yet (e.g. offscreen construction)
        if self._event_value is None or len(self._event_value) == 2:
            self._event_value = self.center
        dist = (np.asarray(p1, float) - np.asarray(p2, float)) / norm \
            * self._scale_factor
        dist[1] *= -1
        dx, dy, dz = self._dist_to_trans(dist)
        ff = self._flip_factors
        up, forward, right = self._get_dim_vectors()
        dx, dy, dz = right * dx + forward * dy + up * dz
        dx, dy, dz = ff[0] * dx, ff[1] * dy, dz * ff[2]
        c = self._event_value
        self.center = (c[0] + dx, c[1] + dy, c[2] + dz)

    def viewbox_mouse_event(self, event):
        if event.handled or not self.interactive:
            return
        # Middle-button drag = pan (button 4 in vispy's button set).
        if event.type == "mouse_move" and event.press_event is not None \
                and 4 in getattr(event, "buttons", ()) \
                and 1 not in event.buttons and 2 not in event.buttons:
            self._pan_mouse(event.mouse_event.press_event.pos,
                            event.mouse_event.pos)
            event.handled = True
            return
        super().viewbox_mouse_event(event)


class Scene3D:
    def __init__(self, size=(1080, 300)):
        self.canvas = scene.SceneCanvas(keys="interactive", size=size, show=False,
                                        title="LensMovie 3D — observer / lens / source")
        self.view = self.canvas.central_widget.add_view()
        # Side-on ("edge-on") default view: the line of sight (X) runs horizontally
        # while the sky plane (Y-Z) is seen nearly edge-on. A small elevation adds
        # depth so the lens disks are readable; the user can drag to rotate, and
        # middle-drag (or SHIFT+LMB) to pan.
        self.view.camera = PanTurntableCamera(
            elevation=14, azimuth=0, distance=9.5, fov=45, center=(0, 0, 0)
        )
        self.view.bgcolor = "#111318"

        # Geometry bounds at init time (updated per frame too).
        self._L = 2.4            # half line-of-sight (x) extent
        self._half = 2.6         # sky (y/z) half extent
        self._n_rays = 5

        self._rays = []
        self._blobs = []
        self._lens_disks = []
        self._markers = None
        self._point_markers = None
        self._add_axes()
        self.update_scene(lc.Config(), _empty_result())

    # ------------------------------------------------------------------ axes
    def _add_axes(self):
        h = self._half
        L = self._L
        axis_defs = [
            # line of sight (X): observer --- source
            ([-L, 0, 0, L, 0, 0], (1.0, 0.4, 0.4, 0.9)),
            # sky axes Y and Z
            ([0, -h, 0, 0, h, 0], (0.4, 1.0, 0.4, 0.8)),
            ([0, 0, -h, 0, 0, h], (0.4, 0.6, 1.0, 0.8)),
        ]
        self._axis = []
        labels = ["observer", "source"]
        for pts, color in axis_defs:
            self._axis.append(
                visuals.Line(pos=np.array(pts).reshape(2, 3), color=color,
                             parent=self.view.scene)
            )
        # observer / source markers on the line of sight
        self._obs_marker = visuals.Markers(
            pos=np.array([[-L, 0, 0]]), face_color=(1.0, 0.4, 0.4, 1.0),
            size=12, parent=self.view.scene)
        self._src_marker = visuals.Markers(
            pos=np.array([[L, 0, 0]]), face_color=(0.4, 1.0, 0.4, 1.0),
            size=12, parent=self.view.scene)
        self._labels = []
        self._labels.append(visuals.Text("observer", pos=(-L, -0.25, 0), color=(1, 0.6, 0.6, 1),
                                         font_size=12, parent=self.view.scene))
        self._labels.append(visuals.Text("source", pos=(L, 0.25, 0), color=(0.6, 1.0, 0.6, 1),
                                         font_size=12, parent=self.view.scene))

    # ------------------------------------------------------------------ update
    def update_scene(self, config: lc.Config, result: lc.SimResult,
                     show_mass_disks: bool = True):
        """Rebuild the edge-on scene from the current config and result.

        ``show_mass_disks`` lets the caller hide the translucent lens disks
        (e.g. to declutter the rays); rays and source blobs are always drawn.
        """
        for v in (self._rays, self._markers, self._blobs,
                  self._lens_disks, self._point_markers):
            items = v if isinstance(v, list) else [v]
            for item in items:
                if item is not None:
                    try:
                        item.parent = None
                    except Exception:
                        pass
        self._rays, self._markers, self._blobs = [], None, []
        self._lens_disks = []
        self._point_markers = None

        z_max = max([l.redshift for l in config.lenses] + [0.3])

        # One translucent mass disk per lens, on its own lens plane (matched to
        # where the rays bend below).
        if show_mass_disks:
            for li, lens in enumerate(sorted(config.lenses,
                                             key=lambda l: l.redshift)):
                self._lens_disks.append(self._make_lens_disk(lens, li, z_max))

        # One extended source blob per source, on the source plane.
        for si, source in enumerate(config.sources):
            self._blobs.append(self._make_source_blob(source, si))
        self._add_rays(config)
        self._add_point_markers(config)
        # NB: the camera centre is deliberately NOT reset here, so a pan (middle-
        # drag / SHIFT+LMB) survives parameter changes instead of snapping back.
        self.canvas.update()

    def _x_of_redshift(self, z, z_max):
        """Map a redshift to an X (line-of-sight) position between observer and source.

        A margin keeps clear gaps between the observer plane, the lens planes and
        the source plane so the three are visually balanced.
        """
        f = max(0.0, min(1.0, z / z_max))
        margin = 0.55
        return -self._L * margin + f * (2 * self._L * margin)

    # ------------------------------------------------------- lens mass disks
    def _make_lens_disk(self, lens, index, z_max):
        """A translucent mass disk on the lens plane, lying in the (y,z) sky plane.

        The radius follows the deflector's Einstein radius (theta_E — or
        alpha_Rs for an NFW halo), so strengthening the lens visibly grows the
        disk; it sits at the lens centre and redshift, exactly where the rays
        bend.  Deliberately dim/greyed: this is the *mass*, not light.
        """
        radius = _lens_disk_radius(lens)
        x = self._x_of_redshift(lens.redshift, z_max)
        positions, faces, rr = _lens_disk_geometry(lens.center_y,
                                                   lens.center_x, radius)
        positions[:, 0] = x                        # move onto the lens plane
        base = _LENS_COLORS[index % len(_LENS_COLORS)]
        shade = (1.0 - 0.6 * rr)[:, None]
        rgb = (np.array(base)[None, :] * shade).astype(np.float32)
        # Soft halo: core a bit more opaque, edge fades away.
        alpha = (0.14 + 0.26 * (1.0 - 0.8 * rr))[:, None].astype(np.float32)
        rgba = np.concatenate([rgb, alpha], axis=-1)
        mesh = visuals.Mesh(vertices=positions, faces=faces, vertex_colors=rgba,
                            shading="smooth", parent=self.view.scene)
        mesh.set_gl_state(blend=True, depth_test=True)
        return mesh

    # ------------------------------------------------------- extended source
    def _make_source_blob(self, source, index):
        """An extended (resolved) source rendered as a disk on the source plane.

        Its radius follows the profile's effective radius (R_sersic or sigma) so
        changing the source size visibly changes the blob, and its face colour
        matches the colour of that source's rays.
        """
        n = 24
        # Scale the angular size up so small sources stay visible, but keep the
        # relative size ordering between sources.
        radius = min(max(source.effective_radius() * 2.5, 0.12), 1.4)
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = np.linspace(0, radius, 6)
        T, R = np.meshgrid(t, r)
        Y = source.center_y + R * np.cos(T)
        Z = source.center_x + R * np.sin(T)
        X = np.full_like(Y, self._L)          # source plane

        positions = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1).astype(np.float32)
        faces = _grid_faces(*R.shape)

        # Bright core, dimmer edge (a soft extended blob).
        rr = (R / max(radius, 1e-6)).ravel()
        base = _SOURCE_COLORS[index % len(_SOURCE_COLORS)]
        shade = (1.0 - 0.7 * rr)[:, None]
        rgb = (np.array(base)[None, :] * shade).astype(np.float32)

        mesh = visuals.Mesh(vertices=positions, faces=faces, vertex_colors=rgb,
                            shading="smooth", parent=self.view.scene)
        mesh.set_gl_state(blend=True, depth_test=True)
        mesh.opacity = 0.9
        return mesh

    # ------------------------------------------------------------------ rays
    def _ray_path(self, lenses, z_max, sy, sz, y0, z0, dw=0.22):
        """One light ray's smooth 3-D path from the source to the observer.

        Control points: source plane → (arrive / leave) per lens → observer
        plane, with each arrive/leave pair spread ``dw`` to either side of the
        formal lens plane so the Catmull-Rom spline *rounds* the kink into a
        smooth bend instead of a sharp corner.  This is for **display**: the
        per-lens deflection magnitudes and directions are unchanged — only the
        transition is smoothed (a ray would physically turn sharply at a plane).
        """
        ys, zs = y0, z0
        pts = [[self._L, ys, zs]]                    # source plane (right)
        for lens in lenses:
            x_lens = self._x_of_redshift(lens.redshift, z_max)
            dy = lens.center_y - ys
            dz = lens.center_x - zs
            dist = max(float(np.hypot(dy, dz)), 1e-3)
            # Bend scale per model: an NFW halo's deflection is set by alpha_Rs,
            # all isothermal/power-law profiles by theta_E.
            if lens.model == "NFW":
                bend = min(abs(lens.alpha_Rs), 1.6)
            else:
                bend = min(abs(lens.theta_E), 1.6)
            ys2 = ys + (dy / dist) * bend
            zs2 = zs + (dz / dist) * bend
            pts.append([x_lens + dw, ys, zs])        # arrive (before bend)
            pts.append([x_lens - dw, ys2, zs2])      # leave  (after bend)
            ys, zs = ys2, zs2
        pts.append([-self._L, ys, zs])               # observer plane (left)
        return _spline_path(pts)

    def _add_rays(self, config):
        cmap = _SOURCE_COLORS
        lenses = sorted(config.lenses, key=lambda l: l.redshift)
        z_max = max([l.redshift for l in config.lenses] + [0.3])

        self._rays = []
        markers = []
        for si, source in enumerate(config.sources):
            sy, sz = source.center_y, source.center_x
            offsets = np.linspace(-0.55, 0.55, self._n_rays)
            for i, off in enumerate(offsets):
                y0 = sy + off
                z0 = sz + 0.12 * (i - self._n_rays // 2)
                path = self._ray_path(lenses, z_max, sy, sz, y0, z0)
                color = cmap[si % len(cmap)] + (1.0,)
                self._rays.append(
                    visuals.Line(pos=path.astype(np.float32), color=color,
                                 width=2.2, connect="strip",
                                 parent=self.view.scene)
                )
            markers.append(([self._L, sy, sz], cmap[si % len(cmap)] + (1.0,)))
        if markers:
            self._markers = visuals.Markers(
                pos=np.array([m[0] for m in markers]),
                face_color=np.array([m[1] for m in markers], dtype=np.float32),
                edge_color=(0.15, 0.15, 0.15, 1), size=6, parent=self.view.scene)

    def _add_point_markers(self, config):
        """Bright point-source markers.

        A LENSED point source sits on the source plane (it is lensed into
        multiple images like the extended source); an UNLENSED star lives in the
        image plane, drawn mid-scene at x=0 (unaffected by the lens).
        """
        pts, colors = [], []
        for point in config.point_sources:
            ra, dec, _ = point.position_and_redshift(config)
            if point.model == "UNLENSED":
                pts.append([0.0, dec, ra])            # image plane, mid-scene
                colors.append((1.0, 0.55, 1.0, 1.0))  # magenta star
            else:
                pts.append([self._L, dec, ra])        # source plane
                colors.append((1.0, 0.9, 0.3, 1.0))   # amber
        if pts:
            self._point_markers = visuals.Markers(
                pos=np.array(pts, dtype=np.float32),
                face_color=np.array(colors, dtype=np.float32),
                edge_color=(0.1, 0.1, 0.1, 1), size=14,
                parent=self.view.scene)

    @property
    def native(self):
        return self.canvas.native

    def close(self):
        try:
            self.canvas.close()
        except Exception:
            pass


def _lens_disk_radius(lens) -> float:
    """The mass disk's sky-plane radius from the deflector's Einstein scale.

    NFW halos are sized by ``alpha_Rs`` (their deflection scale), every other
    profile by ``theta_E``; a 1.5× factor keeps the disk readable next to the
    rays, clamped so tiny/huge lenses never degenerate the mesh.
    """
    base = abs(lens.alpha_Rs) if lens.model == "NFW" else abs(lens.theta_E)
    return min(max(1.5 * base, 0.10), 1.3)


def _lens_disk_geometry(center_y, center_z, radius, n_edge=28, n_ring=6):
    """A disk mesh lying in the y-z sky plane at x=0.

    Returns ``(positions, faces, rings)``: vertices (N,3) with X left at 0 (the
    caller places the disk on the lens plane), triangle faces, and the per-ring
    radii that drive the core→rim shading — same polar-grid construction as the
    source blob.
    """
    t = np.linspace(0, 2 * np.pi, n_edge, endpoint=False)
    r = np.linspace(0, radius, n_ring)
    T, R = np.meshgrid(t, r)
    Y = center_y + R * np.cos(T)
    Z = center_z + R * np.sin(T)
    X = np.zeros_like(Y)
    positions = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1).astype(np.float32)
    faces = _grid_faces(*R.shape)
    return positions, faces, R.ravel()


def _grid_faces(nx, ny):
    idx = np.arange(nx * ny).reshape(nx, ny)
    tri = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j], idx[i + 1, j + 1]
            tri.append((a, b, d))
            tri.append((a, d, c))
    return np.array(tri, dtype=np.uint32)


def _spline_path(pts, n_seg=22):
    """Dense Catmull-Rom spline through the control points ``pts`` (C¹ smooth).

    Uniform Catmull-Rom interpolates every control point with a continuous first
    derivative, so a polyline that kinks sharply at a lens plane is rendered as a
    smooth curve that still passes through each control point.  ``n_seg`` is the
    number of samples per control segment.
    """
    pts = [np.asarray(p, dtype=float) for p in pts]
    if len(pts) < 2:
        return np.array(pts)
    # Duplicate the endpoints so every real segment is an interior segment.
    p = [pts[0]] + pts + [pts[-1]]
    out = []
    for i in range(len(p) - 3):
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        for k in range(n_seg):
            t = k / n_seg
            out.append(
                0.5 * (2 * p1
                       + (-p0 + p2) * t
                       + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                       + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t)
            )
    out.append(p[-2])
    return np.array(out)


def _empty_result():
    return lc.SimResult(
        image=np.zeros((10, 10)), fermat=np.zeros((10, 10)),
        time_delay=np.zeros((10, 10)), cc_ra=np.array([]), cc_dec=np.array([]),
        caustic_ra=np.array([]), caustic_dec=np.array([]),
        image_positions=[], num_pix=10, delta_pix=0.05, ref_z_source=1.5,
    )
