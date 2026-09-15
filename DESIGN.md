# LensMovie — Interactive Gravitational Lensing Viewer

## Objective
An interactive Qt application that visualizes gravitational lensing as a function of
user-controlled parameters. Real-time re-render of both a 2D image-plane view and an
interactive 3D scene.

## Confirmed design decisions (from user)
- **Lens model**: multiple lens planes, each selectable (SIS / SPEP-ELLIPSE / PEMD,
  extensible), each with its own redshift and parameters, addable/removable.
- **Source**: multiple sources, each with position/shape and its own redshift,
  addable/removable.
- **Interaction**: parameter controls re-render in real time.
- **3D**: real interactive 3D scene (GPU, Vispy) rendered as a **full-width bar
  across the top** of the window (lens-mass planes + light-ray schematic).
- **Layout** (single window):
  - Top: 3D scene (Vispy), full width.
  - Middle-left: 2D display area, two columns — Fermat potential + time-delay on
    the left; lensed image + critical curve/caustic on the right.
  - Right: config panel for the multiple lenses and multiple sources.
- **Physics**: lenstronomy multi-plane lensing (`LensModel(..., multi_plane=True,
  lens_redshift_list=[...], z_source=...)`); Fermat potential / time delay from
  `arrival_time`; critical curve + caustic from `LensModelExtensions`.
- **Build order**: Phase 1 = 2D only (**done**); Phase 2 = add 3D (**done**);
  Phase 3 = multi-plane, multi-lens/multi-source + new layout (**done**).

## Tech stack
| Layer     | Choice                                        |
|-----------|-----------------------------------------------|
| GUI       | PyQt5                                         |
| 2D image  | lenstronomy (simulation) + matplotlib canvas  |
| 3D scene  | Vispy (GPU, interactive rotate/zoom)          |
| compute   | numpy / scipy                                 |

## Window layout (single window)
```
+-----------------------------------------------------------------------------+
|                          3D scene (Vispy)  — full width bar                  |
+--------------------------+--------------------------+------------------------+
|  Fermat potential (2D)   |   Lens image (2D)        |  Lenses (config list):  |
|                          |   + critical curve       |    lens 1: model sel    |
|  Time delay (2D)         |   + caustic              |    + params + redshift  |
|                          |                          |    lens 2: ... (add/rm) |
|                          |                          |  Sources (config list): |
|                          |                          |    source 1: pos/shape  |
|                          |                          |    + redshift (add/rm)  |
+--------------------------+--------------------------+------------------------+
```

## Parameters (per lens / per source)
Each lens plane carries: model type, theta_E, shear g1/g2, center x/y, **redshift**.
Each source carries: position x/y, R_sersic, e1/e2, amplitude, **redshift**.
Display: numPix, colormap, stretch.

## Modules
```
LensMovie/
  app/
    __init__.py
    main.py        # entry point
    main_window.py # main window: top 3D bar, left 2D cols, right config panel
    controls.py    # multi-lens + multi-source configurable panel (add/remove)
    plotting.py    # matplotlib canvases: image/cc/caustic, Fermat potential, time delay
    lensing_calc.py# lenstronomy physics: multi-plane sim, arrival time (Fermat), cc/caustic, image positions
    scene3d.py     # vispy -> 3D scene (top bar)
  pyproject.toml
  DESIGN.md
  README.md
```

## Physics core (lensing_calc)
- `LensModel(..., multi_plane=True, lens_redshift_list=[...], z_source=...)`.
  Multiple source redshifts ⇒ rebuild the LensModel per source z_source.
- Lensed image: `ImageModel` (LightModel per source, ImageData PIXEL PSF 1x1).
- Fermat potential / time delay: `lens_model.arrival_time` over the 2D grid.
- Critical curve + caustic: `LensModelExtensions.critical_curve_caustics`.
- Image positions: `LensEquationSolver.findBrightImage`.
- All API paths validated against lenstronomy 1.13.2 in the conda env.

## Run
```bash
conda run -n lenstronomy_env python -m app.main
```

## Phase 2 (Vispy) — implemented
- Installed `vispy` into `lenstronomy_env`. Environment has DISPLAY=:1 and
  NVIDIA EGL/GL libs; the qt5 backend embeds as a real QWidget.
- Scene content: translucent lens-mass plane (density ~ r^-2 -> height/color)
  plus colored light-ray strips bending through the lens.
- Implementation notes for vispy 0.14 (encountered during build):
  * `SurfacePlot` + `colors` has an ordering bug (set_vertex_colors before
    faces); build the surface as an explicit `scene.visuals.Mesh` instead.
  * `Line` with per-vertex color arrays + `connect="segments"` trips
    `_interpret_color`; use one `Line` per ray with a single colour and
    `connect="strip"`.
  * `Markers`, `Mesh(vertex_colors=...)`, and single-colour `Line` all work fine.
- The 3D view is optional: `MainWindow` falls back to 2D-only if vispy fails.
