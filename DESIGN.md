# LensMovie — Interactive Gravitational Lensing Viewer

## Objective
An interactive Qt application that visualizes gravitational lensing as a function of
user-controlled parameters. Real-time re-render of both a 2D image-plane view and an
interactive 3D scene.

## Confirmed design decisions (from user)
- **Lens model**: start with SIS, extensible to more models later.
- **Interaction**: parameter sliders re-render in real time.
- **3D**: real interactive 3D scene (GPU, Vispy) rendering a lens-mass plane +
  light-ray schematic, linked to the same parameters and updated live.
- **Layout**: single window — parameter panel left; 2D image (matplotlib) and 3D
  scene (Vispy) on the right.
- **Build order**: Phase 1 = 2D only (no new deps, quick to verify) — **done**;
  Phase 2 = add 3D — **done**.

## Tech stack
| Layer     | Choice                                        |
|-----------|-----------------------------------------------|
| GUI       | PyQt5                                         |
| 2D image  | lenstronomy (simulation) + matplotlib canvas  |
| 3D scene  | Vispy (GPU, interactive rotate/zoom)          |
| compute   | numpy / scipy                                 |

## Window layout (single window)
```
+------------------+-----------------------------------+
|  Parameter panel  |  2D image plane (matplotlib)       |
|  - Lens params    |                                  |
|  - Source params  +-----------------------------------+
|  - Display        |  3D scene (vispy)                |
|    settings       |  lens mass plane + rays          |
|  [Refresh]        |                                  |
+------------------+-----------------------------------+
```

## Parameters
| Group     | Parameter        | Default | Range      |
|-----------|------------------|---------|------------|
| Lens      | theta_E (arcsec) | 1.0     | 0.2 – 3.0  |
|           | shear gamma1     | 0.05    | -0.3 – 0.3 |
|           | shear gamma2     | 0.0     | -0.3 – 0.3 |
|           | lens center x/y  | 0, 0    | -2 – 2     |
| Source    | position x/y     | 0.1, -0.1| -2 – 2    |
|           | R_sersic (arcsec)| 0.1     | 0.02 – 1.0 |
|           | e1, e2           | 0.1, -0.2 | -0.8 – 0.8 |
|           | amplitude        | 1.0     | 0.1 – 5    |
| Display   | numPix           | 150     | 60 – 400   |
|           | colormap         | viridis | (list)     |
|           | stretch          | log     | log/linear |

## Modules
```
LensMovie/
  app/
    __init__.py
    main.py        # entry point
    main_window.py # main window, layout, signal wiring
    controls.py    # parameter slider panel
    sim2d.py       # lenstronomy -> 2D image array (render() function)
    scene3d.py     # vispy -> 3D scene (Phase 2)
  pyproject.toml
  DESIGN.md
  README.md
```

## Simulation core (sim2d)
`render(kwargs_lens, kwargs_light, num_pix, delta_pix) -> np.ndarray`
- Wraps lenstronomy LensModel/LightModel/Data/PSF/ImageModel.
- Validated: SIS + SHEAR lens, SERSIC_ELLIPSE source, PIXEL PSF with 1x1 kernel.

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
