# LensMovie

An interactive Qt application that visualizes **gravitational lensing** with
[lenstronomy](https://lenstronomy.readthedocs.io/), re-rendered in real time as
you adjust parameters with sliders.

## Features
Single-window layout, three areas:
- **Top — 3D scene** (Vispy, GPU): one translucent lens-mass plane per lens (with
  depth scaled by redshift) plus light-ray strips; drag to rotate, scroll to zoom.
- **Middle-left — 2D views**, two columns:
  - Fermat potential (relative) + time-delay (relative) 2D fields.
  - Lensed image with **critical curve**, **caustic** and per-source image positions.
- **Right — config panel**:
  - **Multiple lenses**: each with model selection (SIS / SIE / PEMD), own
    parameters (theta_E, shear, ellipticity, center) and **redshift**; add/remove.
  - **Multiple sources**: each with position/shape and **redshift**; add/remove.
  - Physics via lenstronomy **multi-plane** lensing; one source-plane redshift is
    used as the reference for the 2D scalar fields.
- Debounced throttled redraw keeps slider dragging smooth.
- If Vispy/OpenGL is unavailable, the app degrades gracefully to 2D-only.

## Requirements
The app is developed against a conda environment named `lenstronomy_env`:

```
python >=3.9
numpy, scipy, matplotlib, PyQt5, lenstronomy, vispy
```

## Run
Run from the project root with `python -m` so the `app` package is importable:

```bash
cd /home/cyan/Documents/GitHub/LensMovie
conda run -n lenstronomy_env python -m app.main

# or, after editable install (pip install -e .), from anywhere:
#   conda run -n lenstronomy_env lensmovie
```

## Tests
Run from the project root (again, `-m` matters):

```bash
cd /home/cyan/Documents/GitHub/LensMovie
conda run -n lenstronomy_env python -m pytest tests/ -q
```
(UI smoke tests use the offscreen Qt platform and run without a display; the 3D
scene needs a live OpenGL context and is covered by a smoke run with a display.)

## Project layout
```
app/
  main.py          # entry point
  main_window.py   # main window: top 3D bar, 2D columns, config panel
  controls.py      # multi-lens + multi-source configurable panel
  plotting.py      # matplotlib canvases (image/cc/caustic, Fermat, time delay)
  lensing_calc.py  # lenstronomy physics core (multi-plane, fields, cc/caustic)
  scene3d.py       # vispy -> 3D scene
tests/
DESIGN.md
```

## License
MIT
