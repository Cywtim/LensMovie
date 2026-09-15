# LensMovie

An interactive Qt application that visualizes **gravitational lensing** with
[lenstronomy](https://lenstronomy.readthedocs.io/), re-rendered in real time as
you adjust parameters with sliders.

## Features
Single-window layout:
- **Top — 3D scene** (Vispy, GPU), full width and **edge-on** along the line of
  sight: `observer --------- lens plane(s) --------- source`. Each lens is a mass
  disk perpendicular to the line of sight, positioned along it by its redshift;
  light rays bend in the sky plane at each lens. Drag to rotate, scroll to zoom.
- **Display strip**: grid size (numPix), colormap, log/linear stretch.
- **Lower half — a 2-row x 3-column grid**:
  | | col 1 | col 2 | col 3 |
  |---|---|---|---|
  | row 1 | Fermat potential | Lens image (+ image positions) | **Lenses** config |
  | row 2 | Time delay | Critical curve + caustic | **Sources** config |
- **Config panels**:
  - **Multiple lenses**: each with model selection (SIS / SIE / PEMD), own
    parameters (theta_E, shear, ellipticity, center) and **redshift**; add/remove.
  - **Multiple sources**: each with position/shape and **redshift**; add/remove.
  - Physics via lenstronomy **multi-plane** lensing; one source-plane redshift is
    used as the reference for the 2D scalar fields.
- Debounced throttled redraw keeps slider dragging smooth.
- All four 2D canvases share one figure size + Expanding size policy, so the grid
  stays aligned and each canvas fills its cell on resize.
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
  main_window.py   # main window: top 3D bar, display strip, 2x3 grid
  controls.py      # LensesPanel + SourcesPanel + DisplayBar
  plotting.py      # matplotlib canvases (Field/Image/Curves)
  lensing_calc.py  # lenstronomy physics core (multi-plane, fields, cc/caustic)
  scene3d.py       # vispy -> edge-on 3D scene
tests/
DESIGN.md
```

## License
MIT
