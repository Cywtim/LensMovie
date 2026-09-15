# LensMovie

An interactive Qt application that visualizes **gravitational lensing** with
[lenstronomy](https://lenstronomy.readthedocs.io/), re-rendered in real time as
you adjust parameters with sliders.

## Features
- Single-window layout: parameter panel (left) + 2D image and 3D scene (right).
- Real-time slider control of:
  - **Lens** (SIS + external shear): Einstein radius `theta_E`, shear `gamma1/2`, center.
  - **Source** (Sersic ellipse): amplitude, size `R_sersic`, ellipticity `e1/e2`, position.
  - **Display**: grid size, colormap, log/linear stretch.
- **2D image plane** via lenstronomy (matplotlib canvas), debounced throttled redraw.
- **3D scene** via Vispy (GPU): lens-mass plane whose density (r^-2) is shown as
  height/color, plus light-ray schematic bending through the lens; mouse-drag to
  rotate and scroll to zoom. Updates live with the same sliders.
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
  main_window.py   # main window, layout, signal wiring
  controls.py      # parameter slider panel
  plotting.py      # matplotlib canvas (2D image)
  sim2d.py         # lenstronomy -> 2D image array
tests/
DESIGN.md
```

## License
MIT
