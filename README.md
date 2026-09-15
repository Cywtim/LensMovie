# LensMovie

An interactive Qt application that visualizes **gravitational lensing** with
[lenstronomy](https://lenstronomy.readthedocs.io/), re-rendered in real time as
you adjust parameters with sliders.

## Features (Phase 1 — 2D image plane)
- Single-window layout: parameter panel (left) + rendered image (right).
- Real-time slider control of:
  - **Lens** (SIS + external shear): Einstein radius `theta_E`, shear `gamma1/2`, center.
  - **Source** (Sersic ellipse): amplitude, size `R_sersic`, ellipticity `e1/e2`, position.
  - **Display**: grid size, colormap, log/linear stretch.
- Debounced throttled redraw keeps dragging smooth.
- A 3D interactive scene (Vispy) is planned as Phase 2 — see `DESIGN.md`.

## Requirements
The app is developed against a conda environment named `lenstronomy_env`:

```
python >=3.9
numpy, scipy, matplotlib, PyQt5, lenstronomy
```

## Run
```bash
conda run -n lenstronomy_env python -m app.main
# or, after editable install:
#   pip install -e .
#   lensmovie
```

## Tests
```bash
conda run -n lenstronomy_env python -m pytest tests/ -q
```
(UI smoke tests use the offscreen Qt platform, so they run without a display.)

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
