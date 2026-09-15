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
+------------------------------------------------------+--------------------+
|        3D scene (Vispy) — edge-on, stretches          |  External image    |
|         observer --------- lens plane(s) --------- source | (square, loads |
|                                                       |  npy/fits/mat/...) |
+------------------------------------------------------+--------------------+
|   numPix [..]    colormap [..]    stretch [..]      (display strip)        |
+---------------------+---------------------+-------------------------------+
|  Fermat potential   |  Lens image          |  Lenses config                |
|                     |  (+ image positions) |   lens1: model/params/z       |
+---------------------+---------------------+-------------------------------+
|  Time delay         |  Critical curve      |  Sources config               |
|                     |  + caustic           |   source1: pos/shape/z        |
+---------------------+---------------------+-------------------------------+
```

All four 2D canvases share one figure size and an Expanding size policy so the
grid stays aligned and each canvas fills its cell on resize.

Top row notes:
- The 3D scene has an Expanding (horizontal) / Fixed (vertical) size policy, so it
  widens with the window while the height stays at ``_3d_height`` (280).
- The **3D scene** checkbox in the display strip switches rendering off: only the
  GL canvas is hidden, leaving a black background, so the layout does not reflow
  and the scene rebuild (mesh + GL upload) is skipped.
- The external-image panel is a fixed square of side ``_3d_height`` on the right of
  the same row and displays matrices loaded by ``external_image.load_image_file``.

## Parameters (per lens / per source)
Each lens plane carries: model type, theta_E, shear g1/g2, center x/y, **redshift**.
Each source is an **extended** profile (`SERSIC_ELLIPSE`, `SERSIC`,
`GAUSSIAN_ELLIPSE`, `GAUSSIAN`) and carries: position, ellipticity, size
(`R_sersic` or `sigma`), `n_sersic`, amplitude and **redshift**.
Display: numPix, **pixel scale (delta_pix, arcsec/px)**, **PSF FWHM**, colormap,
stretch.

Every parameter slider carries a **fix (lock)** toggle. Fixing freezes the value:
the slider is disabled and both ``_Slider.set_value`` and a direct
``QSlider.setValue`` are reverted, so no code path can change it. Unfixing is
reserved for the user — ``_Slider.set_fixed(False)`` raises ``PermissionError``
unless called with ``user=True``, which only the lock button handler does.

Source model -> lenstronomy kwargs (all resolved/extended, none are point sources):
| model | kwargs |
|---|---|
| SERSIC_ELLIPSE | amp, R_sersic, n_sersic, e1, e2, center_x, center_y |
| SERSIC | amp, R_sersic, n_sersic, center_x, center_y |
| GAUSSIAN_ELLIPSE | amp, sigma, e1, e2, center_x, center_y |
| GAUSSIAN | amp, sigma, center_x, center_y |

## Modules
```
LensMovie/
  app/
    __init__.py
    main.py        # entry point
    main_window.py # main window: top 3D bar, display strip, 2x3 grid
    controls.py    # LensesPanel + SourcesPanel + DisplayBar (add/remove entries)
    plotting.py    # matplotlib canvases: Field/Image/Curves/External
    external_image.py # load user-supplied matrices (npy/fits/mat/text/images)
    fit_data.py    # resample data/noise/mask onto the model grid + chi2
    lensing_calc.py# lenstronomy physics: multi-plane sim, arrival time (Fermat), cc/caustic, image positions
    scene3d.py     # vispy -> edge-on 3D scene (top bar)
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


## Fitting readiness (data layer)
Groundwork so a modelled lens image can be compared with a loaded image:

- ``Config.delta_pix`` is user-settable (display strip) and defines the model grid;
  ``Config.psf_kernel`` convolves the model. ``lensing_calc.gaussian_psf_kernel``
  builds a normalised kernel from a FWHM (0 -> 1x1 delta).
- ``fit_data.prepare_fit_data`` puts a loaded image (with optional noise and mask)
  onto the model grid: resampling for a pixel-scale change, and a ``center_offset``
  that centres the grid on the lens. Invalid noise pixels are excluded from the mask.
- ``fit_data.FitData`` bundles image/noise/mask/PSF on the grid;
  ``fit_data.chi2`` evaluates a chi-squared over the usable pixels;
  ``fit_data.effective_psf_kernel`` returns the loaded kernel or one from the FWHM.

Still required before an actual fit runs:
1. optional lens-light / sky-background model so lens galaxy light is not absorbed
   into the source;
2. an optimiser/sampler: `scipy.optimize` needs no new dependencies, while
   `lenstronomy.Workflow.fitting_sequence` + PSO/MCMC/nested sampling needs
   `tqdm` (a hard requirement today), `emcee` and `dynesty`;
3. a fitting panel that varies only the *unlocked* parameters
   (``fixed_params()``) and runs off the GUI thread.
