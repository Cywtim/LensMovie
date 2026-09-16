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
|  3D scene (Vispy) — edge-on, stretches                |  External image    |
|   observer --------- lens plane(s) --------- source   |  (tall panel —     |
|                                                       |   spans all the     |
|   numPix [..] colormap [..] stretch [..]  (display row)|  top rows at the    |
|   [ data: load image | clear | noise | mask | psf ]    |  right; npy/fits/   |
|   [ Fit | Cancel ]            (fitting strip)          |  mat/... )          |
+------------------------------------------------------+--------------------+
|   Fermat potential  |  Lens image         | Lenses config                  |
|                     |  (+ image positions)|  lens1: model/params/z         |
+---------------------+---------------------+--------------------------------+
|   Time delay        |  Critical curve     | Sources config                 |
|                     |   + caustic         |  source1: pos/shape/z          |
+---------------------+---------------------+--------------------------------+
```
```

All four 2D canvases share one figure size and an Expanding size policy so the
grid stays aligned and each canvas fills its cell on resize.

Panel geometry inside the grid: a sky field is **square** and drawn with equal
aspect, so on a wide cell it is bounded by the cell *height*.  ``plotting``
places the main axes at the largest such square on every resize
(``_MplCanvas._fit_axes_to_widget``): it reserves the same pixel margins and
the same right-side colourbar strip for *every* panel (with or without an
actual colourbar), then centres the square — so every panel gets the same
geometric frame (a uniform look).  A colourbar, when present, hugs the square's
right edge in its own axes.  ``tight_layout`` is deliberately not used, because
it reflows axes whenever a colourbar appears.  Ticks adapt to each panel's own
range: ``MaxNLocator(nbins=5, prune="both")`` keeps a handful of round,
readable ticks strictly *inside* the panel's arcsec limits (the old locator
rounded a ±3.725 field up to ±4.5, floating past the image edge).

Fields of view are **deliberately independent** between the Lens image and the
Critical curve + caustic panels.  The image keeps the model grid's FOV (±3.725″
for the default 150px/0.05″ grid).  The curves panel auto-scales to the curves
themselves (symmetric limits with ~15% padding), so a critical curve or caustic
that extends *past* the grid FOV is shown in full instead of being clipped at
the panel edge — the two panels no longer line up 1:1, by design.  On the
physics side, the critical-curve **trace window adapts to the lens scale**
(``_critical_curve``: lens centre + ~2.5×θ_E, grid spacing scaled with the
window so cost stays ~constant) — a large θ_E ring is returned whole instead of
being clipped/vanishing at a fixed ±2.5 window.

Rows and columns are **resizable** with ``QSplitter`` widgets:
- ``col_split`` (horizontal): the two view panes and the configuration pane sit
  side by side, so the column widths can be dragged.  Initial sizes are seeded
  once (≈3:3:4 for view0 / view1 / config) and later window *resizes* keep the
  user's drags (a single ``_col_split_seeded`` guard in ``_update_grid_width``).
- ``row_split`` (vertical): the whole top block (3D + display row + fitting
  strip, with the external panel beside it) vs the 2D grid.  Its default split
  is seeded from size hints so the top block stays at its content height
  (~306 px: 3D 230 + display 46 + fit) and the grid gets the taller share; the
  user can drag the handle to enlarge the top block (taller external panel).

Filling: a square sky field can only fill a square cell, so the 2x3 grid sits
in a **centred, width-capped band** (``MainWindow._update_grid_width``) and the
3D bar is 230 px tall — the two view columns end up near-square (≈361×342 px
cells, image square ~290 px at the default 1500×1050 window), the square fills
~its binding dimension, and gutters stay symmetric instead of leaving ~150 px of
dead space per side.

Display row notes:
- The display settings and the external-image file buttons sit in **one row**,
  divided by a vertical separator into a `data:` group. The row's natural minimum
  width is ~1400 px, so it lives in a horizontal ``QScrollArea`` (vertical bar
  always off) and scrolls rather than clipping on a narrower window.
- The buttons live in the left column of the top block, so the external panel
  to its right can span the full height of the top area.

Top area notes:
- The top block is a **left column** (3D bar, then the numPix/display row, then
  the fitting strip) with the **external-image panel beside it**, spanning the
  whole left column's height — a tall right-hand strip rather than a small
  square (its vertical size hint is Ignored and it has a modest minimum, so the
  matplotlib canvas inside cannot inflate the top block and shrink the grid).
- The 3D scene has an Expanding (horizontal) / Fixed (vertical) size policy, so it
  widens with the window while the height stays at ``_3d_height`` (230).
- The **3D scene** checkbox in the display strip switches rendering off: only the
  GL canvas is hidden, leaving a black background, so the layout does not reflow
  and the scene rebuild (mesh + GL upload) is skipped.
- The external panel displays matrices loaded by ``external_image.load_image_file``
  (data / on model grid / best-fit model / residual), so it doubles as the
  fit-result viewer.

## Parameters (per lens / per source)
Each lens plane carries: model type, theta_E, shear g1/g2, center x/y, **redshift**.
Each lens also carries **deflector light**: `light_model`
(`NONE` / `SERSIC_ELLIPSE` / `SERSIC` / `GAUSSIAN_ELLIPSE` / `GAUSSIAN`) with
`light_amp`, `light_R_sersic`, `light_n_sersic`, `light_sigma`, `light_e1/e2`.
This is image-plane light and is **not lensed**; it is rendered once and added to
the model image (`_render_lens_light`). `Config.sky_amp` adds a constant pedestal.
Each source is an **extended** profile (`SERSIC_ELLIPSE`, `SERSIC`,
`GAUSSIAN_ELLIPSE`, `GAUSSIAN`) and carries: position, ellipticity, size
(`R_sersic` or `sigma`), `n_sersic`, amplitude and **redshift**.
Display: numPix, **pixel scale (delta_pix, arcsec/px)**, **PSF FWHM**, colormap,
stretch.

Every parameter is a **slider plus an editable number box**. The box is
authoritative (``_Slider.value()`` reads it) and is given one more decimal than
the readout used to show, i.e. finer than the slider's 1000-step quantisation, so
typing ``1.105`` is not snapped to ``1.1044``. Dragging mirrors the slider into the
box; typing moves the slider to its nearest step with the slider's signals blocked,
so the typed value is not bounced back. ``setKeyboardTracking(False)`` makes edits
commit on Enter/focus-out rather than per keystroke.

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
    main.py        # entry point (+ applies the theme)
    controller.py  # LensMovieController: program state + operations; the UI↔program boundary
    main_window.py # presenter: top 3D bar, display strip, 2x3 grid; widget↔controller binding
    controls.py    # LensesPanel + SourcesPanel + DisplayBar + DataBar + FitBar
    plotting.py    # matplotlib canvases (Field/Image/Curves/External), theme-aware
    external_image.py # load user-supplied matrices (npy/fits/mat/text/images)
    fit_data.py    # resample data/noise/mask onto the model grid + chi2
    fitting.py     # FittingSequence inputs + PSO run (locks -> kwargs_fixed)
    fit_worker.py  # QThread wrapper for a non-blocking fit
    lensing_calc.py# lenstronomy physics: multi-plane sim, arrival time (Fermat), cc/caustic, image positions
    scene3d.py     # vispy -> edge-on 3D scene (top bar)
    theme.py       # palette + QSS loader + matplotlib dark style (the app's "skin")
    theme.qss      # Qt stylesheet — restyle the whole app in this one file
  tools/render_screenshot.py  # render the themed window to a PNG (visual checks)
  pyproject.toml
  DESIGN.md
  README.md
```

### UI / program separation (topology)
```
┌────────────────────────────── UI layer (beautify freely) ──────────────────┐
│  theme.py + theme.qss   ← the single "skin" (colours, fonts, widget chrome) │
│  main_window.py         ← presenter: maps widgets ↔ controller, draws views │
│  controls.py / plotting.py / scene3d.py  ← widgets & pure display views     │
└────────────────────────────────────▲────────────────────────────────────────┘
           reads/writes state,       │  Qt signals
           no widget knowledge       │
┌────────────────────────────────────▼────────────────────────────────────────┐
│  controller.LensMovieController    ← owns data/noise/mask/psf, fit data+run │
└────────────────────────────────────▲────────────────────────────────────────┘
           Config / SimResult (stable contracts)
┌────────────────────────────────────▼────────────────────────────────────────┐
│  lensing_calc.py / fitting.py / fit_data.py  ← physics core, no Qt          │
└─────────────────────────────────────────────────────────────────────────────┘
```

Why this shape: the physics core was already a clean `Config → compute →
SimResult` contract; what had leaked into the window (state, styling, fit
coordination) is now owned by `controller.py` (program) and `theme.qss` (ui),
so either half can be updated without touching the other.

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
- **Smooth ray bends**: each ray is a dense **Catmull-Rom spline** through
  control points (source → leave/arrive at every lens plane → observer), with
  each plane's pair spread slightly in x so the spline rounds the kink into a
  smooth bend (max segment turn ~14° vs a ~90° corner before).  Deliberately a
  *display* change — the per-lens deflection amounts/directions are unchanged.
- **Camera interaction** (`PanTurntableCamera`, subclass of TurntableCamera):
  LMB rotate · RMB / scroll zoom · **middle-drag pan** (new) · SHIFT+LMB pan
  (vispy-native) · SHIFT+RMB fov.  The camera centre is *not* reset on scene
  rebuilds, so a pan survives parameter changes.
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

## Fitting (implemented)
``app/fitting.py`` builds lenstronomy's own ``FittingSequence`` inputs:
``kwargs_data_joint`` (a one-band ``multi_band_list`` from ``FitData``),
``kwargs_model`` (lens / lens-light / source lists + multi-plane redshifts) and
``kwargs_params`` = ``[init, sigma, fixed, lower, upper]``. The **lock buttons map
directly onto ``kwargs_fixed``**: only unlocked parameters are free; a parameter
with no UI control is fixed too (otherwise it would become unbounded and free).

Notes learned while building it:
- ``SHEAR``'s fitting parameter names are ``gamma1, gamma2, ra_0, dec_0``; the
  reference point must be supplied or ``LensParam`` raises ``KeyError: 'ra_0'``.
- The fit must use the same ``SUPERSAMPLING_FACTOR`` as the renderer, or the model
  cannot reproduce the data and the fit stalls above the noise floor.
- PSO is stochastic, so ``run_pso`` restarts N times, polishes each with SIMPLEX,
  and keeps the lowest chi-squared solution; ``sigma_scale=4`` gives the initial
  swarm a wide enough spread to find the global solution reliably.
- ``FittingSequence.fit_sequence([['PSO', ...]])`` is a single blocking call with
  no progress hook, so ``_run_swarm_with_preview`` drives the swarm directly
  through ``ParticleSwarmOptimizer.sample()`` — the very generator
  ``FittingSequence.pso`` consumes — building the starting bounds exactly as that
  method does, then pushing the result back with ``FittingSequence.update_state``
  so the SIMPLEX polish continues from it. That is what makes the live preview
  possible without giving up lenstronomy's own optimiser.
- Preview rendering is user-controllable: ``FitBar`` has a **preview** checkbox and
  an interval spin box, forwarded as ``preview``/``preview_interval``. With it
  unchecked the worker passes ``preview=None`` so the fit loop never touches the
  renderer. Benchmarked on a 200-iteration fit: 10.7 s off vs 10.0 s at 0.1 s,
  10.3 s at 0.5 s and 10.3 s at 2 s - the overhead is below run-to-run noise,
  because a preview only fires a handful of times and uses the cheap render path.
- The previewed chi2 is the fitter's **own** objective (``-2*logL``, calibrated
  onto the chi2 scale on the first preview), not a recomputed chi2: the swarm's
  global best is monotonic by construction, whereas recomputing with a different
  noise/mask path wobbled and made the reported progress look non-monotonic.
- Previews use ``lensing_calc.render_image`` (image only) rather than ``compute``,
  which also does the Fermat/time-delay fields, critical curve, caustic and image
  positions. Previews are throttled by *time* (0.35 s) so the extra rendering
  cannot dominate the fit's runtime, and a failing preview is swallowed so it can
  never break a fit.
- A fit of the deflector-light + extended-source model recovers injected
  parameters exactly (theta_E 1.10 -> 1.100, source 0.08/-0.06 -> 0.080/-0.060,
  chi2 -> the noise floor) and locked parameters are provably unchanged.

Earlier items now resolved:
1. ~~optional lens-light / sky-background model~~ — **done** (see above);
2. ~~an optimiser/sampler~~ — **done** with lenstronomy's own FittingSequence;
   MCMC/nested posteriors would still need `emcee`/`dynesty`: — **lenstronomy's own `FittingSequence` is the plan**:
   installing `tqdm` (a one-line dependency) was the only blocker for
   `lenstronomy.Workflow.fitting_sequence`, and a real PSO fit now recovers
   parameters correctly in this env (verified: theta_E 1.10 -> 1.100, source
   0.08/-0.06 -> 0.080/-0.060 in ~0.8 s). `FittingSequence.fit_sequence` offers
   `'PSO'` and `'SIMPLEX'` out of the box (pure Python / scipy). Posterior
   sampling needs extra small packages: `emcee` (or `zeus`) for `'MCMC'`, and
   `dynesty`/`ultranest`/`pymultinest` for nested sampling + evidence;
3. ~~a fitting panel varying only the unlocked parameters, off the GUI thread~~
   — **done** (`FitBar` + `FitWorker`).
