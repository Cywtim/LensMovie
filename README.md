# LensMovie

An interactive Qt application that visualizes **gravitational lensing** with
[lenstronomy](https://lenstronomy.readthedocs.io/), re-rendered in real time as
you adjust parameters with sliders.

![LensMovie demo](img/VideoSnap_Sequence%2001.gif)

## Features
Single-window layout:
- **Top — 3D scene** (Vispy, GPU), full width and **edge-on** along the line of
  sight: `observer --------- lens plane(s) --------- source`. Lenses sit along
  the line of sight by their redshift (balanced so observer / lenses / source are
  evenly spaced), each drawn as a **translucent mass disk** on its own lens plane
  — sized by its Einstein radius (θ_E, or α_Rs for an NFW halo), offset by its
  sky centre, and fading core→rim; light rays bend in the sky plane at each lens
  plane. Sources are drawn as **extended blobs** sized by their profile radius.
  The default camera is
  a side-on view (drag to rotate, scroll to zoom). Rays bend smoothly at each
  lens plane (Catmull-Rom, for display) instead of kinking sharply. **Controls**:
  left-drag rotate, right-drag / scroll zoom, **middle-drag** or **Shift+left-drag**
  pan. The bar spans the full window width with a fixed height.
- **Display row**: the display settings share one row with the external-image
  file buttons (**Load… / Clear / Noise… / Mask… / PSF…**), visually separated into
  their own labelled `data:` group so they read as two distinct groups. The row
  scrolls horizontally on a narrow window instead of clipping.
- **Display strip**: grid size (numPix), colormap, log/linear stretch, and a
  **3D scene toggle**. Unchecking 3D stops rendering the scene (only the black
  background area remains — the layout does not reflow) and skips rebuilding it,
  which saves the per-update mesh construction / GL upload cost.
- **External image panel** (tall strip at the right of the window, spanning the
  3D bar / display row / fit strip): load your own lensed-image matrix with
  **Load image…** and it is drawn with the same colormap/stretch. Supported
  inputs:
  `.npy` / `.npz`, `.fits` / `.fit` / `.fts` (first image HDU; a cube reduces to
  its first plane), `.mat`, `.txt` / `.csv` / `.dat` / `.tsv`, and image files
  `.png` / `.jpg` / `.tif` / `.bmp` (converted to luminance). The panel reports
  the loaded shape, value range and any conversion applied.
- **Fit the model to your data** (lenstronomy's own `FittingSequence`, PSO):
  a **Fit (PSO)** strip with particle / iteration / restart settings. The fit runs
  on a **background thread** (the GUI stays responsive) and only the **unlocked 🔓**
  parameters are varied — locked ones are held exactly. **Cancel** stops the fit
  promptly (the fit loop polls the cancel flag between restarts and inside each
  swarm iteration) and returns the strip to the idle state. On completion the
  best-fit values are written back into the sliders, the strip reports
  `χ² initial → best (free, ndof) + reduced χ²ν`, and the external data panel
  can show **best-fit model**, **residual**, or a per-pixel **chi2 map**
  (`((data−model)/σ)²`, with the total, worst pixel and the share of pixels
  above 1 — plus a hint whether the excess is spread everywhere, i.e. σ
  underestimated, or concentrated, i.e. a model component missing) through its
  mode selector.
  The raw χ² is a per-pixel sum and grows with the grid (150×150 → ≈22 500 even
  for a perfect model), so judge fits by the **reduced χ²ν ≈ 1**.  When χ²ν
  clearly departs from 1 the strip appends a ⚠ diagnostic: χ²ν≫1 usually means
  the noise σ is underestimated / the PSF error map is missing / a model
  component is missing; χ²ν≪1 means σ is overestimated.
  The strip also has an optional **stop χ²ν** control (default off): when enabled,
  a restart that reaches the target reduced chi² halts the swarm and skips the
  remaining restarts instead of burning iterations on an already-converged fit.
  A second optional **🔒 lock on good fit** control (default off, independent
  threshold, default `1.0`) goes one step further: once a finished fit meets its
  reduced-χ²ν target, **every currently-unlocked 🔓 parameter is automatically
  locked**, pinning the fitted model as the new working point (handy after a fit
  converges — no manual tap-through of every lock button).
  **You can watch it converge**: while the swarm runs, the *lower model panels*
  (Fermat potential, Lens image, time delay, critical curve) live-update with
  the swarm's current best model, and the unlocked sliders track the running
  values (`fitting… iter i/N`, with the running χ² in the fit strip), so a long
  fit is not a black box. The external data panel is left on the user's selected
  mode — previews no longer overwrite it with “fitting…” frames. Preview
  rendering is optional and rate-limited — a **preview** checkbox plus an
  interval (default `0.5 s`).
  Measured cost is negligible either way (a 200-iteration fit took 10.7 s with
  previews off vs 10.0–10.3 s at every 0.1–2 s, i.e. within run-to-run noise), so
  it can normally be left on; uncheck it on a slow machine for the guaranteed
  minimum work.
  - **Export the fit** (buttons light up on the strip after a successful fit):
    **“Chain…”** opens the per-parameter **trajectory preview** in-app (χ² on
    top, each free parameter below) without writing any file; **“Save fit…”**
    writes a dark-theme **data | model | residual** report PNG with the χ²
    before/after; **“Save chain…”** writes the **parameter chain** — one row per
    swarm iteration with the χ² and every free parameter (CSV) plus a
    per-parameter trajectory figure (PNG), so you can see exactly how each
    fitted value converged. Preview and PNG come from the **same** figure
    builder, so they can never disagree. The chain is recorded by the fit loop
    independently of the (optional) image previews, so it is always available.
- **Ready-to-fit example** in `examples/fit/` (regenerate with
  `tools/make_fit_example.py`): a 150×150 lensed image rendered by the app's own
  forward model, blurred by a 0.12″ PSF with noise — plus the noise map, the PSF
  kernel, the exact truth configuration (`truth.json`) and step-by-step
  instructions (`README.md`), so you can load it and run a fit in a few clicks.
- **Fit-data layer** (inputs for the fit):
  - **pixel scale** (`arcsec/px`) is a control in the display strip and drives the
    model grid, so model and data can share one grid.
  - **PSF**: a Gaussian `FWHM` control, or load a PSF **kernel** file.
    `FWHM = 0` keeps the default delta PSF.
  - **Load noise** (a per-pixel σ map): it supplies the **chi² weight** for the
    fit and adds a *display-only* noise realisation to the **"Lens image"** panel
    (so the main lens view can show a noisy observation).  It does **not** change
    the **external image** panel (that always shows the image you loaded) and it
    is **not** added to the fitted data.  No σ map: the chi-squared fit is
    blocked.
  - **Load mask** (same-shape file) for a proper chi-squared likelihood (kept
    pixels only; reported in the “on model grid” summary).
  - **Load PSF err** (a per-pixel 1-sigma PSF-model-error map): combined with the
    noise in **quadrature** into the effective chi² sigma — the same variance-
    addition semantics as lenstronomy's `C_D + |error_map|`.  Feed a map when the
    PSF is uncertain (e.g. from `psf_error_map` conventions) so the fit over-
    weights poorly-known PSF regions less.  Without it the effective sigma is
    exactly the noise map.
  - **“preview on model grid”** resamples the loaded data onto the model grid
    (pixel scale + centre) so the alignment can be verified; the label then
    reports the grid, resampling, noise, mask coverage and PSF.
- **Lower half — a 2-row x 3-column grid**:
  | | col 1 | col 2 | col 3 |
  |---|---|---|---|
  | row 1 | Fermat potential (+ image positions) | Lens image (+ image positions) | **Lenses** config |
  | row 2 | Time delay (+ image positions) | Critical curve + caustic (+ image positions, source star) | **Sources / points** config |
  - The **image positions** (solved per source, colour-coded by source) are
    overlaid on Fermat potential, Time delay, Lens image and Critical curve —
    all four live in the lens plane.  The Critical-curve panel additionally
    marks each **source position** with a gold star and draws each extended
    source's **own extent ellipse** (effective radius + ellipticity e1/e2) in
    the source plane, right next to the caustic — a source straddling the
    caustic is exactly the strongly-magnified / multiple-image regime.  The
    curve panel auto-scales to include the markers and outlines, so nothing is
    clipped.
- **Config panels**:
  - **Cosmology** (top of the config column): the multi-plane background, with
    one slider row per knob — `H0` [km/s/Mpc], `Ωm`, `ΩΛ`, `w0`, `wa` — following
    the same 🔓/🔒 convention as every parameter.  The knobs default to **locked**:
    the cosmological background enters the distance computations (so the
    **time-delay / Fermat fields scale with `1/H0`**, and physical quantities
    change), but a lensed *image* is cosmology-blind with angular θ_E profiles,
    so the image-channel fit cannot constrain them.  Unlocking a knob lets the
    fit sample it via lenstronomy `cosmology_sampling` (harmless, but degenerate
    for image-only fits); the same astropy `w0waCDM` is used by the fit engine
    and the renderer, so the two can never disagree about distances.
  - **Multiple lenses**: each with model selection (`SIS` / `SIE` / `SPEP` /
    `PEMD` / `NFW` / `SIS_TRUNCATED`; PEMD only when `fastell4py` is present), own
    parameters (theta_E, shear, ellipticity, center) and **redshift**; add/remove.
    Entry cards are **auto-numbered** (`Lens 1`, `Lens 2`, …; likewise
    `Source 1`, … and `Point source 1`, …) and renumbered on add/remove — the
    point-source "attach to Source N" dropdown follows the same 1-based labels.
    Each card shows **only the sliders its model actually uses** (e.g. NFW gives
    `R_s`, `alpha_Rs` with no theta_E/ellipticity; `SIS_TRUNCATED` adds `r_trunc`;
    external shear and the centroid stay for every model), and the deflector-light
    sliders follow the chosen light profile (GAUSSIAN vs SERSIC), hidden while the
    light model is `NONE`. Switching models never loses a value — hidden sliders
    keep it and come back on return.
  - **Deflector (lens galaxy) light per lens**: a light-model selector
    (`NONE` / `SERSIC_ELLIPSE` / `SERSIC` / `GAUSSIAN_ELLIPSE` / `GAUSSIAN`) with
    its own amplitude, size, Sersic index, ellipticity and Gaussian sigma. This
    light sits in the image plane and is **not lensed**. It matters for real data:
    without it the deflector's light would be absorbed into the source by a fit.
    Its sliders are disabled while the model is `NONE`.
  - **Sky background**: a constant pedestal added to the model image (display strip).
  - **Multiple sources** — all **extended (resolved)** profiles, selectable per
    source: `SERSIC_ELLIPSE`, `SERSIC`, `GAUSSIAN_ELLIPSE`, `GAUSSIAN`,
    `HERNQUIST`, `CORE_SERSIC`, each with
    position, ellipticity, size (`R_sersic` / `sigma` / `Rs`), `n_sersic` and
    **redshift**; add/remove.
  - **Multiple point sources** (image-plane PSF spikes, configurable in their own
    "Point sources" panel):
    - `LENSED` — a **source-plane** point (lensed quasar/AGN): its multiple
      images and magnification are *solved* from the source plane, amplitudes are
      source-plane fluxes; drawn amber on the source plane in the 3D scene.
    - `UNLENSED` — an image-plane star fixed on the sky (`point_amp`), magenta in
      3D at mid-scene.
    - Each entry can carry its own position/redshift or **attach to a source** —
      the Sources card's **"point source"** checkbox gives a source an AGN at its
      centre; the two controls stay in sync either way.
    - Point-source amplitudes are **integral fluxes**, so they have separate,
      much brighter slider ranges from extended profiles.
  - **Point sources in the fit** — reverse of rendering: a fit works in the
    **image plane** (`LENSED_POSITION`), seeded by a forward solve, while
    rendering solves forward from the source plane (`SOURCE_POSITION`). After the
    fit, the image-plane positions and flux are ray-shot back to the source plane
    so re-rendering reproduces exactly what the fitter saw.
  - Physics via lenstronomy **multi-plane** lensing; one source-plane redshift is
    used as the reference for the 2D scalar fields.
  - **Every parameter is slider _and_ numeric input**: each slider has an editable
    number box beside it, so you can drag for a quick feel or type an exact value
    (the box carries one more decimal than the slider's step). The typed value is
    what the model and any fit use, so precision is not lost to the slider's snap.
  - **Fix (lock) button per parameter**: the 🔓 button beside every lens/source
    slider locks that parameter. A fixed parameter cannot be changed by anything
    — the slider is disabled, the value is frozen, and even programmatic updates
    are rejected. Only the user can release it, by clicking the same button (the
    API enforces this: unfixing requires an explicit ``user=True``).
- Debounced throttled redraw keeps slider dragging smooth.
- All four 2D canvases share one figure size + Expanding size policy, so the grid
  stays aligned and each canvas fills its cell on resize.
- **Resizable panes**: the columns of the 2D grid (two view panes vs the
  configuration pane) and the row between the top block and the grid are
  draggable ``QSplitter`` handles — give the sliders more room, or enlarge the
  external panel, without touching any code.
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
  controller.py    # LensMovieController: program state + operations (UI↔program boundary)
  main_window.py   # presenter: top area (3D + display row + external panel), resizable grid, widget↔controller binding
  controls.py      # config panels (LensesPanel + SourcesPanel + DisplayBar + DataBar + FitBar)
  plotting.py      # matplotlib canvases (Field/Image/Curves/External)
  lensing_calc.py  # lenstronomy physics core (multi-plane, fields, cc/caustic)
  scene3d.py       # vispy -> edge-on 3D scene
  external_image.py# load user-supplied matrices (npy/fits/mat/text/images)
  fit_data.py      # resample data/noise/mask onto the model grid + chi2
  fitting.py       # build FittingSequence inputs, run PSO (locks -> kwargs_fixed)
  fit_worker.py    # QThread wrapper so a fit does not block the GUI
  theme.py         # palette + QSS loader + matplotlib dark style (the app's "skin")
  theme.qss        # the Qt stylesheet — restyle the whole app here
img/               # screenshots and generated test data
tests/
tools/render_screenshot.py  # render the themed window to a PNG for visual checks
DESIGN.md
```

## UI and program are designed separately
The app is split so the interface and the program can be updated independently:

- **Program core** — `lensing_calc.py` (pure physics, no Qt): the stable
  `Config → compute() → SimResult` contract; `fitting.py`, `fit_data.py`.
- **Program state & operations** — `controller.LensMovieController`
  (`controller.py`): owns the external/noise/mask/PSF data, the prepared fit
  data and fit result, and runs the background fit. It knows **nothing about
  widgets**; presenters read state through it and subscribe to its signals.
- **UI / presenter** — `main_window.py` maps widgets ↔ controller and draws;
  `controls.py` builds the panels; `plotting.py` / `scene3d.py` are pure views.
- **Skin** — `theme.qss` + `theme.PALETTE` (theme.py) are the *single* place
  colours/fonts live. Restyling is a one-file change; the physics core never
  imports the theme.

So you can restyle the interface without touching the program, and improve the
physics without touching a single widget — as long as `Config` / `SimResult`
and the controller API stay stable.

## License
MIT
