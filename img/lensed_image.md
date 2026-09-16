# `lensed_image.npy` — generation parameters

A simple, noise-free, PSF-free lensed image produced by LensMovie's own
forward model (`app/lensing_calc.py`, lenstronomy multi-plane).

## File

| Property | Value |
|---|---|
| File | `img/lensed_image.npy` |
| Format | NumPy `.npy`, 2-D `float64`, C order |
| Shape | `150 x 150` (y, x) |
| Data range | `min = 5.606e-09`, `max = 0.06923` |
| Units | surface-brightness-like model units (arbitrary, uncalibrated) |

## Grid

| Property | Value |
|---|---|
| `num_pix` | `150` |
| `delta_pix` | `0.05` arcsec / pixel |
| Field of view | `+/-3.725` arcsec per axis |
| Origin | centred on the lens (sky `(0, 0)` is the image centre) |

The grid follows lenstronomy's `util.make_grid` convention: coordinates run from
`-(num_pix/2 - 0.5) * delta_pix` to `+(num_pix/2 - 0.5) * delta_pix`.

## Lens (deflector)

| Parameter | Value |
|---|---|
| `model` | `SIS` |
| `theta_E` | `1.0` arcsec |
| `center_x`, `center_y` | `0.0`, `0.0` arcsec |
| `redshift` | `0.5` |
| `light_model` | `NONE` (no deflector galaxy light) |

## Source

| Parameter | Value |
|---|---|
| `model` | `SERSIC_ELLIPSE` (extended, resolved) |
| `amp` | `1.0` |
| `R_sersic` | `0.1` arcsec |
| `n_sersic` | `4.0` |
| `e1`, `e2` | `0.1`, `-0.2` |
| `center_x`, `center_y` | `0.12`, `-0.1` arcsec |
| `redshift` | `1.5` |

## Observation model

| Setting | Value |
|---|---|
| PSF | **none** — a 1x1 delta kernel (no convolution) |
| Noise | **none** — the array is a noise-free model |
| Sky background | `0.0` (no pedestal) |
| Supersampling | `3` (pixel integration factor) |

## Verified properties

| Check | Result |
|---|---|
| Lensed? | yes — a centred **Einstein ring** |
| Ring radius | `21 px` = `1.05` arcsec (input `theta_E = 1.0`) |
| Bright pixels (>5% of peak) | `127` of `22500` |
| Ring centring | source is offset by `(0.12, -0.1)`, so the ring is slightly asymmetric — the brightness centroid sits at `(78.9, 70.5)` px while the geometric centre is `74.5` px. This is physically correct for an off-centre source. |

## Reproduce

```python
import numpy as np
from app import lensing_calc as lc

cfg = lc.Config(
    lenses=[lc.LensParams(model="SIS", theta_E=1.0,
                          center_x=0.0, center_y=0.0,
                          redshift=0.5)],
    sources=[lc.SourceParams(model="SERSIC_ELLIPSE", amp=1.0,
                             R_sersic=0.1, n_sersic=4.0,
                             e1=0.1, e2=-0.2,
                             center_x=0.12, center_y=-0.1,
                             redshift=1.5)],
    num_pix=150, delta_pix=0.05,
    psf_kernel=None,   # no PSF
    sky_amp=0.0,       # no sky background
)
np.save("img/lensed_image.npy", lc.compute(cfg).image)
```

## Use in the app

```bash
conda run -n lenstronomy_env python -m app.main
```

Then **Load image…** -> `img/lensed_image.npy`. Because the recipe above matches
the app's defaults (`numPix = 150`, pixel scale `0.05`), switching the panel's
mode to **on model grid** should show it unchanged.

To fit it, also load a noise map (a chi-squared needs one); with noiseless data
use a small constant, e.g. `noise = 1e-4`.
