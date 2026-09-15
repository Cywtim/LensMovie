"""Load an external 2D matrix (a user-supplied lensed image) from a file.

Supported inputs:
  * ``.npy`` / ``.npz``          — numpy arrays (first array in an npz)
  * ``.fits`` / ``.fit`` / ``.fts`` — FITS image (first HDU holding 2D data)
  * ``.mat``                     — MATLAB file (largest 2D array)
  * ``.txt`` / ``.csv`` / ``.dat`` / ``.tsv`` — plain text / delimited numbers
  * ``.png`` / ``.jpg`` / ``.jpeg`` / ``.tif`` / ``.tiff`` / ``.bmp`` — images
                                   (converted to luminance)

Everything is normalised to a 2D float array. The returned description string is
shown in the UI so the user can see what was loaded.
"""

from __future__ import annotations

import os

import numpy as np


class ImageLoadError(Exception):
    """Raised when a file cannot be turned into a 2D array."""


def _to_2d(arr, path) -> tuple[np.ndarray, str]:
    """Coerce an arbitrary array to 2D, returning (array, note)."""
    arr = np.asarray(arr)
    note = ""

    if arr.ndim == 0:
        raise ImageLoadError("file contains a single scalar, not an image")

    if arr.ndim == 1:
        # A single row/column: treat as a 1 x N image.
        arr = arr.reshape(1, -1)
        note = "1-D data reshaped to 1xN"
    elif arr.ndim == 3:
        # Colour image (H, W, 3/4) or a data cube: collapse to luminance / first plane.
        if arr.shape[2] in (3, 4):
            rgb = arr[..., :3].astype(float)
            arr = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            note = "colour image -> luminance"
        else:
            # A cube is usually (plane, y, x): take the first plane.
            n_planes = arr.shape[0]
            arr = arr[0]
            note = f"cube of {n_planes} planes -> first plane"
    elif arr.ndim > 3:
        original = arr.shape
        arr = np.squeeze(arr)
        while arr.ndim > 2:
            arr = arr[0]
        note = f"higher-dimensional data {original} -> first plane"

    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ImageLoadError(f"could not reduce data to 2-D (shape {arr.shape})")
    return arr, note


def _largest_2d(mapping) -> np.ndarray:
    """Pick the largest 2-D array out of a dict-like container."""
    best = None
    for key, val in mapping.items():
        if key.startswith("__"):
            continue
        try:
            a = np.asarray(val)
        except Exception:
            continue
        a = np.squeeze(a)
        if a.ndim == 2 and (best is None or a.size > best.size):
            best = a
    if best is None:
        raise ImageLoadError("no 2-D array found in file")
    return best


def load_image_file(path: str) -> tuple[np.ndarray, str]:
    """Load ``path`` into a 2D float array.

    Returns:
        (array, description) where description is a short human-readable summary
        of what was read (including any conversion that was applied).

    Raises:
        ImageLoadError: if the file cannot be interpreted as a 2D image.
    """
    if not os.path.isfile(path):
        raise ImageLoadError(f"file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    note = ""

    try:
        if ext == ".npy":
            raw = np.load(path, allow_pickle=False)
        elif ext == ".npz":
            with np.load(path, allow_pickle=False) as npz:
                keys = list(npz.keys())
                if not keys:
                    raise ImageLoadError("npz archive contains no arrays")
                raw = npz[keys[0]]
                note = f"npz key '{keys[0]}'"
        elif ext in (".fits", ".fit", ".fts"):
            from astropy.io import fits

            with fits.open(path) as hdul:
                raw = None
                for i, hdu in enumerate(hdul):
                    data = getattr(hdu, "data", None)
                    if data is not None and np.asarray(data).squeeze().ndim >= 2:
                        raw = data
                        note = f"HDU {i}"
                        break
                if raw is None:
                    raise ImageLoadError("no image-like HDU found in FITS file")
        elif ext == ".mat":
            from scipy.io import loadmat

            raw = _largest_2d(loadmat(path))
        elif ext in (".txt", ".csv", ".dat", ".tsv"):
            delim = "," if ext == ".csv" else ("\t" if ext == ".tsv" else None)
            raw = np.genfromtxt(path, delimiter=delim)
        elif ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            import matplotlib.image as mpimg

            raw = mpimg.imread(path)
        else:
            # Last resort: try numpy, then text.
            try:
                raw = np.load(path, allow_pickle=False)
            except Exception:
                raw = np.genfromtxt(path)
    except ImageLoadError:
        raise
    except Exception as exc:
        raise ImageLoadError(f"{type(exc).__name__}: {exc}") from exc

    arr, conv_note = _to_2d(raw, path)
    if conv_note:
        note = f"{note}; {conv_note}" if note else conv_note

    finite = np.isfinite(arr)
    if not finite.any():
        raise ImageLoadError("array contains no finite values")
    if not finite.all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        note = f"{note}; NaNs/infs zeroed" if note else "NaNs/infs zeroed"

    desc = (
        f"{os.path.basename(path)}  shape={arr.shape[0]}x{arr.shape[1]}  "
        f"min={arr.min():.3g} max={arr.max():.3g}"
    )
    if note:
        desc += f"  [{note}]"
    return arr, desc
