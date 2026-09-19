"""Export the artefacts of a fit: a report figure and the parameter chain.

Qt-free: the GUI hands in a :class:`~app.fit_data.FitData` + a
:class:`~app.fitting.FitResult` and an output path; this module writes the files
with matplotlib / csv.  The figures reuse the app's dark canvas theme so what is
saved looks like what is on screen.

Exportable artefacts:
  * ``save_fit_report_png``  — data | model | residual | per-pixel χ² report PNG
    (± the reduced-χ²ν diagnostic; the χ² panel needs a noise map),
  * ``save_chain_csv``       — the parameter chain as rows (iteration, χ², free…),
  * ``save_chain_png``       — per-parameter trajectories over the fit.
"""

from __future__ import annotations

import csv

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from . import theme


def _dark_rc():
    """rcParams matching the app's dark canvases (used only while saving)."""
    s = theme.CANVAS_STYLE
    return {
        "figure.facecolor": s["figure_face"],
        "axes.facecolor": s["axes_face"],
        "axes.edgecolor": s["spine"],
        "axes.labelcolor": s["dim"],
        "xtick.color": s["dim"],
        "ytick.color": s["dim"],
        "text.color": s["text"],
        "legend.facecolor": s["legend_face"],
        "legend.edgecolor": s["legend_edge"],
        "legend.labelcolor": s["text"],
    }


def _extent(num_pix: int, delta_pix: float):
    """Same sky extent convention as the live canvases (symmetric about 0)."""
    half = (num_pix / 2 - 0.5) * delta_pix
    return (-half, half, -half, half)


def save_fit_report_png(path: str, data, result) -> str:
    """Write the data | model | residual | per-pixel χ² report to ``path`` (PNG).

    The χ²-map panel needs a noise map; without one the report falls back to
    data | model | residual.  The title also carries the noise-consistency
    diagnostic (reduced χ²ν ≫1/≪1) as a second line when it fires.
    """
    from .fit_data import effective_noise

    image = np.asarray(data.image, dtype=float)
    model = np.asarray(result.model, dtype=float)
    residual = np.asarray(result.residual, dtype=float)
    eff = effective_noise(data)

    with matplotlib.rc_context(_dark_rc()):
        if eff is None:
            vmax = float(np.nanmax(np.abs(residual))) or 1.0
            fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
            panels = [
                (axes[0], image, "magma", None, "data"),
                (axes[1], model, "magma", None, "model"),
                (axes[2], residual, "RdBu_r", (-vmax, vmax), "model − data"),
            ]
        else:
            chi2map = ((image - model) / eff) ** 2
            if data.mask is not None:
                chi2map = np.where(np.asarray(data.mask, dtype=bool),
                                   chi2map, np.nan)
            fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.4))
            vmax = float(np.nanmax(np.abs(residual))) or 1.0
            panels = [
                (axes[0], image, "magma", None, "data"),
                (axes[1], model, "magma", None, "model"),
                (axes[2], residual, "RdBu_r", (-vmax, vmax), "model − data"),
                (axes[3], chi2map, "viridis", None, "per-pixel χ²"),
            ]
        extent = _extent(data.num_pix, data.delta_pix)
        for ax, arr, cmap, clim, title in panels:
            im = ax.imshow(arr, origin="lower", extent=extent,
                           cmap=cmap, interpolation="nearest")
            if clim is not None:
                im.set_clim(*clim)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("arcsec", fontsize=8)
            ax.set_ylabel("arcsec", fontsize=8)
            ax.tick_params(labelsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03) \
                .ax.tick_params(labelsize=7)
        title = (f"LensMovie fit — χ²: {result.chi2_before:.4g}"
                 f" → {result.chi2_after:.4g}"
                 f"    reduced χ²ν {result.reduced_chi2:.4g}"
                 f"    free: {result.n_free}    ndof: {result.ndof}")
        hint = result.noise_hint()
        if hint:
            title += "\n" + hint
        fig.suptitle(title, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


def save_chain_csv(path: str, result) -> str:
    """Write the parameter chain as CSV: iteration, chi2, one column per free
    parameter.  NaN entries are written as blanks so spreadsheets stay clean."""
    names = list(result.chain)
    iter_nums = list(result.chain_iter)
    chi2s = list(result.chain_chi2)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["iteration", "chi2"] + names)
        for i in range(len(iter_nums)):
            row = [iter_nums[i], _num(chi2s, i)]
            for nm in names:
                row.append(_num(result.chain.get(nm) or [], i))
            writer.writerow(row)
    return path


def _num(seq, i):
    """A CSV cell for seq[i]: blank for NaN/empty so Excel/pandas read cleanly."""
    if i >= len(seq):
        return ""
    v = seq[i]
    try:
        v = float(v)
    except Exception:
        return ""
    return "" if not np.isfinite(v) else f"{v:.8g}"


def build_chain_figure(result, title: str = "fit parameter chain"):
    """Build (but do not save) the per-parameter trajectory figure: χ² on top,
    then one subplot per free parameter, over the fit's iterations.

    Uses a plain :class:`matplotlib.figure.Figure` (never pyplot), so the same
    figure serves the PNG exporter *and* the in-app Chain preview dialog.
    Returns the (already dark-styled, tight-layouted) figure.
    """
    from matplotlib.figure import Figure

    names = [n for n in result.chain if len(result.chain[n]) >= 2]
    iters = np.asarray(list(result.chain_iter), dtype=float)
    chi2 = np.asarray(list(result.chain_chi2), dtype=float)
    rows = len(names) + 1

    with matplotlib.rc_context(_dark_rc()):
        fig = Figure(figsize=(7.5, max(1.6 * rows, 2.4)))
        axes = []
        first = None
        for i in range(rows):
            ax = fig.add_subplot(rows, 1, i + 1) if first is None \
                else fig.add_subplot(rows, 1, i + 1, sharex=first)
            axes.append(ax)
            if first is None:
                first = ax
        axes[0].plot(iters, chi2, color="#4fc3f7", lw=1.4)
        axes[0].set_ylabel("χ²", fontsize=9)
        axes[0].set_title(title, fontsize=10)
        for ax, nm in zip(axes[1:], names):
            vals = np.asarray([float(v) for v in result.chain[nm]])
            ax.plot(iters, vals, color="#ffb74d", lw=1.2)
            ax.set_ylabel(nm, fontsize=7)
        axes[-1].set_xlabel("iteration", fontsize=8)
        for ax in axes:
            ax.tick_params(labelsize=7)
        fig.tight_layout()
        return fig


def save_chain_png(path: str, result) -> str:
    """Write the per-parameter trajectory figure (χ² on top, then each free
    parameter) over the fit's iterations."""
    fig = build_chain_figure(result)
    with matplotlib.rc_context(_dark_rc()):
        fig.savefig(path, dpi=150)
    return path


def save_all(path_base: str, data, result, save_trajectories=True) -> dict:
    """Convenience: write the report PNG, the chain CSV and (optionally) the
    trajectory PNG from one base path (e.g. ``img/fit_2024``).  Returns a dict
    ``{kind: written_path}`` for the caller to report."""
    out = {}
    out["report"] = save_fit_report_png(f"{path_base}.png", data, result)
    out["chain_csv"] = save_chain_csv(f"{path_base}_chain.csv", result)
    if save_trajectories and result.chain:
        out["chain_png"] = save_chain_png(f"{path_base}_chain_trajectories.png",
                                          result)
    return out
