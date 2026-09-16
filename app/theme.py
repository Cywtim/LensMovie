"""Central visual theme for LensMovie.

Everything that decides how the app *looks* lives here (plus ``theme.qss``):

  * :data:`PALETTE`   — the one source of truth for colours,
  * :func:`qss`       — the full Qt stylesheet with palette tokens injected,
  * :func:`apply_theme` — sets the stylesheet + application font, and switches
    matplotlib to a matching dark look,
  * :data:`MARKER_COLORS`, :data:`CANVAS_STYLE` — used by the plotting canvases.

The program core (``lensing_calc`` / ``fitting``) never imports this module —
it is purely a presentation concern.  Restyling the whole app is a one-file
change (``theme.qss`` / ``PALETTE``) and nothing in the physics needs to move.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------- palette
# The single source of truth for colours.  ``theme.qss`` references these via
# ``@name@`` tokens so editing one dict restyles every widget, canvas and chart.
PALETTE = {
    "app_bg":       "#14171c",   # window background
    "panel":        "#1c222b",   # group boxes / panels / card surfaces
    "card":         "#243040",   # buttons, hover surfaces
    "input":        "#0f1318",   # combo/spin/slider-track background
    "selection":    "#123a4d",   # highlighted rows
    "border":       "#2b3542",   # quiet borders
    "border_hi":    "#3a4a5e",   # brighter borders / scrollbars
    "text":         "#d7dde4",   # primary text
    "text_dim":     "#8b97a5",   # secondary text / axes labels
    "text_faint":   "#5c6672",   # disabled text / hints
    "accent":       "#4fc3f7",   # interactive accent (focus, hover, sliders)
    "accent_dim":   "#123d52",   # checked / selected accent wash
    "accent_text":  "#8fd3ff",   # titles on accent
    "slider_fill":  "#3a6ea5",   # filled slider groove
    "slider_thumb": "#8fd3ff",   # slider handle
    "tooltip_bg":   "#1c222b",   # tooltip background
}

# Marker colours for source image positions, chosen to stay visible on the
# dark canvas.  Used by ``app/plotting``.
MARKER_COLORS = [
    "#4fc3f7", "#ffb74d", "#f06292", "#a5d6a7",
    "#ba68c8", "#e57373", "#4dd0e1", "#dce775",
]

# Qt widget chrome that the matplotlib 2D panels reuse so that the canvases
# blend with the surrounding widgets instead of showing a white figure.
CANVAS_STYLE = {
    "figure_face": PALETTE["panel"],     # thin frame behind the axes
    "axes_face":   PALETTE["input"],     # the plot area
    "spine":       PALETTE["border_hi"],
    "text":        PALETTE["text"],
    "dim":         PALETTE["text_dim"],
    "grid":        PALETTE["border"],
    "legend_face": PALETTE["panel"],
    "legend_edge": PALETTE["border_hi"],
}

# Preferred UI font; the first candidate that exists is used by Qt.
FONT_HINTS = ["Segoe UI", "Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"]

_QSS_PATH = Path(__file__).with_name("theme.qss")


def qss() -> str:
    """The full Qt stylesheet, with ``@name@`` palette tokens substituted.

    Any token left unresolved is an error — the palette is the single place
    colours are defined, so forgetting one should fail loudly.
    """
    text = _QSS_PATH.read_text(encoding="utf-8")
    for name, value in PALETTE.items():
        text = text.replace(f"@{name}@", str(value))
    leftover = _unresolved_tokens(text)
    if leftover:
        raise ValueError(
            f"theme.qss references palette tokens missing from PALETTE: "
            f"{sorted(leftover)}"
        )
    return text


def _unresolved_tokens(text: str) -> set:
    import re

    return set(re.findall(r"@([a-z_0-9]+)@", text))


def apply_matplotlib_dark() -> None:
    """Switch the global matplotlib defaults to the dark scientific look.

    Canvas-level figures pick their colours from :data:`CANVAS_STYLE` directly
    (so a single window is deterministic), while this makes any *other* figure
    (e.g. debug plots) match too.
    """
    import matplotlib

    rc = matplotlib.rcParams
    s = CANVAS_STYLE
    rc["figure.facecolor"] = s["figure_face"]
    rc["axes.facecolor"] = s["axes_face"]
    rc["axes.edgecolor"] = s["spine"]
    rc["axes.labelcolor"] = s["dim"]
    rc["xtick.color"] = s["dim"]
    rc["ytick.color"] = s["dim"]
    rc["text.color"] = s["text"]
    rc["grid.color"] = s["grid"]
    rc["legend.facecolor"] = s["legend_face"]
    rc["legend.edgecolor"] = s["legend_edge"]
    rc["legend.labelcolor"] = s["text"]
    rc["savefig.facecolor"] = s["figure_face"]


def apply_theme(qapp) -> None:
    """Apply the theme to a running ``QApplication``.  Idempotent.

    This is the one entry point the rest of the app calls; it styles every Qt
    widget *and* switches matplotlib to the matching dark look.
    """
    qapp.setStyleSheet(qss())
    font = qapp.font()
    for family in FONT_HINTS:
        font.setFamily(family)
        break  # Qt falls back gracefully if the family is missing anyway
    font.setPointSize(9)
    qapp.setFont(font)
    apply_matplotlib_dark()
