"""One visual system for every figure: validated categorical palette (fixed slot order),
single-hue sequential ramp, hairline chrome, no dual axes."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gmcn.config import FIGURES  # noqa: E402

# Categorical slots in fixed order (CVD-validated adjacent pairs); never cycled past 8.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
       "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CRITICAL = "#d03b3b"


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "lines.linewidth": 2.0, "legend.frameon": False, "legend.fontsize": 9,
    })


def subtitle(ax, text: str) -> None:
    """Secondary line under the left-aligned title."""
    ax.text(0, 1.02, text, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


def source_note(fig, text: str) -> None:
    """Attach a source line; `save` places it below every other artist so it never collides."""
    fig._gmcn_note = text


def save(fig, name: str) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    note = getattr(fig, "_gmcn_note", None)
    if note:
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())  # inches
        w, h = fig.get_size_inches()
        fig.text(bb.x0 / w, (bb.y0 - 0.08) / h, note, transform=fig.transFigure, fontsize=7.5,
                 color=MUTED, ha="left", va="top")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)
