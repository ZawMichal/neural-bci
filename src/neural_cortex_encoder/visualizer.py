"""Neuralink-inspired static waterfall rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_waterfall(signal: np.ndarray, path: str | Path, *, title: str = "") -> None:
    """Save a black-to-blue-to-yellow heatmap without imposing plotting dependencies."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as exc:
        raise RuntimeError("visualization requires matplotlib; install it separately") from exc

    if signal.ndim != 2:
        raise ValueError("signal must have shape (time, channels)")
    palette = LinearSegmentedColormap.from_list(
        "neural_activity",
        ["#000000", "#07134f", "#123f9b", "#00a8d6", "#f2d35e", "#fff4c2"],
    )
    figure, axis = plt.subplots(figsize=(14, 7), facecolor="black")
    axis.imshow(signal.T, aspect="auto", origin="lower", cmap=palette, vmin=0.0, vmax=1.0)
    axis.set_facecolor("black")
    axis.set_xlabel("Time (frames)", color="white")
    axis.set_ylabel("Channel", color="white")
    axis.tick_params(colors="white")
    axis.set_title(title, color="white")
    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)
