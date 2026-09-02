"""Generic matplotlib helpers shared across the dfasttf plotting submodules.

This module holds figure/axis setup helpers, styling utilities and small
reusable building blocks (colormaps, marker proxies, legends) that are used
by the domain-specific plotting modules (maps2d, ice1d, cross_flow).
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from shapely.geometry import LineString

from dfastmi.batch.plotting import chainage_markers

FIGWIDTH: float = 5.748  # Deltares report width
TEXTFONT = "arial"
TEXTSIZE = 12
CRS: str = "EPSG:28992"  # Netherlands
XMAJORTICK: float = 1000
XMINORTICK: float = 100


# ============================================================
# Generic plot helpers
# ============================================================

def initialize_figure(figwidth: float | None = FIGWIDTH) -> Figure:
    font = {"family": TEXTFONT, "size": TEXTSIZE}
    plt.rc("font", **font)
    fig = plt.figure(layout="constrained")
    fig.set_figwidth(figwidth)
    return fig


def initialize_subplot(
    fig: Figure, nrows: int, ncols: int, index: int, xlabel: str, ylabel: str
) -> Axes:
    ax = fig.add_subplot(nrows, ncols, index)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def difference_plot(ax: Axes, ylabel: str, color: str) -> Axes:
    secax_y2 = ax.twinx()
    secax_y2.set_ylabel(ylabel)
    secax_y2.yaxis.label.set_color(color)
    secax_y2.tick_params(color=color, labelcolor=color, which="both")
    secax_y2.spines["right"].set_color(color)
    return secax_y2


def invert_xaxis(ax: Axes) -> None:
    ax.xaxis.set_inverted(True)


def plot_variable(
    ax: Axes, x: np.ndarray, y: np.ndarray, color: str = "black"
) -> list[Line2D]:
    return ax.plot(x, y, "-", linewidth=0.5, color=color)


def plot_chainage_markers(riverkm: LineString, ax: Axes) -> None:
    filtered_coords = np.array([coord for coord in riverkm.coords if coord[2] % 1 == 0])
    chainage_markers(filtered_coords, ax, scale=1, ndec=0)


def modify_axes(ax: Axes, x_major_tick: float) -> Axes:
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / x_major_tick}"))
    ax.tick_params(which="major", length=8)
    ax.tick_params(which="minor", length=4)
    return ax


def construct_figure_filename(figdir: Path, base: str, extension: str) -> Path:
    return Path(figdir) / f"{base}{extension}"


def style_1d_axis(ax: Axes, inverse_xaxis_flag: bool = False) -> None:
    modify_axes(ax, XMAJORTICK)
    if inverse_xaxis_flag:
        invert_xaxis(ax)
    ax.grid(visible=True, which="major", linestyle="-")
    ax.grid(
        visible=True,
        which="minor",
        axis="y",
        linestyle="--",
        color="lightgrey",
    )


def format_datetime_xaxis(fig: Figure, ax: Axes, x: np.ndarray) -> None:
    """Format a datetime64 x-axis with a concise date locator/formatter.

    Rotates and right-aligns only this axis's own tick labels. We deliberately
    avoid ``fig.autofmt_xdate()`` here: it hides the tick labels (and clears
    the xlabel) of every *other* axes in the figure that isn't in the last
    gridspec row, which silently breaks unrelated subplots in multi-panel
    figures that mix a date axis with other 1D/2D panels.
    """
    if np.issubdtype(np.asarray(x).dtype, np.datetime64):
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")


def format_datetime_yaxis(ax: Axes, y: np.ndarray) -> None:
    if np.issubdtype(np.asarray(y).dtype, np.datetime64):
        locator = mdates.AutoDateLocator()
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def make_rkm_colormap(rkm: np.ndarray):
    rk_km = np.asarray(rkm, dtype=float) / 1000.0
    norm = Normalize(vmin=float(np.nanmin(rk_km)), vmax=float(np.nanmax(rk_km)))
    cmap = matplotlib.colormaps["viridis"]
    return rk_km, norm, cmap


def make_marker_proxy(
    marker: str,
    edgecolor: str = "k",
    facecolor: str = "white",
    markersize: int = 6,
    markeredgewidth: float = 0.5,
) -> Line2D:
    return Line2D(
        [0],
        [0],
        marker=marker,
        linestyle="None",
        markerfacecolor=facecolor,
        markeredgecolor=edgecolor,
        markeredgewidth=markeredgewidth,
        color="k",
        markersize=markersize,
    )


def add_side_legend(
    lax: Axes,
    handles: list,
    labels: list[str],
    fontsize: int = 8,
) -> None:
    lax.axis("off")
    lax.legend(
        handles,
        labels,
        loc="upper left",
        frameon=False,
        fontsize=fontsize,
        handlelength=1.2,
        handletextpad=0.4,
        labelspacing=0.4,
        borderaxespad=0.0,
    )


def scatter_idx_points_on_timeseries(
    ax: Axes,
    x: np.ndarray,
    y_tn: np.ndarray,
    idx_t: np.ndarray | None,
    rkm: np.ndarray,
    norm: Normalize,
    cmap,
    marker: str,
    edgecolor: str,
    zorder: int,
    size: int = 18,
    linewidth: float = 0.35,
) -> None:
    if idx_t is None:
        return

    idx_t = np.asarray(idx_t, dtype=int)
    i_all = np.arange(y_tn.shape[1])
    m = idx_t >= 0
    if not np.any(m):
        return

    ii = i_all[m]
    tt = idx_t[m]
    x_pts = np.asarray(x[tt]).ravel()
    y_pts = np.asarray(y_tn[tt, ii], dtype=float).ravel()
    c_pts = cmap(norm(rkm[ii]))

    ax.scatter(
        x_pts,
        y_pts,
        s=size,
        marker=marker,
        c=c_pts,
        alpha=0.9,
        edgecolors=edgecolor,
        linewidths=linewidth,
        zorder=zorder,
    )

def savefig(fig: matplotlib.figure.Figure, filename: str) -> None:
    """
    Save a single figure to file.

    Arguments
    ---------
    fig : matplotlib.figure.Figure
        Figure to a be saved.
    filename : str
        Name of the file to be written.
    """
    print("saving figure {file}".format(file=filename))
    fig.savefig(filename, dpi=300, bbox_inches='tight', pad_inches='layout')
