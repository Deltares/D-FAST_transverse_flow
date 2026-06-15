from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import shapely.plotting
import xarray as xr
import xugrid as xu
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import LineString
from xarray import DataArray

from dfastmi.batch.plotting import chainage_markers, savefig
from dfasttf.config import Config

FIGWIDTH: float = 5.748  # Deltares report width
TEXTFONT = "arial"
TEXTSIZE = 12
CRS: str = "EPSG:28992"  # Netherlands
XMAJORTICK: float = 1000
XMINORTICK: float = 100


# ============================================================
# Generic plot helpers
# ============================================================

def initialize_figure(figwidth: Optional[float] = FIGWIDTH) -> Figure:
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
    if np.issubdtype(np.asarray(x).dtype, np.datetime64):
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        fig.autofmt_xdate()


def format_datetime_yaxis(ax: Axes, y: np.ndarray) -> None:
    if np.issubdtype(np.asarray(y).dtype, np.datetime64):
        locator = mdates.AutoDateLocator()
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def make_rkm_colormap(rkm: np.ndarray):
    rk_km = np.asarray(rkm, dtype=float) / 1000.0
    norm = Normalize(vmin=float(np.nanmin(rk_km)), vmax=float(np.nanmax(rk_km)))
    cmap = get_cmap("viridis")
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


# ============================================================
# Plot configs
# ============================================================

@dataclass
class Plot1DConfig:
    XLABEL: str = "raai km"
    DELTARES_BLUE = "#0D38DF"
    DELTARES_DARKGREEN = "#00B389"
    COLORS = (
        "k",
        DELTARES_BLUE,
        DELTARES_DARKGREEN,
    )  # reference, intervention, difference
    LABELS = ["Referentie", "Plansituatie", "Verschil"]


@dataclass
class FlowfieldConfig:
    VELOCITY_YLABEL: str = "stroomsnelheid\nmagnitude [m/s]"
    VELOCITY_DIFF_YLABEL: str = "verschil plansituatie\n-referentie [m/s]"
    VELOCITY_YLIM: tuple = (0.0, 2.0)
    VELOCITY_YTICKS_MAJOR: float = 0.4
    VELOCITY_YTICKS_MINOR: float = 0.1
    ANGLE_YTICKS_MAJOR: float = 30.0
    ANGLE_YTICKS_MINOR: float = 10.0
    ANGLE_YLIM: tuple = (-90.0, 90.0)
    ANGLE_PRIMARY_YLABEL: str = "stromingshoek t.o.v.\nprofiellijn [graden]"
    ANGLE_DIFF_YLABEL: str = "verschil plansituatie\n-referentie [graden]"
    FRACTION: float = 5.0


@dataclass
class FroudeConfig:
    legend_title = "Froude getal"
    profile_line_color: str = "green"

    class Abs:
        colorbar_label: str = "Froude getal"
        levels: tuple = (0, 0.08, 0.1, 0.15)
        colormap: str = "RdBu"

    class Diff:
        bins: list = [0, 0.08, 0.1, 0.15, np.inf]
        colors = ("blue", "red")
        labels: list[str] = [
            f"van < {bins[1]} naar >= {bins[1]}",
            f"van > {bins[1]} naar <= {bins[1]}",
        ]


@dataclass
class CrossFlowConfig:
    XLABEL = Plot1DConfig.XLABEL
    YLABEL: str = "representatieve dwars-\nstroomsnelheid [m/s]"
    DIFF_YLABEL: str = FlowfieldConfig.VELOCITY_DIFF_YLABEL
    CRIT_LABEL: str = "Lokaal criterium"
    YLIM: tuple = (-0.3, 0.3)
    FRACTION: int = 3
    YTICKS_MAJOR = 0.15
    YTICKS_MINOR = 0.05


# ============================================================
# 2D plotting
# ============================================================

@dataclass
class Plot2D:
    xlabel: str = "x-coördinaat [km]"
    ylabel: str = "y-coördinaat [km]"


    def initialize_map(self) -> tuple[Figure, Axes]:
        fig = initialize_figure(figwidth=FIGWIDTH)
        fig.set_figheight(FIGWIDTH)   # oorspronkelijke meer vierkante verhouding terug
        ax = initialize_subplot(fig, 1, 1, 1, self.xlabel, self.ylabel)
        ax.grid(True)
        return fig, ax


    def modify_axes(self, ax: Axes) -> Axes:
        ax.set_title("")
        ax.set_aspect("equal")
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x / XMAJORTICK:.1f}")
        )
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, _: f"{y / XMAJORTICK:.1f}")
        )
        return ax

    def plot_profile_line(
        self,
        profile: LineString,
        bedlevel: xr.DataArray,
        riverkm: LineString,
        filename: Path,
    ) -> tuple[Figure, Axes]:
        fig, ax = self.initialize_map()
        p = bedlevel.ugrid.plot.pcolormesh(
            ax=ax, add_colorbar=False, cmap="terrain", center=False
        )
        fig.colorbar(
            p,
            ax=ax,
            label="bodemligging [m]",
            orientation="horizontal",
            shrink=0.25,
        )
        shapely.plotting.plot_line(profile, ax=ax, add_points=False, color="black")
        self.modify_axes(ax)
        plot_chainage_markers(riverkm, ax)
        savefig(fig, filename)
        return fig, ax


class Ice2D:
    def create_map(
        self,
        data: DataArray,
        riverkm: LineString,
        profile_line: LineString | None,
        filename: Path,
    ) -> None:
        fig, ax = Plot2D().initialize_map()
        p = data.ugrid.plot(
            ax=ax,
            add_colorbar=False,
            levels=FroudeConfig.Abs.levels,
            cmap=FroudeConfig.Abs.colormap,
            extend="max",
        )
        fig.colorbar(
            p,
            ax=ax,
            label=FroudeConfig.Abs.colorbar_label,
            orientation="horizontal",
            shrink=0.25,
        )
        ax = Plot2D().modify_axes(ax)
        plot_chainage_markers(riverkm, ax)
        if profile_line is not None:
            shapely.plotting.plot_line(
                profile_line,
                ax=ax,
                add_points=False,
                color=FroudeConfig.profile_line_color,
            )
        savefig(fig, filename)

    def create_diff_map(
        self,
        ref_data: xr.DataArray,
        variant_data: xr.DataArray,
        riverkm: LineString,
        profile_line: LineString | None,
        filename: Path,
    ) -> None:
        plt.close("all")
        bins = FroudeConfig.Diff.bins
        colors = FroudeConfig.Diff.colors
        labels = FroudeConfig.Diff.labels

        ref_data_digitized = self._digitize(ref_data.values, bins)
        variant_data_digitized = self._digitize(variant_data.values, bins)

        classes = self._compute_change_classes(ref_data_digitized, variant_data_digitized)
        variant_data.values = classes

        fig, ax = Plot2D().initialize_map()
        color = "lightgrey"
        ref_masked = ref_data[ref_data_digitized == 0]
        ref_masked.ugrid.plot(
            ax=ax,
            cmap=ListedColormap([color]),
            add_colorbar=False,
            vmin=bins[0],
            vmax=bins[1],
        )

        ax, legend_elements = self._plot_diff_map(ax, variant_data, labels, colors)

        ax = Plot2D().modify_axes(ax)
        lgd = fig.legend(
            [Patch(facecolor=color), *legend_elements],
            [f"< {bins[1]} in referentie", *labels],
        )
        lgd.set_title(FroudeConfig.legend_title)
        ax.grid(True)
        plot_chainage_markers(riverkm, ax)
        if profile_line is not None:
            shapely.plotting.plot_line(
                profile_line,
                ax=ax,
                add_points=False,
                color=FroudeConfig.profile_line_color,
            )
        savefig(fig, filename)

    def _plot_diff_map(
        self, ax: Axes, diff_data: xr.DataArray, labels: list[str], colors: tuple
    ) -> tuple[Axes, list]:
        xu.plot.pcolormesh(
            diff_data.grid,
            diff_data,
            ax=ax,
            add_colorbar=False,
            cmap=ListedColormap(colors),
            zorder=1,
        )

        legend_elements = [
            Patch(facecolor=colors[i], label=labels[i]) for i in range(len(labels))
        ]
        return ax, legend_elements

    def _digitize(self, data: Any, bins: Any) -> np.ndarray:
        return np.digitize(data, bins) - 1

    def _compute_change_classes(
        self, ref_data: np.ndarray, variant_data: np.ndarray
    ) -> np.ndarray:
        classes = variant_data * np.nan
        conditions = [
            (ref_data < 1) & (variant_data >= 1),
            (ref_data > 0) & (variant_data <= 0),
        ]
        for i, cond in enumerate(conditions, start=1):
            classes[cond] = i
        return classes


# ============================================================
# 1D ice plotting
# ============================================================

class Ice1D:
    """Class for plotting 1D river flow velocity and angle."""

    def plot_velocity_magnitude(
        self, ax: Axes, distance: np.ndarray, velocity: np.ndarray, color: str
    ) -> Axes:
        plot_variable(ax, distance, velocity, color)
        ax.set_ylim(FlowfieldConfig.VELOCITY_YLIM)
        ax.yaxis.set_major_locator(
            ticker.MultipleLocator(FlowfieldConfig.VELOCITY_YTICKS_MAJOR)
        )
        ax.yaxis.set_minor_locator(
            ticker.MultipleLocator(FlowfieldConfig.VELOCITY_YTICKS_MINOR)
        )
        return ax

    def plot_velocity_angle(
        self, ax: Axes, distance: np.ndarray, angle: np.ndarray, color: str
    ) -> Axes:
        plot_variable(ax, distance, angle, color)
        ax.set_ylim(FlowfieldConfig.ANGLE_YLIM)
        ax.yaxis.set_major_locator(
            ticker.MultipleLocator(FlowfieldConfig.ANGLE_YTICKS_MAJOR)
        )
        ax.yaxis.set_minor_locator(
            ticker.MultipleLocator(FlowfieldConfig.ANGLE_YTICKS_MINOR)
        )
        return ax

    def create_figure(
        self,
        distance: np.ndarray,
        velocity: list,
        angle: list,
        configuration: Config,
        filename: Path,
    ) -> None:
        plt.close("all")
        fig = initialize_figure()
        config = Plot1DConfig()

        ax1 = initialize_subplot(
            fig, 2, 1, 1, config.XLABEL, FlowfieldConfig.VELOCITY_YLABEL
        )
        ax2 = initialize_subplot(
            fig, 2, 1, 2, config.XLABEL, FlowfieldConfig.ANGLE_PRIMARY_YLABEL
        )

        for i, (v, a) in enumerate(zip(velocity, angle)):
            self.plot_velocity_magnitude(ax1, distance, v, Plot1DConfig.COLORS[i])
            self.plot_velocity_angle(ax2, distance, a, Plot1DConfig.COLORS[i])

        fraction = FlowfieldConfig.FRACTION
        axs_diff = []
        if len(velocity) > 1:
            for ax, data, ylabel in [
                (ax1, velocity[1] - velocity[0], FlowfieldConfig.VELOCITY_DIFF_YLABEL),
                (ax2, angle[1] - angle[0], FlowfieldConfig.ANGLE_DIFF_YLABEL),
            ]:
                ax_diff = difference_plot(ax, ylabel, Plot1DConfig.COLORS[-1])
                plot_variable(ax_diff, distance, data, Plot1DConfig.COLORS[-1])
                ax_diff.set_ylim(-ax.get_ylim()[1] / fraction, ax.get_ylim()[1] / fraction)
                axs_diff.append(ax_diff)

            axs_diff[0].yaxis.set_major_locator(
                ticker.MultipleLocator(FlowfieldConfig.VELOCITY_YTICKS_MAJOR / fraction)
            )
            axs_diff[1].yaxis.set_major_locator(
                ticker.MultipleLocator(FlowfieldConfig.ANGLE_YTICKS_MAJOR / fraction)
            )

        for ax in [ax1, ax2]:
            style_1d_axis(ax, configuration.general.bool_flags["invertxaxis"])

        ax1.legend(
            Plot1DConfig.LABELS,
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            ncols=2,
            borderaxespad=0.0,
        )
        savefig(fig, filename)


# ============================================================
# Cross-flow plotting
# ============================================================

class CrossFlow:
    def __init__(self, config: CrossFlowConfig = CrossFlowConfig()):
        self.config = config

    def plot_discharge(
        self,
        ax: Axes,
        xy_segments: list[list[tuple]],
        crit_values: list[np.ndarray],
    ) -> Optional[LineCollection]:
        crit_handle = None
        xy_segments = xy_segments[-1]
        crit_values = crit_values[-1]

        for (xi, yi), crit_value in zip(xy_segments, crit_values):
            ax.fill_between(xi, yi, color="lightgrey", interpolate=True)
            crit_handle = ax.hlines(crit_value, xi[0], xi[-1], color="red", lw=1, ls="-")
            ax.hlines(-crit_value, xi[0], xi[-1], color="red", lw=1, ls="-")

        return crit_handle

    def create_figure(
        self,
        distance: np.ndarray,
        transverse_velocity: list[np.ndarray],
        xy_segments: list[list],
        crit_values: list[np.ndarray],
        inverse_xaxis: bool,
        filename: Path,
    ) -> None:
        plt.close("all")
        fig = initialize_figure()
        ax1 = initialize_subplot(fig, 1, 1, 1, self.config.XLABEL, self.config.YLABEL)
        ax1.set_ylim(self.config.YLIM)

        crit_handle = self.plot_discharge(ax1, xy_segments, crit_values)

        lines = []
        for i, v in enumerate(transverse_velocity):
            (line,) = plot_variable(ax1, distance, v, Plot1DConfig.COLORS[i])
            lines.append(line)

        fraction = self.config.FRACTION
        ax2 = None
        diff = None
        if len(transverse_velocity) > 1:
            ax2 = difference_plot(ax1, CrossFlowConfig.DIFF_YLABEL, Plot1DConfig.COLORS[-1])
            data = transverse_velocity[1] - transverse_velocity[0]
            ax2.set_ylim([y / fraction for y in ax1.get_ylim()])
            (diff,) = plot_variable(ax2, distance, data, color=Plot1DConfig.COLORS[-1])

        style_1d_axis(ax1, inverse_xaxis)

        handles = [*lines]
        labels = [*Plot1DConfig.LABELS[0:len(transverse_velocity)]]

        if crit_handle is not None:
            handles.append(crit_handle)
            labels.append(CrossFlowConfig.CRIT_LABEL)

        ax1.yaxis.set_major_locator(ticker.MultipleLocator(CrossFlowConfig.YTICKS_MAJOR))
        ax1.yaxis.set_minor_locator(ticker.MultipleLocator(CrossFlowConfig.YTICKS_MINOR))

        if ax2 is not None and diff is not None:
            ax2.yaxis.set_major_locator(
                ticker.MultipleLocator(CrossFlowConfig.YTICKS_MAJOR / fraction)
            )
            ax2.yaxis.set_minor_locator(
                ticker.MultipleLocator(CrossFlowConfig.YTICKS_MINOR / fraction)
            )
            handles.append(diff)
            labels.append(Plot1DConfig.LABELS[-1])

        ax1.legend(
            handles,
            labels,
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            ncols=2,
            borderaxespad=0.0,
        )
        fig.set_figheight(0.5 * FIGWIDTH)
        savefig(fig, filename)

    def create_figure_tide_velocities(
        self,
        distance: np.ndarray,
        transverse_velocity_ebb: list[np.ndarray | None],
        transverse_velocity_flood: list[np.ndarray | None],
        inverse_xaxis: bool,
        filename: Path,
        threshold: float | None = None,
        annotation: str | None = None,
    ) -> None:
        plt.close("all")
        fig = initialize_figure(figwidth=1.35 * FIGWIDTH)

        ax1 = initialize_subplot(fig, 2, 1, 1, self.config.XLABEL, self.config.YLABEL)
        ax2 = initialize_subplot(fig, 2, 1, 2, self.config.XLABEL, self.config.YLABEL)

        ax1.set_title("Dwarsstroomsnelheid bij piek eb (per locatie)")
        ax2.set_title("Dwarsstroomsnelheid bij piek vloed (per locatie)")

        ax1.set_ylim(self.config.YLIM)
        ax2.set_ylim(self.config.YLIM)

        ebb_lines = []
        for i, v in enumerate(transverse_velocity_ebb):
            if v is None:
                continue
            (line,) = plot_variable(ax1, distance, v, Plot1DConfig.COLORS[i])
            ebb_lines.append(line)

        for i, v in enumerate(transverse_velocity_flood):
            if v is None:
                continue
            plot_variable(ax2, distance, v, Plot1DConfig.COLORS[i])

        for ax in (ax1, ax2):
            style_1d_axis(ax, inverse_xaxis)
            ax.yaxis.set_major_locator(
                ticker.MultipleLocator(CrossFlowConfig.YTICKS_MAJOR)
            )
            ax.yaxis.set_minor_locator(
                ticker.MultipleLocator(CrossFlowConfig.YTICKS_MINOR)
            )

        n_present = sum(v is not None for v in transverse_velocity_ebb)
        labels = Plot1DConfig.LABELS[0:n_present]
        handles = ebb_lines[:n_present]
        ax1.legend(
            handles,
            labels,
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            ncols=2,
            borderaxespad=0.0,
        )

        if annotation:
            fig.text(
                0.01,
                0.01,
                annotation,
                ha="left",
                va="bottom",
                fontsize=9,
                color="dimgray",
            )

        fig.set_figheight(0.9 * FIGWIDTH)
        savefig(fig, filename)

    def create_figure_alongstream_timeseries(
        self,
        time: np.ndarray | None,
        upar_tn: np.ndarray,
        rkm: np.ndarray,
        filename: Path,
        threshold: float | None = None,
        annotation: str | None = None,
        idx_ebb: np.ndarray | None = None,
        idx_flood: np.ndarray | None = None,
    ) -> None:
        plt.close("all")

        fig = initialize_figure(figwidth=1.35 * FIGWIDTH)
        gs = fig.add_gridspec(
            nrows=2,
            ncols=2,
            width_ratios=[20, 1.2],
            height_ratios=[10, 2],
        )

        ax = fig.add_subplot(gs[:, 0])
        cax = fig.add_subplot(gs[0, 1])
        lax = fig.add_subplot(gs[1, 1])

        if time is None or np.asarray(time).size == 0:
            x = np.arange(upar_tn.shape[0])
        else:
            x = np.asarray(time)

        rk_km, norm, cmap = make_rkm_colormap(rkm)

        nt, n = upar_tn.shape
        for i in range(n):
            ax.plot(
                x,
                upar_tn[:, i],
                color=cmap(norm(rk_km[i])),
                lw=0.4,
                alpha=0.35,
            )

        (mean_line,) = ax.plot(
            x,
            np.nanmean(upar_tn, axis=1),
            color="black",
            lw=1.2,
            alpha=0.9,
            label="gemiddelde",
        )

        thr_handle = None
        if threshold is not None and np.isfinite(threshold) and threshold > 0:
            thr_handle = ax.axhline(
                +threshold,
                color="red",
                lw=1.0,
                ls="--",
                label=f"drempel ±{threshold:.2f} m/s",
            )
            ax.axhline(-threshold, color="red", lw=1.0, ls="--")

        format_datetime_xaxis(fig, ax, x)

        ax.set_xlabel("tijd")
        ax.set_ylabel("u [m/s]")
        ax.grid(True, which="major", linestyle="-", alpha=0.6)
        ax.grid(True, which="minor", linestyle="--", color="lightgrey", alpha=0.6)

        proxies = []
        legend_labels = []

        if idx_ebb is not None:
            scatter_idx_points_on_timeseries(
                ax,
                x,
                upar_tn,
                idx_ebb,
                rk_km,
                norm,
                cmap,
                marker="o",
                edgecolor="red",
                zorder=6,
            )
            proxies.append(make_marker_proxy(marker="o", edgecolor="red", markeredgewidth=0.35))
            legend_labels.append("max ebb (per rkm)")

        if idx_flood is not None:
            scatter_idx_points_on_timeseries(
                ax,
                x,
                upar_tn,
                idx_flood,
                rk_km,
                norm,
                cmap,
                marker="^",
                edgecolor="red",
                zorder=7,
            )
            proxies.append(make_marker_proxy(marker="^", edgecolor="red", markeredgewidth=0.35))
            legend_labels.append("max flood (per rkm)")

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
        cbar.set_label("rkm [km]")

        leg_handles = [mean_line]
        leg_labels = ["gemiddelde"]
        if thr_handle is not None:
            leg_handles.append(thr_handle)
            leg_labels.append(f"drempel ±{threshold:.2f} m/s")
        leg_handles.extend(proxies)
        leg_labels.extend(legend_labels)

        add_side_legend(lax, leg_handles, leg_labels)

        if annotation:
            fig.text(
                0.01,
                0.01,
                annotation,
                ha="left",
                va="bottom",
                fontsize=9,
                color="dimgray",
            )

        fig.set_figheight(0.65 * FIGWIDTH)
        savefig(fig, filename)

    def create_figure_tide_max_transverse(
        self,
        rkm: np.ndarray,
        tv_max_list: list[np.ndarray | None],
        time: np.ndarray | None,
        upar_tn: np.ndarray,
        idx_tvmax: np.ndarray,
        tv_tn: np.ndarray,
        filename: Path,
        inverse_xaxis: bool,
    ) -> None:
        """
        Figure with:
        1) maximum transverse velocity over the full tide cycle vs rkm
        2) alongstream velocity through time with points at idx_tvmax
        3) x-t heatmap of |tv_tn| with the same points overlaid
        """
        plt.close("all")
        fig = initialize_figure(figwidth=1.35 * FIGWIDTH)

        gs = fig.add_gridspec(
            nrows=4,
            ncols=2,
            width_ratios=[20, 1.4],
            height_ratios=[7, 10, 8, 2],
        )

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[2, 0])

        cax1 = fig.add_subplot(gs[1, 1])  # colorbar subplot 2
        cax2 = fig.add_subplot(gs[2, 1])  # colorbar subplot 3
        lax = fig.add_subplot(gs[3, 1])

        # ------------------------------------------------------------
        # Subplot 1: max transverse velocity vs rkm
        # ------------------------------------------------------------
        lines1 = []
        for i, y in enumerate(tv_max_list):
            if y is None:
                continue
            (line,) = plot_variable(ax1, rkm, y, Plot1DConfig.COLORS[i])
            lines1.append(line)

        ax1.set_title("Maximale dwarsstroomsnelheid over getijcyclus")
        ax1.set_ylabel("dwarsstroomsnelheid [m/s]")
        style_1d_axis(ax1, inverse_xaxis)

        n_present = sum(y is not None for y in tv_max_list)
        labels = Plot1DConfig.LABELS[0:n_present]
        handles = lines1[:n_present]

        # ------------------------------------------------------------
        # Subplot 2: alongstream velocity through time + idx_tvmax points
        # ------------------------------------------------------------
        if time is None or np.asarray(time).size == 0:
            x = np.arange(upar_tn.shape[0])
        else:
            x = np.asarray(time)

        rk_km, norm, cmap = make_rkm_colormap(rkm)

        nt, n = upar_tn.shape
        for i in range(n):
            ax2.plot(
                x,
                upar_tn[:, i],
                color=cmap(norm(rk_km[i])),
                lw=0.4,
                alpha=0.35,
            )

        (mean_line,) = ax2.plot(
            x,
            np.nanmean(upar_tn, axis=1),
            color="black",
            lw=1.2,
            alpha=0.9,
            label="gemiddelde",
        )

        scatter_idx_points_on_timeseries(
            ax2,
            x,
            upar_tn,
            idx_tvmax,
            rk_km,
            norm,
            cmap,
            marker="s",
            edgecolor="magenta",
            zorder=8,
            size=18,
            linewidth=0.35,
        )

        format_datetime_xaxis(fig, ax2, x)

        ax2.set_title("Moment van maximale dwarsstroomsnelheid")
        ax2.set_xlabel("tijd")
        ax2.set_ylabel("langsstroomsnelheid [m/s]")
        ax2.grid(True, which="major", linestyle="-", alpha=0.6)
        ax2.grid(True, which="minor", linestyle="--", color="lightgrey", alpha=0.6)

        sm_rkm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm_rkm.set_array([])
        cbar1 = fig.colorbar(sm_rkm, cax=cax1, orientation="vertical")
        cbar1.set_label("rkm [km]")

        # ------------------------------------------------------------
        # Subplot 3: x-t heatmap of |tv_tn| + idx_tvmax points
        # ------------------------------------------------------------
        if time is None or np.asarray(time).size == 0:
            y_time = np.arange(tv_tn.shape[0])
        else:
            y_time = np.asarray(time)

        x_rkm = np.asarray(rkm, dtype=float)

        pcm = ax3.pcolormesh(
            x_rkm,
            y_time,
            np.abs(tv_tn),
            shading="auto",
            cmap="viridis",
        )

        idx_tvmax = np.asarray(idx_tvmax, dtype=int)
        m = idx_tvmax >= 0
        if np.any(m):
            x_pts = np.asarray(x_rkm[m], dtype=float).ravel()
            y_pts = np.asarray(y_time[idx_tvmax[m]]).ravel()

            ax3.scatter(
                x_pts,
                y_pts,
                s=18,
                marker="s",
                facecolors="white",
                edgecolors="magenta",
                linewidths=0.5,
                zorder=8,
            )

        ax3.set_title("Absolute dwarsstroomsnelheid in tijd")
        ax3.set_xlabel("rivierkilometer")
        ax3.set_ylabel("tijd")
        style_1d_axis(ax3, inverse_xaxis)
        format_datetime_yaxis(ax3, y_time)

        cbar2 = fig.colorbar(pcm, cax=cax2, orientation="vertical")
        cbar2.set_label("|dwarsstroomsnelheid| [m/s]")

        # ------------------------------------------------------------
        # Shared legend
        # ------------------------------------------------------------
        proxy_max_tv = make_marker_proxy(
            marker="s",
            edgecolor="magenta",
            facecolor="white",
            markersize=6,
            markeredgewidth=0.5,
        )

        legend_handles = [*handles, mean_line, proxy_max_tv]
        legend_labels = [*labels, "gemiddelde", "moment max dwarsstroming"]

        add_side_legend(lax, legend_handles, legend_labels)

        fig.set_figheight(1.0 * FIGWIDTH)
        savefig(fig, filename)