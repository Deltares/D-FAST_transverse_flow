from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import dask
import matplotlib.pyplot as plt
import numpy as np
import shapely.plotting
import xarray as xr
import xugrid as xu
from geopandas import GeoDataFrame
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely import LineString
from xarray import DataArray

from dfastmi.batch.plotting import chainage_markers, savefig

# from dfastmi.batch.PlotOptions import PlotOptions
from dfastrbk.src.config import Config

# import contextily as ctx
# from xyzservices import TileProvider

FIGWIDTH: float = 5.748  # Deltares report width
TEXTFONT = "arial"
TEXTSIZE = 12
CRS: str = "EPSG:28992"  # Netherlands
XMAJORTICK: float = 1000
XMINORTICK: float = 100


def initialize_figure(figwidth: Optional[float] = FIGWIDTH) -> Figure:
    font = {"family": TEXTFONT, "size": TEXTSIZE}
    plt.rc("font", **font)
    fig = plt.figure(layout="constrained")
    # fig.set_figwidth(figwidth)
    return fig


def initialize_subplot(
    fig: Figure, nrows: int, ncols: int, index: int, xlabel: str, ylabel: str
):
    ax = fig.add_subplot(nrows, ncols, index)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def difference_plot(ax: Axes, ylabel: str, color: str):
    secax_y2 = ax.twinx()
    secax_y2.set_ylabel(ylabel)
    secax_y2.yaxis.label.set_color(color)
    secax_y2.tick_params(color=color, labelcolor=color)
    secax_y2.spines["right"].set_color(color)
    return secax_y2


def invert_xaxis(ax: Axes):
    ax.xaxis.set_inverted(True)


def plot_variable(
    ax: Axes, x: np.ndarray, y: np.ndarray, color: str = "black"
) -> list[Line2D]:
    p = ax.plot(x, y, "-", linewidth=0.5, color=color)
    return p


def plot_chainage_markers(riverkm: LineString, ax: Axes):
    # first filter chainage by 1000 m
    filtered_coords = np.array([coord for coord in riverkm.coords if coord[2] % 1 == 0])
    chainage_markers(filtered_coords, ax, scale=1, ndec=0)


def align_twinx_grid_centered(
    primary,
    secondary,
    *,
    center=0.0,
    keep_symmetric=True,
    add_centerline=True,
    label_formatter=None,
    _eps=1e-12,
):
    """
    Align secondary-axis (right) ticks to the primary (left) axis horizontal gridlines,
    and (optionally) keep the secondary axis symmetric around `center` (default: 0).

    Parameters
    ----------
    primary : matplotlib.axes.Axes
        Left axis.
    secondary : matplotlib.axes.Axes
        Right axis, created with twinx().
    center : float
        Value the secondary axis should be centered on (default: 0).
    keep_symmetric : bool
        If True, force secondary ylim to be [center - M, center + M], where
        M = max(|ylow - center|, |yhigh - center|).
    add_centerline : bool
        If True, draws/updates a dashed horizontal line at secondary==center
        (mapped into primary coordinates) so the midline is visible even if
        the primary has no tick there.
    label_formatter : callable or None
        Optional function to format secondary tick labels. Receives float -> str.
    _eps : float
        Internal epsilon to avoid divide-by-zero loops.
    """
    state = {"centerline": None, "in_update": False}

    def update(_evt):
        if state["in_update"]:
            return
        state["in_update"] = True
        try:
            # Current limits
            y1_lo, y1_hi = primary.get_ylim()
            y2_lo, y2_hi = secondary.get_ylim()

            # Ensure secondary is symmetric about `center`
            if keep_symmetric:
                span_lo = abs(y2_lo - center)
                span_hi = abs(y2_hi - center)
                M = max(span_lo, span_hi, _eps)
                new_lo, new_hi = center - M, center + M
                # Only set if it actually changes to avoid endless callbacks
                if (abs(new_lo - y2_lo) > _eps) or (abs(new_hi - y2_hi) > _eps):
                    secondary.set_ylim(new_lo, new_hi)
                    y2_lo, y2_hi = new_lo, new_hi

            # Protect against zero primary span
            y1_span = y1_hi - y1_lo
            if abs(y1_span) < _eps:
                return

            # Linear mapping primary -> secondary: y2 = a*y1 + b
            a = (y2_hi - y2_lo) / y1_span
            b = y2_lo - a * y1_lo

            # Align secondary ticks to primary gridlines
            yt1 = primary.get_yticks()
            yt2 = a * yt1 + b
            secondary.set_yticks(yt2)
            if label_formatter is not None:
                secondary.set_yticklabels([label_formatter(val) for val in yt2])

            # Grid: only on primary (so lines are shared across both)
            primary.set_axisbelow(True)
            primary.grid(True, axis="y")
            secondary.grid(False)

            # Optional: visible centerline at secondary==center
            if add_centerline and abs(a) > _eps:
                y1_at_center = (center - b) / a
                if state["centerline"] is None:
                    # One line that we update on every callback
                    state["centerline"] = primary.axhline(
                        y1_at_center, color="black", ls="--", lw=1
                    )
                else:
                    state["centerline"].set_ydata([y1_at_center, y1_at_center])
                # Hide line if center is outside current primary limits (e.g., zoom)
                vis = (
                    min(y1_lo, y1_hi) - _eps <= y1_at_center <= max(y1_lo, y1_hi) + _eps
                )
                state["centerline"].set_visible(vis)

            primary.figure.canvas.draw_idle()
        finally:
            state["in_update"] = False

    # Recompute whenever y-lims change (zoom/pan/autoscale)
    primary.callbacks.connect("ylim_changed", update)
    secondary.callbacks.connect("ylim_changed", update)
    update(None)


# def add_satellite_image(ax: Axes, background_image: TileProvider):
#     ctx.add_basemap(ax=ax, source=background_image, crs=CRS, attribution=False, zorder=-1)


@dataclass
class Plot1DConfig:
    XLABEL: str = "afstand [rivierkilometer]"
    COLORS = ("k", "b", "r")  # reference, intervention, difference
    LABELS = ["Referentie", "Plansituatie"]


@dataclass
class Plot2D:
    xlabel: str = "x-coördinaat [km]"
    ylabel: str = "y-coördinaat [km]"
    # background_image = ctx.providers.OpenStreetMap.Mapnik #ctx.providers.Esri.WorldImagery

    def initialize_map(self) -> tuple[Figure, Axes]:
        fig = initialize_figure()
        ax = initialize_subplot(fig, 1, 1, 1, self.xlabel, self.ylabel)
        # add_satellite_image(ax, Plot2D.background_image)
        ax.grid(True)
        return fig, ax

    def modify_axes(self, ax: Axes) -> Axes:
        ax.set_title("")
        ax.set_aspect("equal")
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x/XMAJORTICK:.1f}")
        )
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x/XMAJORTICK:.1f}")
        )
        return ax

    def plot_profile_line(
        self,
        profile: LineString,
        bedlevel: xr.DataArray,
        riverkm: LineString,
        filename: Path,
    ) -> tuple[Figure, Axes]:
        """Plot the profile line in a 2D plot"""
        fig, ax = self.initialize_map()
        p = bedlevel.ugrid.plot.pcolormesh(
            ax=ax, add_colorbar=False, cmap="terrain", center=False
        )
        fig.colorbar(
            p, ax=ax, label="bodemligging [m]", orientation="horizontal", shrink=0.25
        )
        shapely.plotting.plot_line(profile, ax=ax, add_points=False, color="black")
        self.modify_axes(ax)
        plot_chainage_markers(riverkm, ax)
        savefig(fig, filename)
        return fig, ax


def modify_axes(ax: Axes, x_major_tick: float) -> Axes:
    # x-axis:
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/x_major_tick}"))
    ax.tick_params(which="major", length=8)
    ax.tick_params(which="minor", length=4)
    return ax


def construct_figure_filename(figdir: Path, base: str, extension: str) -> Path:
    """Construct full path for saving a figure."""
    return Path(figdir) / f"{base}{extension}"


@dataclass
class FlowfieldConfig:
    VELOCITY_YLABEL: str = "stroomsnelheid\nmagnitude" + r" [$m/s$]"
    VELOCITY_DIFF_YLABEL: str = "verschil plansituatie\n-referentie" + r" [$m/s$]"
    VELOCITY_YMIN: float = 0.0
    ANGLE_YTICKS = ticker.FixedLocator(list(np.arange(-90, 91, 22.5)))
    ANGLE_PRIMARY_YLABEL: str = "stromingshoek t.o.v.\nprofiellijn" + r" [$graden$]"
    # ANGLE_SECONDARY_YLABEL: str = r'stromingshoek [richting]'
    ANGLE_DIFF_YLABEL: str = "verschil plansituatie\n-referentie" + r" [$graden$]"
    # ANGLE_SECONDARY_YTICKLABELS = ticker.FixedFormatter(['Z','ZW','W','NW','N','NO','O','ZO','Z'])


@dataclass
class FroudeConfig:
    legend_title = "Froude getal"

    class Abs:
        colorbar_label: str = "Froude getal"
        levels: tuple = (0, 0.08, 0.1, 0.15)
        colormap: str = "RdBu"

    class Diff:
        bins: list = [0, 0.08, 0.1, 0.15, np.inf]

        # the following variables are linked to the classes returned by _compute_change_classes
        # colors: tuple = ("#d1ffbf", '#49e801', '#267500', '#f80000', '#fea703', '#fffe00')
        colors = ("blue", "red")
        labels: list[str] = [  # f"van < {bins[3]} naar >= {bins[3]}",
            # f"van < {bins[2]} naar >= {bins[2]}",
            f"van < {bins[1]} naar >= {bins[1]}",
            f"van > {bins[1]} naar <= {bins[1]}",
            # f"van > {bins[2]} naar <= {bins[2]}",
            # f"van > {bins[3]} naar <= {bins[3]}"
        ]


class Ice2D:

    def create_map(
        self,
        data: DataArray,
        riverkm: LineString,
        profile_line_df: GeoDataFrame,
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
        profile_line_df.plot(ax=ax, linewidth=1, color="green")
        savefig(fig, filename)

    def create_diff_map(
        self,
        ref_data: xr.DataArray,
        variant_data: xr.DataArray,
        riverkm: LineString,
        profile_line_df: GeoDataFrame,
        filename: Path,
    ) -> None:
        plt.close("all")
        bins = FroudeConfig.Diff.bins
        colors = FroudeConfig.Diff.colors
        labels = FroudeConfig.Diff.labels

        # Step 1: Digitize inputs
        ref_data_digitized = self._digitize(ref_data.values, bins)
        variant_data_digitized = self._digitize(variant_data.values, bins)

        # Step 2: Classify change categories
        classes = self._compute_change_classes(
            ref_data_digitized, variant_data_digitized
        )
        variant_data.values = classes

        # Step 3: Initialize figure with background plot
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

        # Step 4: Difference plot
        ax, legend_elements = self._plot_diff_map(ax, variant_data, labels, colors)

        # Step 5: finalisation
        ax = Plot2D().modify_axes(ax)
        lgd = fig.legend(
            [Patch(facecolor=color), *legend_elements],
            [f"< {bins[1]} in referentie", *labels],
        )
        lgd.set_title(FroudeConfig.legend_title)
        ax.grid(True)
        plot_chainage_markers(riverkm, ax)
        profile_line_df.plot(ax=ax, linewidth=0.5, color="black")
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
        """Computes how classes change between two digitized datasets"""
        classes = variant_data * np.nan

        conditions = [  # (ref_data < 3) & (variant_data >= 3),
            # (ref_data < 2) & (variant_data >= 2),
            (ref_data < 1) & (variant_data >= 1),
            (ref_data > 0) & (variant_data <= 0),
            # (ref_data > 1) & (variant_data <= 1),
            # (ref_data > 2) & (variant_data <= 2)
        ]

        for i, cond in enumerate(conditions, start=1):
            classes[cond] = i

        return classes


class Ice1D:
    """Class for plotting 1D river flow velocity and angle."""

    def plot_velocity_magnitude(
        self, ax: Axes, distance: np.ndarray, velocity: np.ndarray, color: str
    ) -> Axes:
        """
        Plot the velocity magnitude.
        """
        plot_variable(ax, distance, velocity, color)
        # ax.set_ylim(bottom=FlowfieldConfig.VELOCITY_YMIN)
        return ax

    def plot_velocity_angle(
        self, ax: Axes, distance: np.ndarray, angle: np.ndarray, color: str
    ) -> Axes:
        """
        Plot the velocity angle in a separate subplot.
        """
        plot_variable(ax, distance, angle, color)
        return ax

    # def angle_direction(self, ax: Axes):
    #     secax_y = ax.secondary_yaxis(-0.2)
    #     for ax in [ax,secax_y]:
    #     secax_y.yaxis.set_major_formatter(FlowfieldConfig.ANGLE_SECONDARY_YTICKLABELS)
    #     secax_y.set_ylabel(FlowfieldConfig.ANGLE_SECONDARY_YLABEL)
    #     return secax_y

    def create_figure(
        self,
        distance: np.ndarray,
        velocity: list,
        angle: list,
        configuration: Config,
        filename: Path,
    ) -> None:
        """
        Create and display a figure with velocity magnitude and angle.
        """
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
            ax1 = self.plot_velocity_magnitude(ax1, distance, v, Plot1DConfig.COLORS[i])
            ax2 = self.plot_velocity_angle(ax2, distance, a, Plot1DConfig.COLORS[i])

        axs_diff = []
        if len(velocity) > 1:
            for ax, data, ylabel in [
                (ax1, velocity[1] - velocity[0], FlowfieldConfig.VELOCITY_DIFF_YLABEL),
                (ax2, angle[1] - angle[0], FlowfieldConfig.ANGLE_DIFF_YLABEL),
            ]:
                ax_diff = difference_plot(ax, ylabel, Plot1DConfig.COLORS[-1])
                plot_variable(ax_diff, distance, data, Plot1DConfig.COLORS[-1])
                yabs_max = abs(max(ax_diff.get_ylim(), key=abs))
                ax_diff.set_ylim(ymin=-yabs_max, ymax=yabs_max)
                axs_diff.append(ax_diff)

        # Align gridlines and keep secondary centered at 0
        for primary_axis, secondary_axis in zip([ax1, ax2], axs_diff):
            align_twinx_grid_centered(
                primary_axis,
                secondary_axis,
                center=0.0,
                keep_symmetric=True,
                add_centerline=True,
            )

        for ax in [ax1, ax2]:
            ax1 = modify_axes(ax1, XMAJORTICK)
            ax2 = modify_axes(ax2, XMAJORTICK)
            if configuration.general.bool_flags["invertxaxis"]:
                invert_xaxis(ax)
        ax2.yaxis.set_major_locator(FlowfieldConfig.ANGLE_YTICKS)
        ax2.set_ylim(-90, 90)
        # ax2.axhline(0,color='black',ls='--')

        ax1.legend(
            Plot1DConfig.LABELS,
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            ncols=2,
            borderaxespad=0.0,
        )
        savefig(fig, filename)


@dataclass
class CrossFlowConfig:
    XLABEL = Plot1DConfig.XLABEL
    YLABEL: str = "dwarsstroom-\nsnelheid" + r" [$m/s$]"
    DIFF_YLABEL: str = "verschil in dwars-\nstroomsnelheid" + r" [$m/s$]"


class CrossFlow:
    def __init__(self, config: CrossFlowConfig = CrossFlowConfig()):
        self.config = config

    def plot_discharge(
        self,
        ax: Axes,
        xy_segment: list[tuple],
        crit_values: list,
    ) -> Optional[LineCollection]:
        """
        Calculate and plot perpendicular discharge according to RBK specifications,
        along with the discharge criteria line.

        Returns:
            A matplotlib Line2D object representing the criteria line, or None if no data was plotted.
        """
        crit_handle = None

        for (xi, yi), crit_value in zip(xy_segment, crit_values):
            # TODO: fix fill between not filling in everything
            ax.fill_between(xi, yi, color="lightgrey", interpolate=True)
            ax.axvline(xi[0], color="lightgrey", lw=0.5, ls="--")
            ax.axvline(xi[-1], color="lightgrey", lw=0.5, ls="--")

            # positive criterium:
            crit_handle = ax.hlines(
                crit_value, xi[0], xi[-1], color="red", lw=1, ls="-"
            )
            # negative criterium:
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
        axs = []
        ax1 = initialize_subplot(
            fig, len(transverse_velocity), 1, 1, self.config.XLABEL, self.config.YLABEL
        )
        axs.append(ax1)

        crit_handle = self.plot_discharge(ax1, xy_segments[-1], crit_values[-1])

        lines = []
        for i, v in enumerate(transverse_velocity):
            (line,) = plot_variable(ax1, distance, v, Plot1DConfig.COLORS[i])
            lines.append(line)

        if len(transverse_velocity) > 1:
            ax2 = initialize_subplot(
                fig, 2, 1, 2, self.config.XLABEL, CrossFlowConfig.DIFF_YLABEL
            )
            plot_variable(
                ax2, distance, transverse_velocity[1] - transverse_velocity[0]
            )
            axs.append(ax2)

        for ax in axs:
            modify_axes(ax, XMAJORTICK)
            yabs_max = abs(max(ax.get_ylim(), key=abs))
            ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)
            ax.axhline(0, color="black", ls="--")
            if inverse_xaxis:
                invert_xaxis(ax)
            ax.grid(visible=True, which="major", linestyle="-")
            ax.grid(
                visible=True,
                which="minor",
                axis="y",
                linestyle="--",
                color="lightgrey",
                lw=0.5,
            )

        # Combine lines and crit_handle, filtering out None
        handles = [*lines]
        labels = [*Plot1DConfig.LABELS[0 : len(transverse_velocity)]]

        if crit_handle is not None:
            handles.append(crit_handle)
            labels.append("criteria")

        ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.15))
        ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        ax1.legend(
            handles,
            labels,
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            ncols=3,
            borderaxespad=0.0,
        )

        savefig(fig, filename)
