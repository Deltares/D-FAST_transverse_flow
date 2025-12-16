from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import shapely.plotting
import xarray as xr
import xugrid as xu
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
from dfasttf.config import Config

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
    fig.set_figwidth(figwidth)
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
    secax_y2.tick_params(color=color, labelcolor=color, which='both')
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

@dataclass
class Plot1DConfig:
    XLABEL: str = "raai km"
    DELTARES_BLUE = "#0D38DF"
    DELTARES_DARKGREEN = "#00B389"
    COLORS = ("k", DELTARES_BLUE, DELTARES_DARKGREEN)  # reference, intervention, difference
    LABELS = ["Referentie", "Plansituatie", "Verschil"]

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
    VELOCITY_YLABEL: str = "stroomsnelheid\nmagnitude" + r" [m/s]"
    VELOCITY_DIFF_YLABEL: str = "verschil plansituatie\n-referentie" + r" [m/s]"
    VELOCITY_YLIM: tuple = (0., 2.0)
    VELOCITY_YTICKS_MAJOR: float = 0.4
    VELOCITY_YTICKS_MINOR: float = 0.1
    ANGLE_YTICKS_MAJOR: float = 30.
    ANGLE_YTICKS_MINOR: float = 10.
    ANGLE_YLIM: tuple = (-90., 90.)
    # ANGLE_SECONDARY_YLABEL: str = r'stromingshoek [richting]'
    ANGLE_PRIMARY_YLABEL: str = "stromingshoek t.o.v.\nprofiellijn" + r" [graden]"
    ANGLE_DIFF_YLABEL: str = "verschil plansituatie\n-referentie" + r" [graden]"
    # ANGLE_SECONDARY_YTICKLABELS = ticker.FixedFormatter(['Z','ZW','W','NW','N','NO','O','ZO','Z'])
    FRACTION: float = 5.

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
            shapely.plotting.plot_line(profile_line, ax=ax, add_points=False, color=FroudeConfig.profile_line_color)
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
        if profile_line is not None:
            shapely.plotting.plot_line(profile_line, ax=ax, add_points=False, color=FroudeConfig.profile_line_color)
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
        ax.set_ylim(FlowfieldConfig.VELOCITY_YLIM)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(FlowfieldConfig.VELOCITY_YTICKS_MAJOR))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(FlowfieldConfig.VELOCITY_YTICKS_MINOR))
        return ax

    def plot_velocity_angle(
        self, ax: Axes, distance: np.ndarray, angle: np.ndarray, color: str
    ) -> Axes:
        """
        Plot the velocity angle in a separate subplot.
        """
        plot_variable(ax, distance, angle, color)
        ax.set_ylim(FlowfieldConfig.ANGLE_YLIM)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(FlowfieldConfig.ANGLE_YTICKS_MAJOR))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(FlowfieldConfig.ANGLE_YTICKS_MINOR))
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
        fraction = FlowfieldConfig.FRACTION
        if len(velocity) > 1:
            for ax, data, ylabel in [
                (ax1, velocity[1] - velocity[0], FlowfieldConfig.VELOCITY_DIFF_YLABEL),
                (ax2, angle[1] - angle[0], FlowfieldConfig.ANGLE_DIFF_YLABEL),
            ]:
                ax_diff = difference_plot(ax, ylabel, Plot1DConfig.COLORS[-1])
                plot_variable(ax_diff, distance, data, Plot1DConfig.COLORS[-1])
                ax_diff.set_ylim(-ax.get_ylim()[1] / fraction, ax.get_ylim()[1] / fraction)
                axs_diff.append(ax_diff)

            axs_diff[0].yaxis.set_major_locator(ticker.MultipleLocator(FlowfieldConfig.VELOCITY_YTICKS_MAJOR / fraction))
            axs_diff[1].yaxis.set_major_locator(ticker.MultipleLocator(FlowfieldConfig.ANGLE_YTICKS_MAJOR / fraction))

        for ax in [ax1, ax2]:
            ax1 = modify_axes(ax1, XMAJORTICK)
            ax2 = modify_axes(ax2, XMAJORTICK)
            if configuration.general.bool_flags["invertxaxis"]:
                invert_xaxis(ax)
            ax.grid(visible=True, which="major", linestyle="-")
            ax.grid(
                visible=True,
                which="minor",
                axis="y",
                linestyle="--",
                color="lightgrey"
            )

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
    YLABEL: str = "representatieve dwars-\nstroomsnelheid" + r" [m/s]"
    DIFF_YLABEL: str = FlowfieldConfig.VELOCITY_DIFF_YLABEL
    CRIT_LABEL: str = "Lokaal criterium"
    YLIM: tuple = (-0.3, 0.3)
    FRACTION: int = 3
    YTICKS_MAJOR = 0.15
    YTICKS_MINOR = 0.05

class CrossFlow:
    def __init__(self, config: CrossFlowConfig = CrossFlowConfig()):
        self.config = config

    def plot_discharge(
        self,
        ax: Axes,
        xy_segments: list[list[tuple]],
        crit_values: list[np.ndarray],
    ) -> Optional[LineCollection]:
        """
        Calculate and plot perpendicular discharge according to RBK specifications,
        along with the discharge criteria line.

        Returns:
            A matplotlib Line2D object representing the criteria line, or None if no data was plotted.
        """
        crit_handle = None
        xy_segments = xy_segments[-1]
        crit_values = crit_values[-1]
        
        for (xi, yi), crit_value in zip(xy_segments, crit_values):
            ax.fill_between(xi, yi, color="lightgrey", interpolate=True)

            # positive criterium:
            crit_handle = ax.hlines(
                crit_value, xi[0], xi[-1], color='red', lw=1, ls="-"
            )
            # negative criterium:
            ax.hlines(-crit_value, xi[0], xi[-1], color='red', lw=1, ls="-")

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
            fig, 2, 1, 1, self.config.XLABEL, self.config.YLABEL
        )
        axs.append(ax1)
        ax1.set_ylim(self.config.YLIM)

        crit_handle = self.plot_discharge(ax1, xy_segments, crit_values)

        lines = []
        for i, v in enumerate(transverse_velocity):
            (line,) = plot_variable(ax1, distance, v, Plot1DConfig.COLORS[i])
            lines.append(line)

        fraction = self.config.FRACTION
        if len(transverse_velocity) > 1:
            ax2 = difference_plot(ax1, CrossFlowConfig.DIFF_YLABEL, Plot1DConfig.COLORS[-1])
            data = transverse_velocity[1] - transverse_velocity[0]
            ax2.set_ylim([y / fraction for y in ax1.get_ylim()])
            (diff,) = plot_variable(
                ax2, distance, data, color=Plot1DConfig.COLORS[-1]
            )
            axs.append(ax2)

        modify_axes(ax1, XMAJORTICK)
        if inverse_xaxis:
            invert_xaxis(ax1)
        ax1.grid(visible=True, which="major", linestyle="-")
        ax1.grid(
            visible=True,
            which="minor",
            axis="y",
            linestyle="--",
            color="lightgrey"
        )

        # Combine lines and crit_handle, filtering out None
        handles = [*lines]
        labels = [*Plot1DConfig.LABELS[0 : len(transverse_velocity)]]

        if crit_handle is not None:
            handles.append(crit_handle)
            labels.append(CrossFlowConfig.CRIT_LABEL)

        ax1.yaxis.set_major_locator(ticker.MultipleLocator(CrossFlowConfig.YTICKS_MAJOR))
        ax1.yaxis.set_minor_locator(ticker.MultipleLocator(CrossFlowConfig.YTICKS_MINOR))
        if len(transverse_velocity) > 1:
            ax2.yaxis.set_major_locator(ticker.MultipleLocator(CrossFlowConfig.YTICKS_MAJOR / fraction))
            ax2.yaxis.set_minor_locator(ticker.MultipleLocator(CrossFlowConfig.YTICKS_MINOR / fraction))
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

        savefig(fig, filename)
