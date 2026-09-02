"""2D map plotting (bed level maps, Froude number maps) built on xugrid."""
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shapely.plotting
import xarray as xr
import xugrid as xu
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from shapely.geometry import LineString
from xarray import DataArray
from xugrid import UgridDataArray

from dfasttf.batch.plotting.common import (
    FIGWIDTH,
    XMAJORTICK,
    initialize_figure,
    initialize_subplot,
    plot_chainage_markers,
    savefig,
)
from dfasttf.batch.plotting.configs import FroudeConfig


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
            ax=ax, add_colorbar=False, cmap="terrain", center=False, edgecolors='none'
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
            edgecolors='none'
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
        ref_data: UgridDataArray,
        variant_data: UgridDataArray,
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
        dry_ref_mask = np.isnan(ref_data.values)
        wet_variant_mask = ~np.isnan(variant_data.values)

        classes = self._compute_change_classes(
            ref_data_digitized,
            variant_data_digitized,
            dry_ref_mask,
            wet_variant_mask,
        )
        variant_data.values = classes

        fig, ax = Plot2D().initialize_map()
        p = ref_data.ugrid.plot(
            ax=ax,
            add_colorbar=False,
            levels=FroudeConfig.Abs.levels,
            cmap=FroudeConfig.Abs.colormap,
            extend="max",
            alpha=0.5,
            edgecolors='none'
        )
        fig.colorbar(
            p,
            ax=ax,
            label=FroudeConfig.Abs.colorbar_label + '\nin referentie',
            orientation="horizontal",
            shrink=0.25,
        )
        # color = "lightgrey"
        # ref_masked = ref_data[ref_data_digitized == 0]
        # ref_masked.ugrid.plot(
        #     ax=ax,
        #     cmap=ListedColormap([color]),
        #     add_colorbar=False,
        #     vmin=bins[0],
        #     vmax=bins[1],
        # )

        ax, legend_elements = self._plot_diff_map(ax, variant_data, labels, colors)

        ax = Plot2D().modify_axes(ax)
        lgd = ax.legend(legend_elements,
                        labels,
                        loc='lower center',
                        bbox_to_anchor=(0.5,1)
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
            edgecolors='none'
        )

        legend_elements = [
            Patch(facecolor=colors[i], label=labels[i]) for i in range(len(labels))
        ]
        return ax, legend_elements

    def _digitize(self, data: np.ndarray, bins: list[float]) -> np.ndarray:
        return np.digitize(data, bins) - 1

    def _compute_change_classes(
        self, ref_data: np.ndarray, variant_data: np.ndarray, dry_ref_mask: np.ndarray, wet_variant_mask: np.ndarray
    ) -> np.ndarray:
        classes = variant_data * np.nan
        conditions = [
            (ref_data < 1) & (variant_data >= 1),
            (ref_data > 0) & (variant_data <= 0),
            dry_ref_mask & wet_variant_mask
        ]
        for i, cond in enumerate(conditions, start=1):
            classes[cond] = i
        return classes
