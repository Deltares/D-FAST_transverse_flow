"""Longitudinal flow (langsstroming) 1D and tide-related plotting."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from dfasttf.batch.plotting.common import (
    FIGWIDTH,
    difference_plot,
    format_datetime_xaxis,
    initialize_figure,
    initialize_subplot,
    make_marker_proxy,
    make_rkm_colormap,
    plot_variable,
    savefig,
    scatter_idx_points_on_timeseries,
    add_side_legend,
    style_1d_axis,
    fix_overlapping_ticklabels,
    set_twin_yaxis_limits,
)
from dfasttf.batch.plotting.configs import LongitudinalFlowConfig, Plot1DConfig, TideLongitudinalFlowConfig

class LongitudinalFlow:
    def __init__(self, config: LongitudinalFlowConfig = LongitudinalFlowConfig()):
        self.config = config

    def create_figure(self, 
                      distance: np.ndarray, 
                      upar: list[np.ndarray], 
                      inverse_xaxis: bool,
                      filename: Path,
                      include_difference: bool = True) -> None:
        """Create a figure of the longitudinal flow velocity along the river distance."""
        plt.close("all")
        fig = initialize_figure()
        ax1 = initialize_subplot(fig, 1, 1, 1, self.config.XLABEL, self.config.YLABEL)
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(self.config.YTICKS_MAJOR))
        ax1.yaxis.set_minor_locator(ticker.MultipleLocator(self.config.YTICKS_MINOR))
        lines = []
        for i, v in enumerate(upar):
            (line,) = plot_variable(ax1, distance, v, Plot1DConfig.COLORS[i])
            lines.append(line)

        fraction = Plot1DConfig.FRACTION
        ax2 = None
        diff = None
        if include_difference and len(upar) > 1:
            ax2 = difference_plot(ax1, self.config.DIFF_YLABEL, Plot1DConfig.COLORS[-1])
            data = upar[1] - upar[0]
            set_twin_yaxis_limits(ax1, ax2, upar[1], data, fraction)
            (diff,) = plot_variable(ax2, distance, data, color=Plot1DConfig.COLORS[-1])

        style_1d_axis(ax1, inverse_xaxis)

        handles = [*lines]
        labels = [*Plot1DConfig.LABELS[0:len(upar)]]

        ax1.yaxis.set_major_locator(ticker.MultipleLocator(self.config.YTICKS_MAJOR))
        ax1.yaxis.set_minor_locator(ticker.MultipleLocator(self.config.YTICKS_MINOR))

        if ax2 is not None and diff is not None:
            ax2.yaxis.set_major_locator(
                ticker.MultipleLocator(self.config.YTICKS_MAJOR / fraction)
            )
            ax2.yaxis.set_minor_locator(
                ticker.MultipleLocator(self.config.YTICKS_MINOR / fraction)
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

        # Check and fix overlapping tick labels on y-axis:
        for ax in [ax1, ax2]:
            if ax is None:
                continue
            fix_overlapping_ticklabels(ax, axis="y")

        savefig(fig, filename)


    def create_figure_tide_timeseries(
            self,
            time: np.ndarray | None,
            upar_tn: np.ndarray,
            rkm: np.ndarray,
            filename: Path,
            threshold: float | None,
            annotation: str | None = None,
            idx_ebb: np.ndarray | None = None,
            idx_flood: np.ndarray | None = None,
        ) -> None:
            """Create a figure of the longitudinal flow velocity over time for each `rkm`."""
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
                label=TideLongitudinalFlowConfig.AVERAGE_FLOW_LABEL,
            )

            thr_handle = None
            if threshold is not None and np.isfinite(threshold) and threshold > 0:
                thr_handle = ax.axhline(
                    +threshold,
                    color="red",
                    lw=1.0,
                    ls="--",
                    label=TideLongitudinalFlowConfig.THRESHOLD_LABEL,
                )
                ax.axhline(-threshold, color="red", lw=1.0, ls="--")

            format_datetime_xaxis(fig, ax, x)

            ax.set_xlabel(TideLongitudinalFlowConfig.XLABEL)
            ax.set_ylabel(LongitudinalFlowConfig.YLABEL)
            ax.grid(True, which="major", linestyle="-", alpha=0.6)
            ax.yaxis.set_major_locator(
                ticker.MultipleLocator(LongitudinalFlowConfig.YTICKS_MAJOR)
            )
            ax.yaxis.set_minor_locator(
                ticker.MultipleLocator(LongitudinalFlowConfig.YTICKS_MINOR)
            )
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
                legend_labels.append(TideLongitudinalFlowConfig.MAX_EBB_LABEL)

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
                legend_labels.append(TideLongitudinalFlowConfig.MAX_FLOOD_LABEL)

            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
            cbar.set_label(TideLongitudinalFlowConfig.CBAR_LABEL)

            leg_handles = [mean_line]
            leg_labels = [TideLongitudinalFlowConfig.AVERAGE_FLOW_LABEL]
            if thr_handle is not None:
                leg_handles.append(thr_handle)
                leg_labels.append(TideLongitudinalFlowConfig.THRESHOLD_LABEL)
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