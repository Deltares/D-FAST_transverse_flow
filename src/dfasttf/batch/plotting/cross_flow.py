"""Cross-flow (dwarsstroming) 1D and tide-related plotting."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from dfasttf.batch.plotting.common import (
    FIGWIDTH,
    add_side_legend,
    difference_plot,
    format_datetime_xaxis,
    format_datetime_yaxis,
    initialize_figure,
    initialize_subplot,
    make_marker_proxy,
    make_rkm_colormap,
    plot_variable,
    savefig,
    scatter_idx_points_on_timeseries,
    style_1d_axis,
)
from dfasttf.batch.plotting.configs import CrossFlowConfig, DirectionalMaximaConfig, Plot1DConfig


class CrossFlow:
    def __init__(self, config: CrossFlowConfig = CrossFlowConfig()):
        self.config = config

    def plot_discharge(
        self,
        ax: Axes,
        xy_segments: list[list[tuple]],
        crit_values: list[np.ndarray],
    ) -> LineCollection | None:
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
        include_difference: bool = True,
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
        if include_difference and len(transverse_velocity) > 1:
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
        annotation: str | None = None,
    ) -> None:
        plt.close("all")
        fig = initialize_figure()

        # extra row on top for legend
        gs = fig.add_gridspec(
            nrows=3,
            ncols=1,
            height_ratios=[1.3, 8, 8],
        )

        lax = fig.add_subplot(gs[0, 0])   # legend axis
        ax1 = fig.add_subplot(gs[1, 0])   # ebb
        ax2 = fig.add_subplot(gs[2, 0])   # flood

        lax.axis("off")

        ax1.set_xlabel(self.config.XLABEL)
        ax1.set_ylabel(self.config.YLABEL)
        ax2.set_xlabel(self.config.XLABEL)
        ax2.set_ylabel(self.config.YLABEL)

        ax1.set_title(self.config.EBB_TITLE)
        ax2.set_title(self.config.FLOOD_TITLE)

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
        legend = lax.legend(
            handles,
            labels,
            loc="center",
            ncols=min(3, n_present),
            frameon=True,
            facecolor="white",
            framealpha=1.0,
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
        #fig.subplots_adjust(top=0.7)
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
        1) maximum representative transverse velocity over the full tide cycle vs rkm
        2) alongstream velocity through time with points at idx_tvmax
        3) x-t heatmap of |tv_tn| with the same points overlaid
        """
        plt.close("all")
        fig = initialize_figure(figwidth=1.35 * FIGWIDTH)

        # left column = plots
        # right column = legends / colorbars
        outer = fig.add_gridspec(
            nrows=3,
            ncols=2,
            width_ratios=[20, 4.5],
            height_ratios=[9, 10, 8],
        )

        ax1 = fig.add_subplot(outer[0, 0])
        ax2 = fig.add_subplot(outer[1, 0])
        ax3 = fig.add_subplot(outer[2, 0])

        # right column: legend-only axis for subplot 1, full-height colorbar
        # axes for subplots 2 and 3 (their small legends are placed on the
        # plot axes themselves, so the colorbars stay aligned with the full
        # height of their corresponding plot).
        lax1 = fig.add_subplot(outer[0, 1])   # legend only for subplot 1
        cax1 = fig.add_subplot(outer[1, 1])   # colorbar for subplot 2
        cax2 = fig.add_subplot(outer[2, 1])   # colorbar for subplot 3

        lax1.axis("off")

        # ------------------------------------------------------------
        # Subplot 1: max representative transverse velocity vs rkm
        # ------------------------------------------------------------
        lines1 = []
        for i, y in enumerate(tv_max_list):
            if y is None:
                continue
            (line,) = plot_variable(ax1, rkm, y, Plot1DConfig.COLORS[i])
            lines1.append(line)

        ax1.set_title("Maximale representatieve dwars-\nstroomsnelheid over getijcyclus (per cel)")
        ax1.set_xlabel("raai km")
        ax1.set_ylabel("representatieve dwars-\nstroomsnelheid [m/s]")
        style_1d_axis(ax1, inverse_xaxis)

        n_present = sum(y is not None for y in tv_max_list)
        labels1 = Plot1DConfig.LABELS[0:n_present]
        handles1 = lines1[:n_present]

        lax1.legend(
            handles1,
            labels1,
            loc="center left",
            frameon=False,
            fontsize=8,
            handlelength=1.2,
            handletextpad=0.4,
            labelspacing=0.4,
            borderaxespad=0.0,
        )

        # ------------------------------------------------------------
        # Subplot 2: alongstream velocity through time + idx_tvmax points
        # ------------------------------------------------------------
        if time is None or np.asarray(time).size == 0:
            x = np.arange(upar_tn.shape[0])
        else:
            x = np.asarray(time)

        # color scale for subplot 2: raai km
        rk_km, norm_rkm, cmap_rkm = make_rkm_colormap(rkm)

        nt, n = upar_tn.shape
        for i in range(n):
            ax2.plot(
                x,
                upar_tn[:, i],
                color=cmap_rkm(norm_rkm(rk_km[i])),
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
            norm_rkm,
            cmap_rkm,
            marker="s",
            edgecolor="magenta",
            zorder=8,
            size=18,
            linewidth=0.35,
        )

        format_datetime_xaxis(fig, ax2, x)
        ax2.tick_params(axis="x", which="both", labelbottom=True)

        ax2.set_title("Moment van maximale dwarsstroomsnelheid")
        ax2.set_xlabel("tijd")
        ax2.set_ylabel("langsstroomsnelheid\n[m/s]")
        ax2.grid(True, which="major", linestyle="-", alpha=0.6)
        ax2.grid(True, which="minor", linestyle="--", color="lightgrey", alpha=0.6)

        # legend for subplot 2 (mean line), placed inside the plot itself so
        # the colorbar column stays free to span the full row height.
        ax2.legend(
            [mean_line],
            ["gemiddelde"],
            loc="upper right",
            frameon=True,
            facecolor="white",
            framealpha=0.85,
            fontsize=8,
            handlelength=1.2,
            handletextpad=0.4,
            labelspacing=0.4,
            borderaxespad=0.3,
        )

        # colorbar for subplot 2: raai km, spanning the full height of its row
        # so it stays aligned with ax2.
        sm_rkm = plt.cm.ScalarMappable(norm=norm_rkm, cmap=cmap_rkm)
        sm_rkm.set_array([])
        cbar1 = fig.colorbar(sm_rkm, cax=cax1, orientation="vertical")
        cbar1.set_label("raai km")

        # ticks on every whole km, kept within [vmin, vmax] so the tick
        # range never extends past the actual colored gradient (which would
        # otherwise leave a blank gap on the colorbar).
        raai_ticks = np.arange(
            np.ceil(np.nanmin(rk_km)),
            np.floor(np.nanmax(rk_km)) + 1,
            1.0,
        )
        cbar1.set_ticks(raai_ticks)
        cbar1.set_ticklabels([f"{tick:.0f}" for tick in raai_ticks])

        # ------------------------------------------------------------
        # Subplot 3: x-t heatmap of |tv_tn| + idx_tvmax points
        # ------------------------------------------------------------
        if time is None or np.asarray(time).size == 0:
            y_time = np.arange(tv_tn.shape[0])
        else:
            y_time = np.asarray(time)

        x_rkm = np.asarray(rkm, dtype=float)

        # different colormap from subplot 2
        cmap_tv = "magma"

        pcm = ax3.pcolormesh(
            x_rkm,
            y_time,
            np.abs(tv_tn),
            shading="auto",
            cmap=cmap_tv,
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

        ax3.set_title("Absolute dwarsstroomsnelheid over tijd (per cel)")
        ax3.set_xlabel("raai km")
        ax3.set_ylabel("tijd")
        style_1d_axis(ax3, inverse_xaxis)
        format_datetime_yaxis(ax3, y_time)
        ax3.tick_params(axis="x", which="both", labelbottom=True)

        # proxy marker + legend for subplot 3
        proxy_max_tv = make_marker_proxy(
            marker="s",
            edgecolor="magenta",
            facecolor="white",
            markersize=6,
            markeredgewidth=0.5,
        )

        # legend for subplot 3 (max transverse-velocity marker), placed inside
        # the plot itself so the colorbar column stays free to span the full
        # row height.
        ax3.legend(
            [proxy_max_tv],
            ["moment max. dwarsstroming"],
            loc="upper right",
            frameon=True,
            facecolor="white",
            framealpha=0.85,
            fontsize=8,
            handlelength=1.2,
            handletextpad=0.4,
            labelspacing=0.4,
            borderaxespad=0.3,
        )

        # colorbar for subplot 3: |tv|, spanning the full height of its row
        # so it stays aligned with ax3.
        cbar2 = fig.colorbar(pcm, cax=cax2, orientation="vertical")
        cbar2.set_label("|dwarsstroom-\nsnelheid| [m/s]")

        fig.set_figheight(1.20 * FIGWIDTH)
        savefig(fig, filename)

    def _plot_directional_scalar_panel(
        self,
        ax: Axes,
        rkm: np.ndarray,
        values_list: list[np.ndarray | None],
        diff_ylabel: str,
        fraction: float,
    ) -> tuple[list[Line2D], list[str]]:
        """Plot Referentie/Plansituatie on the primary axis and, if present,
        Verschil on a smaller secondary axis. Matches the reference/
        intervention/difference pattern already used throughout this module
        (see `create_figure` and `Ice1D.create_figure`), so a panel only
        ever carries one quantity/unit and up to 3 directly comparable lines.
        """
        lines: list[Line2D] = []
        labels: list[str] = []

        n_present = sum(v is not None for v in values_list[:2])
        for i in range(n_present):
            (line,) = plot_variable(ax, rkm, values_list[i], Plot1DConfig.COLORS[i])
            lines.append(line)
            labels.append(Plot1DConfig.LABELS[i])

        if len(values_list) > 2 and values_list[2] is not None:
            ax_diff = difference_plot(ax, diff_ylabel, Plot1DConfig.COLORS[-1])
            # Scale the secondary axis symmetrically around zero, using the
            # primary axis's span (not its absolute limits) divided by
            # `fraction`. Unlike e.g. CrossFlow.create_figure, the primary
            # axis here is not fixed to a zero-centered range (velocity and
            # discharge magnitudes are positive-only), so naively dividing
            # `ax.get_ylim()` would shift the secondary axis away from zero
            # and clip negative "Verschil" values off the visible range.
            y0, y1 = ax.get_ylim()
            half_range = (y1 - y0) / (2.0 * fraction)
            ax_diff.set_ylim(-half_range, half_range)
            (diff_line,) = plot_variable(
                ax_diff, rkm, values_list[2], color=Plot1DConfig.COLORS[-1]
            )
            lines.append(diff_line)
            labels.append(Plot1DConfig.LABELS[-1])

        return lines, labels

    def create_figure_directional_maxima(
        self,
        rkm: np.ndarray,
        tv_max_list: list[np.ndarray | None],
        q_list: list[np.ndarray | None],
        velocity_title: str,
        discharge_title: str,
        inverse_xaxis: bool,
        filename: Path,
    ) -> None:
        """
        Figure for a single transverse-flow direction (bankward or
        riverward), per profile position along the reference line:
        1) the maximum transverse velocity in that direction (top row);
        2) the instantaneous transverse discharge at that same moment
           (bottom row).

        Velocity and discharge are evaluated at the same instant (rather than
        each being independently maximized), matching the RBK review
        methodology: only a same-phase combination of velocity and discharge
        is physically realistic for assessing impulse on passing ships.

        Bankward and riverward directions are rendered as two separate
        figures (call this method once per direction) so every panel carries
        a single unit/quantity instead of overlaying velocity and discharge
        on a shared dual axis.
        """
        plt.close("all")
        # Wider than the default figure width: each panel here has a
        # secondary "Verschil" axis (unlike e.g. create_figure_tide_velocities),
        # which needs extra horizontal room so the primary plot area doesn't
        # end up narrower than in the other tide figures.
        fig = initialize_figure(figwidth=1.35 * FIGWIDTH)
        config = DirectionalMaximaConfig()
        fraction = CrossFlowConfig.FRACTION

        ax_v = initialize_subplot(fig, 2, 1, 1, config.XLABEL, config.VELOCITY_YLABEL)
        ax_q = initialize_subplot(fig, 2, 1, 2, config.XLABEL, config.DISCHARGE_YLABEL)

        ax_v.set_title(velocity_title)
        ax_q.set_title(discharge_title)

        lines, labels = self._plot_directional_scalar_panel(
            ax_v, rkm, tv_max_list, config.VELOCITY_DIFF_YLABEL, fraction
        )
        self._plot_directional_scalar_panel(
            ax_q, rkm, q_list, config.DISCHARGE_DIFF_YLABEL, fraction
        )

        for ax in (ax_v, ax_q):
            style_1d_axis(ax, inverse_xaxis)

        # Figure-level legend above both rows, so it doesn't collide with
        # the top row's own title.
        fig.legend(
            lines,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncols=len(lines),
            fontsize=8,
            frameon=False,
            borderaxespad=0.3,
        )

        # Keep the same absolute height as before; only the width increased,
        # giving each panel a wider, less cramped (banner-like) aspect ratio
        # consistent with the other tide figures in this module.
        fig.set_figheight(0.9 * FIGWIDTH)
        savefig(fig, filename)
