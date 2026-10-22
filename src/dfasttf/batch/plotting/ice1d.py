"""1D ice-scenario flow velocity/angle plotting."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from dfasttf.batch.plotting.common import (
    difference_plot,
    initialize_figure,
    initialize_subplot,
    plot_variable,
    savefig,
    set_twin_yaxis_limits,
    style_1d_axis,
    fix_overlapping_ticklabels,
)
from dfasttf.batch.plotting.configs import FlowfieldConfig, Plot1DConfig
from dfasttf.config import Config


class Ice1D:
    """Class for plotting 1D river flow velocity and angle."""

    def plot_velocity_magnitude(
        self, ax: Axes, distance: np.ndarray, velocity: np.ndarray, color: str
    ) -> list[Line2D]:
        lines = plot_variable(ax, distance, velocity, color)
        # ax.yaxis.set_major_locator(
        #     ticker.MultipleLocator(FlowfieldConfig.VELOCITY_YTICKS_MAJOR)
        # )
        # ax.yaxis.set_minor_locator(
        #     ticker.MultipleLocator(FlowfieldConfig.VELOCITY_YTICKS_MINOR)
        # )
        ax.yaxis.set_major_locator(ticker.MaxNLocator('auto'))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        return lines

    def plot_velocity_angle(
        self, ax: Axes, distance: np.ndarray, angle: np.ndarray, color: str
    ) -> list[Line2D]:
        lines = plot_variable(ax, distance, angle, color)
        ax.yaxis.set_major_locator(
            ticker.MultipleLocator(FlowfieldConfig.ANGLE_YTICKS_MAJOR)
        )
        ax.yaxis.set_minor_locator(
            ticker.MultipleLocator(FlowfieldConfig.ANGLE_YTICKS_MINOR)
        )
        return lines

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

        lines: list[Line2D] = []
        for i, (v, a) in enumerate(zip(velocity, angle)):
            (line,) = self.plot_velocity_magnitude(ax1, distance, v, Plot1DConfig.COLORS[i])
            (line,) = self.plot_velocity_angle(ax2, distance, a, Plot1DConfig.COLORS[i])
            lines.append(line)

        fraction = Plot1DConfig.FRACTION
        axs_diff = []
        if len(velocity) > 1:
            for idx, (ax, data, diff_data, ylabel) in enumerate([
                (ax1, velocity[1], velocity[1] - velocity[0], FlowfieldConfig.VELOCITY_DIFF_YLABEL),
                (ax2, angle[1], angle[1] - angle[0], FlowfieldConfig.ANGLE_DIFF_YLABEL),
            ]):
                ax_diff = difference_plot(ax, ylabel, Plot1DConfig.COLORS[-1])
                (diff,) = plot_variable(ax_diff, distance, diff_data, Plot1DConfig.COLORS[-1])
                if idx == 0:
                    set_twin_yaxis_limits(ax, ax_diff, data, diff_data, fraction)
                axs_diff.append(ax_diff)

            # axs_diff[0].yaxis.set_major_locator(
            #     ticker.MultipleLocator(FlowfieldConfig.VELOCITY_YTICKS_MAJOR / fraction)
            # )
            axs_diff[1].yaxis.set_major_locator(
                ticker.MultipleLocator(FlowfieldConfig.ANGLE_YTICKS_MAJOR / fraction)
            ) 
            axs_diff[1].yaxis.set_minor_locator(
                ticker.MultipleLocator(FlowfieldConfig.ANGLE_YTICKS_MINOR / fraction))

            axs_diff[1].set_ylim([y / fraction for y in FlowfieldConfig.ANGLE_YLIM])

        ax2.set_ylim(FlowfieldConfig.ANGLE_YLIM)

        handles = [*lines]
        labels = [*Plot1DConfig.LABELS[0:len(velocity)]]
        
        if axs_diff:
            handles.append(diff)
            labels.append(Plot1DConfig.LABELS[-1])
            axs_diff[0].yaxis.set_major_locator(ticker.FixedLocator([y / fraction for y in ax1.yaxis.get_majorticklocs()]))
            axs_diff[0].yaxis.set_minor_locator(ticker.FixedLocator([y / fraction for y in ax1.yaxis.get_minorticklocs()]))
        
        ax1.legend(
            handles,
            labels,
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            ncols=2,
            borderaxespad=0.0,
        )

        # Check and fix overlapping tick labels on y-axis:
        for ax in [ax1, ax2, *axs_diff]:
            if ax is None:
                continue
            style_1d_axis(ax, configuration.general.bool_flags["invertxaxis"])
            fix_overlapping_ticklabels(ax, axis="y")

        savefig(fig, filename)
