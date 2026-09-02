"""1D ice-scenario flow velocity/angle plotting."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.axes import Axes

from dfasttf.batch.plotting.common import (
    difference_plot,
    initialize_figure,
    initialize_subplot,
    plot_variable,
    savefig,
    style_1d_axis,
)
from dfasttf.batch.plotting.configs import FlowfieldConfig, Plot1DConfig
from dfasttf.config import Config


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
