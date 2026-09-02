"""Backward-compatible plotting package for dfasttf.

This package replaces the former single-file ``dfasttf/batch/plotting.py``
module (1200+ lines mixing generic helpers, 2D maps, 1D ice plots and
cross-flow/tide plots). It is split by topic into:

- ``common``:  generic figure/axis helpers, colormaps, legends, savefig.
- ``configs``: styling dataclasses (labels, colors, tick spacing).
- ``maps2d``:  2D/ugrid map plotting (``Plot2D``, ``Ice2D``).
- ``ice1d``:   1D ice-scenario velocity/angle plotting (``Ice1D``).
- ``cross_flow``: cross-flow and tide plotting (``CrossFlow``).

Everything previously importable from ``dfasttf.batch.plotting`` is
re-exported here so existing call sites keep working unchanged, e.g.
``from dfasttf.batch.plotting import Plot2D, CrossFlow``.
"""
from dfasttf.batch.plotting.common import (
    CRS,
    FIGWIDTH,
    TEXTFONT,
    TEXTSIZE,
    XMAJORTICK,
    XMINORTICK,
    add_side_legend,
    construct_figure_filename,
    difference_plot,
    format_datetime_xaxis,
    format_datetime_yaxis,
    initialize_figure,
    initialize_subplot,
    invert_xaxis,
    make_marker_proxy,
    make_rkm_colormap,
    modify_axes,
    plot_chainage_markers,
    plot_variable,
    savefig,
    scatter_idx_points_on_timeseries,
    style_1d_axis,
)
from dfasttf.batch.plotting.configs import (
    CrossFlowConfig,
    DirectionalMaximaConfig,
    FlowfieldConfig,
    FroudeConfig,
    Plot1DConfig,
)
from dfasttf.batch.plotting.cross_flow import CrossFlow
from dfasttf.batch.plotting.ice1d import Ice1D
from dfasttf.batch.plotting.maps2d import Ice2D, Plot2D

__all__ = [
    "CRS",
    "FIGWIDTH",
    "TEXTFONT",
    "TEXTSIZE",
    "XMAJORTICK",
    "XMINORTICK",
    "add_side_legend",
    "construct_figure_filename",
    "CrossFlow",
    "CrossFlowConfig",
    "difference_plot",
    "DirectionalMaximaConfig",
    "FlowfieldConfig",
    "format_datetime_xaxis",
    "format_datetime_yaxis",
    "FroudeConfig",
    "Ice1D",
    "Ice2D",
    "initialize_figure",
    "initialize_subplot",
    "invert_xaxis",
    "make_marker_proxy",
    "make_rkm_colormap",
    "modify_axes",
    "Plot1DConfig",
    "Plot2D",
    "plot_chainage_markers",
    "plot_variable",
    "savefig",
    "scatter_idx_points_on_timeseries",
    "style_1d_axis",
]
