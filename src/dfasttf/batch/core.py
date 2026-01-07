import numpy as np
from pandas import DataFrame
from shapely import LineString
from tqdm import tqdm
from xugrid import UgridDataset

from dfasttf.batch import cross_flow, dflowfm, ice
from dfasttf.batch.dflowfm import (
    Variables,
    clip_simulation_data,
    load_simulation_data,
)
from dfasttf.batch.plotting import Plot2D, construct_figure_filename
from dfasttf.config import Config


def run_analysis(
    configuration: Config,
    section: str,
    variables: Variables,
    prof_line_df: DataFrame | None,
    riverkm: LineString | None,
):
    simulation_data = load_simulation_data(configuration, section)

    plot_actions = {
        "1D": lambda: run_1d_analysis(
            configuration, section, simulation_data, variables, prof_line_df, riverkm
        ),
        "2D": lambda: run_2d_analysis(
            configuration, section, simulation_data, variables, prof_line_df
        ),
    }
    plot_actions["both"] = lambda: (plot_actions["1D"], plot_actions["2D"])

    try:
        plot_actions[configuration.plotsettings.type]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown plot type {configuration.plotsettings.type}."
        ) from exc

def run_1d_analysis(
    configuration: Config,
    section: str,
    simulation_data: list[UgridDataset],
    variables: Variables,
    prof_line_df: DataFrame,
    riverkm: LineString,
):
    """Run 1D profile analysis and plotting."""
    riverkm_coords = np.array(riverkm.coords)
    padding = 1000  # metres

    for geom_idx, profile_line in enumerate(
        tqdm(prof_line_df.geometry, desc="geometry", position=0, leave=True)
    ):
        profile_coords = np.array(profile_line.coords)
        profile_index = str(prof_line_df.iloc[geom_idx].name)
        profile_data = {var: [] for var in variables._fields}
        bounds = profile_line.bounds

        for idx, _ in enumerate(
            tqdm(simulation_data, desc="simulation data", position=0, leave=True)
        ):
            data = clip_simulation_data(
                simulation_data[idx],
                [
                    bounds[0] - padding,
                    bounds[2] + padding,
                    bounds[1] - padding,
                    bounds[3] + padding,
                ],
            )

            sliced_ugrid = dflowfm.slice_ugrid(data, profile_coords, riverkm_coords)
            if sliced_ugrid is None:
                continue

            rkm, path_distances, isegment, iface = sliced_ugrid
            angles = np.array(prof_line_df["angle"].iloc[geom_idx][isegment])
            for var, name in variables._asdict().items():
                profile_data[var].append(dflowfm.get_profile_data(data, name, iface))

        if (
            sliced_ugrid is None
        ):  # profile line does not slice reference nor intervention simulation data
            continue

        save_1d_figures(
            configuration,
            section,
            profile_index,
            profile_data,
            angles,
            rkm,
            path_distances,
        )

        bedlevel = data[variables.bl].where(lambda x: x != 999)
        figfile = construct_figure_filename(
            configuration.plotsettings.options.figure_save_directory,
            f"profile{profile_index}_location",
            configuration.plotsettings.options.plot_extension,
        )
        Plot2D().plot_profile_line(profile_line, bedlevel, riverkm, figfile)


def save_1d_figures(
    configuration: Config,
    section: str,
    profile_index: str,
    profile_data: dict,
    angles: np.ndarray,
    rkm: np.ndarray,
    path_distances: np.ndarray,
):
    """Generate and save 1D figures and CSV files."""
    figdir = configuration.plotsettings.options.figure_save_directory
    figext = configuration.plotsettings.options.plot_extension
    outputdir = configuration.outputdir

    base = f"{section}_profile{profile_index}_velocity_angle"
    figfile = construct_figure_filename(figdir, base, figext)
    outputfile = (outputdir / base).with_suffix(".xlsx")
    ice.run_1d(
        profile_data["uc"],
        profile_data["ucx"],
        profile_data["ucy"],
        angles,
        rkm,
        configuration,
        figfile,
        outputfile,
    )

    outputfiles = []
    base = f"{section}_profile{profile_index}_transverse_velocity"
    outputfiles.append((outputdir / base).with_suffix(".xlsx"))

    base = f"{section}_profile{profile_index}_transverse_flow"
    figfile = construct_figure_filename(figdir, base, figext)
    outputfiles.append((outputdir / base).with_suffix(".xlsx"))

    cross_flow.run(
        profile_data["ucx"],
        profile_data["ucy"],
        profile_data["h"],
        path_distances,
        angles,
        rkm,
        configuration,
        figfile,
        outputfiles,
    )


def run_2d_analysis(
    configuration: Config,
    section: str,
    simulation_data: list[UgridDataset],
    variables: Variables,
    prof_line_df: DataFrame | None,
) -> None:
    """Run 2D Froude number analysis and plotting."""
    labels = ("reference", "intervention", "difference")

    waterupliftcorrection = configuration.general.bool_flags["waterupliftcorrection"]
    bedchangecorrection = configuration.general.bool_flags["bedchangecorrection"]

    suffix = ""
    if waterupliftcorrection:
        suffix = suffix + "_wateruplift"
    if bedchangecorrection:
        suffix = suffix + "_bedchange"

    padding = 1000  # metres
    profile_line = None
    if prof_line_df is None or getattr(prof_line_df, "empty", False):
        # derive bbox: prefer configured bbox, otherwise use dataset extent
        if configuration.general.bbox is not None:
            bbox = configuration.general.bbox
        else:
            ds0 = simulation_data[0]
            xs = ds0.ugrid.x.values
            ys = ds0.ugrid.y.values
            xmin, xmax = float(xs.min()), float(xs.max())
            ymin, ymax = float(ys.min()), float(ys.max())
            bbox = [xmin - padding, xmax + padding, ymin - padding, ymax + padding]

        water_depth = [clip_simulation_data(ds[variables.h], bbox) for ds in simulation_data]
        flow_velocity = [clip_simulation_data(ds[variables.uc], bbox) for ds in simulation_data]
        figfiles = [
            construct_figure_filename(
                configuration.plotsettings.options.figure_save_directory,
                f"{section}_{label}_Froude{suffix}",
                configuration.plotsettings.options.plot_extension,
            )
            for label in labels
        ]
        if water_depth and getattr(water_depth[0], "size", 0) != 0:
            ice.run_2d(water_depth, flow_velocity, configuration, profile_line, figfiles)
    else:
        for geom_idx, profile_line in enumerate(
            tqdm(prof_line_df.geometry, desc="geometry", position=0, leave=True)
        ):
            bounds = profile_line.bounds
            bbox = [
                bounds[0] - padding,
                bounds[2] + padding,
                bounds[1] - padding,
                bounds[3] + padding,
            ]

            water_depth = [
                clip_simulation_data(ds[variables.h], bbox) for ds in simulation_data
            ]
            flow_velocity = [
                clip_simulation_data(ds[variables.uc], bbox) for ds in simulation_data
            ]
            figfiles = [
                construct_figure_filename(
                    configuration.plotsettings.options.figure_save_directory,
                    f"{section}_{label}_profile{geom_idx}_Froude{suffix}",
                    configuration.plotsettings.options.plot_extension,
                )
                for label in labels
            ]

            if water_depth[0].size != 0:
                ice.run_2d(water_depth, flow_velocity, configuration, profile_line, figfiles)
