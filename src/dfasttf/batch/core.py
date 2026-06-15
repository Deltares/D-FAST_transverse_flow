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
    check_ship_length_vs_profile_resolution,
)
from dfasttf.batch.plotting import Plot2D, construct_figure_filename
from dfasttf.config import Config


# def build_profile_points_xy(
#     profile_line: LineString,
#     path_distances: np.ndarray,
# ) -> np.ndarray:
#     """
#     Convert cumulative distances along the profile line to XY coordinates.

#     Returns
#     -------
#     np.ndarray
#         Array of shape (n, 2) with XY coordinates of the profile sample points.
#     """
#     d = np.asarray(path_distances, dtype=float)
#     d = np.clip(d, 0.0, float(profile_line.length))

#     return np.array(
#         [profile_line.interpolate(float(di)).coords[0][:2] for di in d],
#         dtype=float,
#     )


# def get_axis_point_xy(
#     profile_line: LineString,
#     riverkm: LineString,
# ) -> np.ndarray:
#     """
#     Determine the river-axis point associated with a profile line.

#     Preferred behaviour:
#     - use the actual intersection point between profile_line and riverkm
#     - if multiple points exist, take the one closest to the profile midpoint
#     - if no intersection exists, fall back to the nearest point on riverkm
#       to the profile midpoint

#     Returns
#     -------
#     np.ndarray
#         XY coordinate of the axis point, shape (2,)
#     """
#     inter = profile_line.intersection(riverkm)
#     mid = profile_line.interpolate(profile_line.length / 2.0)

#     if inter.is_empty:
#         s = riverkm.project(mid)
#         p = riverkm.interpolate(s)
#         return np.array(p.coords[0][:2], dtype=float)

#     if inter.geom_type == "Point":
#         return np.array(inter.coords[0][:2], dtype=float)

#     if inter.geom_type == "MultiPoint":
#         pts = list(inter.geoms)
#         p = min(pts, key=lambda pt: pt.distance(mid))
#         return np.array(p.coords[0][:2], dtype=float)

#     # fallback for uncommon intersection geometries
#     s = riverkm.project(mid)
#     p = riverkm.interpolate(s)
#     return np.array(p.coords[0][:2], dtype=float)


def run_analysis(
    configuration: Config,
    section: str,
    variables: Variables,
    prof_line_df: DataFrame | None,
    riverkm: LineString | None,
):
    # Loader returns both snapshot and tide datasets
    simulation_data, simulation_data_tide = load_simulation_data(configuration, section)

    plot_actions = {
        "1D": lambda: run_1d_analysis(
            configuration,
            section,
            simulation_data,
            simulation_data_tide,
            variables,
            prof_line_df,
            riverkm,
        ),
        "2D": lambda: run_2d_analysis(
            configuration,
            section,
            simulation_data,
            variables,
            prof_line_df,
        ),
    }

    plot_actions["both"] = lambda: (plot_actions["1D"](), plot_actions["2D"]())

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
    simulation_data_tide: list[UgridDataset] | None,  # Implemented for tide analysis
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
        tide = configuration.general.bool_flags.get("tide", False)
        profile_coords = np.array(profile_line.coords)
        profile_index = str(prof_line_df.iloc[geom_idx].name)
        profile_data = {var: [] for var in variables._fields}

        profile_data_tide = None
        if tide and simulation_data_tide is not None:
            profile_data_tide = {var: [] for var in variables._fields}
            profile_data_tide["time"] = []

        bounds = profile_line.bounds
        has_slice = False

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

            has_slice = True

            rkm, path_distances, isegment, iface = sliced_ugrid
            angles = np.array(prof_line_df["angle"].iloc[geom_idx][isegment])

            #profile_points_xy = build_profile_points_xy(profile_line, path_distances)
            #axis_point_xy = get_axis_point_xy(profile_line, riverkm)

            for var, name in variables._asdict().items():
                profile_data[var].append(
                    dflowfm.get_profile_data(data, name, iface, time_index_from_last=0)
                )

            # tide extraction (only if enabled in config and map files are provided)
            if tide and simulation_data_tide is not None:
                data_tide = clip_simulation_data(
                    simulation_data_tide[idx],
                    [
                        bounds[0] - padding,
                        bounds[2] + padding,
                        bounds[1] - padding,
                        bounds[3] + padding,
                    ],
                )

                if "time" in data_tide.coords:
                    profile_data_tide["time"].append(
                        np.asarray(data_tide["time"].values)
                    )
                else:
                    profile_data_tide["time"].append(None)
                # ------------------------------------------------------

                TIME_SERIES_VARS = {
                    "ucx",
                    "ucy",
                    "h",
                    "uc",
                }

                for var, name in variables._asdict().items():
                    ti = None if var in TIME_SERIES_VARS else 0
                    profile_data_tide[var].append(
                        dflowfm.get_profile_data(
                            data_tide, name, iface, time_index_from_last=ti
                        )
                    )

            checked_profile_resolution = False
            if not checked_profile_resolution:
                check_ship_length_vs_profile_resolution(
                    path_distances,
                    configuration.ship_params.length,
                    section,
                    profile_index,
                )
                checked_profile_resolution = True

        if not has_slice:
            continue  # profile line does not slice reference nor intervention simulation data

        save_1d_figures(
            configuration,
            section,
            profile_index,
            profile_data,
            profile_data_tide,
            angles,
            rkm,
            path_distances,
            #profile_points_xy,
            #axis_point_xy,
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
    profile_data_tide: dict | None,
    angles: np.ndarray,
    rkm: np.ndarray,
    path_distances: np.ndarray,
    #profile_points_xy: np.ndarray,
    #axis_point_xy: np.ndarray,
):
    """Generate and save 1D figures and CSV files."""
    figdir = configuration.plotsettings.options.figure_save_directory
    figext = configuration.plotsettings.options.plot_extension
    outputdir = configuration.outputdir

    # ---- Ice (old) ----
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

    # ---- Snapshot cross-flow outputs ----
    outputfiles = []
    base = f"{section}_profile{profile_index}_transverse_velocity"
    outputfiles.append((outputdir / base).with_suffix(".xlsx"))

    base = f"{section}_profile{profile_index}_transverse_flow"
    figfile_cross = construct_figure_filename(figdir, base, figext)
    outputfiles.append((outputdir / base).with_suffix(".xlsx"))

    # ---- Tide analysis inputs (optional) ----
    tide_flag = configuration.general.bool_flags.get("tide", False)
    tide_inputs = None

    if (
        tide_flag
        and profile_data_tide is not None
        and len(profile_data_tide["ucx"]) > 0
    ):
        base = f"{section}_profile{profile_index}_tide_transverse_velocity"
        figfile_tide_vel = construct_figure_filename(figdir, base, figext)

        base = f"{section}_profile{profile_index}_tide_transverse_flow_maxQ"
        figfile_tide_q = construct_figure_filename(figdir, base, figext)

        base = f"{section}_profile{profile_index}_tide_alongstream_velocity_time"
        figfile_tide_upar = construct_figure_filename(figdir, base, figext)

        base = f"{section}_profile{profile_index}_tide_max_transverse"
        figfile_tide_max_tv = construct_figure_filename(figdir, base, figext)

        tide_inputs = cross_flow.TideInputs(
            ucx=profile_data_tide["ucx"],
            ucy=profile_data_tide["ucy"],
            h=profile_data_tide["h"],
            time_list=profile_data_tide["time"],
            fig_vel=figfile_tide_vel,
            fig_qmax=figfile_tide_q,
            fig_upar=figfile_tide_upar,
            fig_max_tv=figfile_tide_max_tv,
        )

    # ---- Single call: snapshot always, tide optional inside run() ----
    cross_flow.run(
        profile_data["ucx"],
        profile_data["ucy"],
        profile_data["h"],
        path_distances,
        angles,
        rkm,
        configuration,
        figfile_cross,
        outputfiles,
        #profile_points_xy,
        #axis_point_xy,
        tide=tide_inputs,
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

        water_depth = [
            clip_simulation_data(ds[variables.h], bbox) for ds in simulation_data
        ]
        flow_velocity = [
            clip_simulation_data(ds[variables.uc], bbox) for ds in simulation_data
        ]
        figfiles = [
            construct_figure_filename(
                configuration.plotsettings.options.figure_save_directory,
                f"{section}_{label}_Froude{suffix}",
                configuration.plotsettings.options.plot_extension,
            )
            for label in labels
        ]
        if water_depth and getattr(water_depth[0], "size", 0) != 0:
            ice.run_2d(
                water_depth, flow_velocity, configuration, profile_line, figfiles
            )
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
                ice.run_2d(
                    water_depth, flow_velocity, configuration, profile_line, figfiles
                )
