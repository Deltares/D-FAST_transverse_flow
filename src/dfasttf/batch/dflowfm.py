from pathlib import Path
from typing import NamedTuple

import numpy as np
import xugrid as xu
from pandas import DataFrame
import os
import warnings

from shapely.geometry import LineString
from xugrid import UgridDataArray, UgridDataset
from dfastbe.io.data_models import LineGeometry
from dfasttf.batch.filetype import detect_file_info, FileKind
from dfasttf.batch import geometry
from dfasttf.config import Config, get_output_files
from dfasttf.batch.operations import sort_a_by_b, group_duplicates

VARN_FACE_X_BND = "mesh2d_face_x_bnd"
VARN_FACE_Y_BND = "mesh2d_face_y_bnd"


class Variables(NamedTuple):
    """Class of relevant variables.
    h: water depth
    uc: flow velocity magnitude
    ucx: flow velocity, x-component
    ucy: flow velocity, y-componentn
    bl: bed level"""

    h: str
    uc: str
    ucx: str
    ucy: str
    bl: str

def _select_time_window(ds, start: str | None, stop: str | None):
    """
    Select a tide analysis window using TideStart and TideStop.

    Parameters
    ----------
    ds : UgridDataset
        Input dataset with time coordinate.
    start : str | None
        Start time in format 'YYYY-MM-DD HH:MM:SS'
    stop : str | None
        Stop time in format 'YYYY-MM-DD HH:MM:SS'

    Returns
    -------
    UgridDataset
        Dataset sliced to the selected time window.

    Raises
    ------
    RuntimeError
        If only one of start/stop is given, if parsing fails,
        or if the requested window is invalid.
    """
    if "time" not in ds.coords:
        return ds

    if start is None and stop is None:
        return ds

    if start is None or stop is None:
        raise RuntimeError(
            "Both TideStart and TideStop must be provided when defining a tide analysis window."
        )

    try:
        t_start = np.datetime64(start.replace(" ", "T"))
        t_stop = np.datetime64(stop.replace(" ", "T"))
    except Exception as exc:
        raise RuntimeError(
            "TideStart and TideStop must have format YYYY-MM-DD HH:MM:SS."
        ) from exc

    if t_stop <= t_start:
        raise RuntimeError("TideStop must be later than TideStart.")

    ds_sel = ds.sel(time=slice(t_start, t_stop))

    if "time" not in ds_sel.coords or ds_sel["time"].size < 2:
        raise RuntimeError(
            "Selected tide window does not contain enough timesteps."
        )

    return ds_sel



def check_time_coverage(
    ds: xu.UgridDataset,
    section: str,
    file: str,
    selected_window: bool = False,
) -> None:
    """
    Check whether the dataset (or selected tide window) covers at least one full tide cycle.

    Parameters
    ----------
    ds : UgridDataset
        Dataset or already-sliced tide window.
    section : str
        Config section name.
    file : str
        Input filename, used in messages.
    selected_window : bool, default False
        If False: treat insufficient duration as an input-file error.
        If True: treat insufficient duration as a user-window warning.

    Raises
    ------
    RuntimeError
        If the dataset has no usable time axis, or if the full dataset is too short.
    """
    if "time" not in ds.coords or ds["time"].size < 2:
        raise RuntimeError(
            f"[{section}] Tide analysis requires a time axis with at least 2 timesteps.\n"
            f"File: {file}"
        )

    t = ds["time"].values
    duration = np.datetime64(t[-1]) - np.datetime64(t[0])

    min_duration = np.timedelta64(24 * 60 + 50, "m")  # 24h50m
    duration_h = float(duration / np.timedelta64(1, "h"))

    if duration < min_duration:
        if selected_window:
            warnings.warn(
                f"[{section}] Selected tide window is shorter than one full tide cycle "
                f"(24h50m). Selected duration = {duration_h:.2f} h. "
                f"Tide results will be based only on the selected time window.\n"
                f"File: {file}",
                RuntimeWarning,
            )
        else:
            raise RuntimeError(
                f"[{section}] MAP file does not cover at least one full tide cycle.\n"
                f"Required: 24h50m, found: {duration_h:.2f} h\n"
                f"File: {file}"
            )


def check_time_interval(ds: UgridDataset, section: str, file: str) -> None:
    """
    Check whether the effective timestep interval in the dataset is suitable for tide analysis.

    Behaviour
    ---------
    - Uses the median timestep as the primary measure, to avoid false errors
      from one large initial gap in the output time series.
    - median dt > 2 hours  -> RuntimeError
    - median dt > 1 hour   -> warning
    - if a much larger gap exists, issue an additional warning
    """
    if "time" not in ds.coords or ds["time"].size < 2:
        return

    t = ds["time"].values
    dt = np.diff(t)

    # keep only positive finite intervals
    dt = dt[np.isfinite(dt) & (dt > np.timedelta64(0, "s"))]
    if dt.size == 0:
        return

    median_dt = np.median(dt)
    max_dt = dt.max()

    one_hour = np.timedelta64(1, "h")
    two_hours = np.timedelta64(2, "h")

    max_dt_s = float(max_dt / np.timedelta64(1, "s"))
    median_dt_s = float(median_dt / np.timedelta64(1, "s"))

    if median_dt > two_hours:
        raise RuntimeError(
            f"[{section}] MAP file time interval is too large for tide analysis.\n"
            f"Median dt = {median_dt_s:.0f}s, allowed <= 2h\n"
            f"File: {file}"
        )

    if median_dt > one_hour:
        warnings.warn(
            f"[{section}] MAP file time interval is relatively large for tide analysis.\n"
            f"Median dt = {median_dt_s:.0f}s (> 1h). Results may be less accurate.\n"
            f"File: {file}",
            RuntimeWarning,
        )

    # optional: warn if there are large irregular gaps
    if max_dt > 10 * median_dt:
        warnings.warn(
            f"[{section}] MAP file contains irregular output intervals.\n"
            f"Median dt = {median_dt_s:.0f}s, but maximum dt = {max_dt_s:.0f}s.\n"
            f"This may indicate an initial gap or non-uniform output scheduling.\n"
            f"File: {file}",
            RuntimeWarning,
        )



def check_ship_length_vs_grid_resolution(
    edge_coords: np.ndarray,
    ship_length: float,
    section: str,
    profile_index: str,
) -> None:
    """
    Check whether the intersected grid cells are sufficiently small relative to the ship length.

    Parameters
    ----------
    edge_coords : np.ndarray
        Array of shape (nfaces, nmax, 2) containing face corner coordinates.
        NaN rows are allowed for unused vertices.
    ship_length : float
        Representative ship length [m].
    section : str
        Config section name (e.g. C1).
    profile_index : str
        Profile identifier.

    Rules
    -----
    - If any intersected cell has a maximum edge length > ship_length:
      raise RuntimeError.
    - If any intersected cell has a maximum edge length > ship_length / 10:
      issue a warning.
    """
    max_edge_lengths = []

    for face_vertices in edge_coords:
        valid_mask = ~np.isnan(face_vertices[:, 0])
        vertices = face_vertices[valid_mask]

        if len(vertices) < 2:
            continue

        # closed polygon edges
        shifted = np.roll(vertices, -1, axis=0)
        edge_lengths = np.hypot(
            shifted[:, 0] - vertices[:, 0],
            shifted[:, 1] - vertices[:, 1],
        )

        max_edge_lengths.append(float(np.max(edge_lengths)))

    if len(max_edge_lengths) == 0:
        raise RuntimeError(
            f"[{section} profile {profile_index}] Could not determine edge lengths "
            f"for intersected grid cells."
        )

    max_edge_lengths = np.asarray(max_edge_lengths)
    max_edge = float(np.max(max_edge_lengths))

    if np.any(max_edge_lengths > ship_length):
        raise RuntimeError(
            f"[{section} profile {profile_index}] Grid resolution is too coarse "
            f"for ship length {ship_length:.2f} m. "
            f"At least one intersected cell has a maximum edge length of {max_edge:.2f} m, "
            f"which exceeds the ship length. "
            f"The transverse discharge analysis is not meaningful."
        )

    if np.any(max_edge_lengths > ship_length / 10.0):
        warnings.warn(
            f"[{section} profile {profile_index}] Grid resolution may be too coarse "
            f"for ship length {ship_length:.2f} m. "
            f"At least one intersected cell has a maximum edge length of {max_edge:.2f} m, "
            f"which exceeds ship_length / 10 = {ship_length / 10.0:.2f} m.",
            RuntimeWarning,
        )

def regrid_data_array(arrays: list[UgridDataArray]) -> UgridDataArray:
    """Check whether the grids on two datasets match and return a regridded source dataset.
    The 2nd dataset is regridded to match the 1st dataset.
    TODO: eventually replace with _map_grids from dflowfm.batch.AnalyserDflowfm() 
    
    Parameters
    ----------
    datasets: list of UgridDatasets (shape (2,))

    Returns
    -------
    UgridDataset: regridded dataset (2nd dataset in datasets)

    """
    if len(arrays)<2:
        raise ValueError("List contains only one UgridDataArray, but regridding requires two UgridDataArrays")

    da1 = arrays[0]
    da2 = arrays[1]
    regridder = xu.OverlapRegridder(source=da2, target=da1)
    da2_regridded = regridder.regrid(da2)
    return da2_regridded


def load_simulation_data(configuration: Config, section: str):
    """
    Content-based loader:

    Returns
    -------
    simulation_data_snapshot : list[UgridDataset]
        Always returned. For MAP input: last timestep is selected.
    simulation_data_tide : list[UgridDataset] | None
        Only returned when Tide=True AND input is MAP (has time). Otherwise None.

    Behavior
    --------
    - MAP:
        original analysis = isel(time=-1)
        tide = selected time window (TideStart/TideStop), or full ds if not specifie

    - FOU (no time but supports original analysis):
        snapshot = ds
        tide = None (warning if Tide=True)
    - INVALID:
        when the required variables are not present hard fail with clear message
    """
    output_files = get_output_files(
        configuration.config, configuration.configdir, section
    )

    tide_flag = configuration.general.bool_flags.get("tide", False)
    
    tide_start = getattr(configuration.general, "tide_start", None)
    tide_stop = getattr(configuration.general, "tide_stop", None)

    if tide_flag and tide_start is None and tide_stop is None:
        warnings.warn(
            "Tide=True but TideStart and TideStop are not set. "
            "Using the full available time series.",
            RuntimeWarning,
        )

    if tide_flag and ((tide_start is None) != (tide_stop is None)):
        raise RuntimeError(
            "Both TideStart and TideStop must be specified together."
        )

    snapshot_datasets: list[UgridDataset] = []
    tide_datasets: list[UgridDataset] = []

    for file in output_files:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"[{section}] File not found: {file}")

        ds = xu.open_dataset(file, chunks={"time": 1, "x": 100, "y": 100})

        if configuration.general.bbox is not None:
            ds = clip_simulation_data(ds, configuration.general.bbox)

        # Optional: skip empty clipped datasets (prevents downstream issues)
        nfaces = int(ds.sizes.get("mesh2d_nFaces", 0))
        if nfaces == 0:
            warnings.warn(
                f"[{section}] Dataset became empty after clipping (mesh2d_nFaces=0). "
                f"File: {file}. Skipping this dataset.",
                RuntimeWarning,
            )
            continue

        info = detect_file_info(ds)

        if info.kind == FileKind.INVALID:
            raise RuntimeError(
                f"[{section}] Input has no time dimension and misses required variables for original analysis.\n"
                f"File: {file}\n"
                f"Missing: {info.missing}\n"
                f"Available vars (sample): {info.vars_sample}"
            )

        # # MAP-specific tide checks
        # if tide_flag and info.kind == FileKind.MAP:
        #     check_time_coverage(ds, section, file, selected_window=False)

        # --- SNAPSHOT dataset ---
        if info.kind == FileKind.MAP:
            ds_snap = ds.isel(time=-1)
        else:
            ds_snap = ds

        snapshot_datasets.append(extract_variables(ds_snap))

        # --- TIDE dataset (only for MAP) ---

        if tide_flag:
            if info.kind == FileKind.MAP:
                ds_tide = _select_time_window(ds, tide_start, tide_stop)

                # selected-window checks
                check_time_coverage(ds_tide, section, file, selected_window=True)
                check_time_interval(ds_tide, section, file)
                
                tide_datasets.append(extract_variables(ds_tide))


            else:
                warnings.warn(
                    f"[{section}] Tide=True but file has no time dimension (FOU). Tide analysis skipped.\n"
                    f"File: {file}",
                    RuntimeWarning,
                )

    simulation_data_tide = tide_datasets if tide_datasets else None
    return snapshot_datasets, simulation_data_tide


def clip_simulation_data(
    data: UgridDataArray | UgridDataset, bbox: list
) -> UgridDataArray | UgridDataset:
    # TODO: implement better bbox data structure based on keywords
    """Clips simulation data based on bounding box [xmin, xmax, ymin, ymax]"""
    return data.ugrid.sel(x=slice(bbox[0], bbox[1]), y=slice(bbox[2], bbox[3]))


def extract_variables(ds: xu.UgridDataset) -> xu.UgridDataset:
    """Extract and standardize variable names from a NetCDF dataset using lazy loading and Dask."""

    bl = find_variable(ds, "altitude")
    wl = find_variable(ds, "sea_surface_height")
    uc = find_variable(ds, "sea_water_speed")
    ucx = find_variable(ds, "sea_water_x_velocity")
    ucy = find_variable(ds, "sea_water_y_velocity")

    bl_da = ds[bl]
    if "time" in bl_da.dims:
        bl_da = bl_da.isel(time=-1)

    bl_face = bl_da.ugrid.to_face().mean("nmax")

    ds = ds.assign(
        mesh2d_waterdepth=ds[wl] - bl_face,
        mesh2d_ucmag=ds[uc],
        mesh2d_ucx=ds[ucx],
        mesh2d_ucy=ds[ucy],
        mesh2d_bl=bl_face,
    )

    return ds


def find_variable(data: UgridDataset, standard_name: str) -> str:
    """Finds a variable in a dataset by its 'standard_name' attribute."""
    selected_var = next(
        (
            var
            for var in data.data_vars
            if data[var].attrs.get("standard_name") == standard_name
        ),
        None,
    )
    if selected_var is None:
        raise IOError(f"No variable found with standard_name '{standard_name}'")
    return selected_var


def get_profile_data(
    profile_dataset: xu.UgridDataset,
    variable_name: str,
    face_idx,
    time_index_from_last: int | None = 0,
) -> np.ndarray:
    """
    Extract values of a face-based variable along a profile.

    This function assumes the variable is stored on the mesh face dimension
    ``mesh2d_nFaces`` and optionally on a time dimension.

    Parameters
    ----------
    profile_dataset : xu.UgridDataset
        Dataset containing the variable.
    variable_name : str
        Name of the variable to extract.
    face_idx : array-like
        Indices of the faces that intersect the profile.
    time_index_from_last : int | None, default 0
        Select which timestep to extract when the variable has a time dimension.

        - ``0``  -> last timestep
        - ``1``  -> one before last
        - ``None`` -> full time series

    Returns
    -------
    np.ndarray
        - shape ``(n,)`` for snapshot extraction
        - shape ``(nt, n)`` for time-series extraction

    Raises
    ------
    ValueError
        If the variable is not stored on ``mesh2d_nFaces``, or if a full time
        series is requested for a variable without a time dimension.

    Notes
    -----
    This function is used for both:
    - snapshot analysis (single timestep)
    - tide analysis (full time series)

    It explicitly indexes the face dimension, so it works correctly for both
    variables with dims ``(mesh2d_nFaces,)`` and ``(time, mesh2d_nFaces)``.
    """
    da = profile_dataset[variable_name]
    data = da.data

    if "mesh2d_nFaces" not in da.dims:
        raise ValueError(
            f"Variable '{variable_name}' is not face-based. dims={da.dims}"
        )

    face_axis = da.dims.index("mesh2d_nFaces")


    if "time" in da.dims:
        if time_index_from_last is None:
            # return full time series for requested faces
            sel = np.take(data, face_idx, axis=face_axis)
        else:
            t = -1 - int(time_index_from_last)
    
            # Select a single timestep first
            data_at_time = data[t]
    
            # After removing the time dimension, the face axis shifts left by one
            face_axis_after_time_selection = face_axis - 1
    
            sel = np.take(data_at_time, face_idx, axis=face_axis_after_time_selection)
    
    else:
        if time_index_from_last is None:
            raise ValueError(
                f"Requested time series for '{variable_name}', but dataset has no time dimension."
            )
    
        # Snapshot variable: just select requested faces
        sel = np.take(data, face_idx, axis=face_axis)


    return sel.compute() if hasattr(sel, "compute") else np.asarray(sel)


def slice_ugrid(
    simulation_data: UgridDataset,
    profile_coords: np.ndarray,
    riverkm_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    edge_coords = extract_edge_coords(simulation_data, VARN_FACE_X_BND, VARN_FACE_Y_BND)
    sliced = slice_mesh_with_polyline(edge_coords, profile_coords, riverkm_coords)
    if sliced is None:
        return None
    rkm, path_distances, segment_idx, face_idx, intersects_ordered = sliced
    return rkm, path_distances, segment_idx, face_idx, intersects_ordered


def read_profile_lines(profiles_file: Path) -> DataFrame:
    profile_lines = geometry.ProfileLines(profiles_file)
    prof_line_df = profile_lines.read_file()
    prof_line_df["angle"] = profile_lines.get_angles()
    return prof_line_df


def intersect_linestring(
    simulation_data: UgridDataset, profile: LineString
) -> UgridDataset:
    """Returns only the data on faces intersected by the profile line"""
    return simulation_data.ugrid.intersect_linestring(profile)


def extract_edge_coords(
    profile_data: UgridDataset, varn_face_x_bnd: str, varn_face_y_bnd: str
) -> np.ndarray:
    x_bnd = profile_data[varn_face_x_bnd].values
    y_bnd = profile_data[varn_face_y_bnd].values
    return np.stack((x_bnd, y_bnd), axis=-1)


def slice_mesh_with_polyline(
    edge_coords: np.ndarray, profile_coords: np.ndarray, xykm_coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Slices mesh edges with a profile line and returns for each intersection point:
    pkm: projected value of xykm, found by interpolation
    path_distances: distance along path formed by intersection points
    segment_idx: index of segment of profile line
    face_idx: index of mesh face"""
    intersects, face_indices = find_intersects(edge_coords, profile_coords)

    if len(intersects) == 0:
        print(
            "No intersects found between profile line(s) and simulation data. "
            "Expand the bounding box, or change the profile line(s)"
        )
        return None

    profile_distances, segment_indices = calculate_intersect_distance(
        profile_coords, intersects
    )
    pkm, intersects_ordered, segment_idx, face_idx = _order_intersection_points(
        intersects, profile_distances, segment_indices, face_indices, xykm_coords
    )

    path_distances = geometry.calculate_curve_distance(
        intersects_ordered[:, 0], intersects_ordered[:, 1]
    )
    return pkm, path_distances, segment_idx, face_idx, intersects_ordered


def find_intersects(
    edge_coords: np.ndarray, line_coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Find intersection points between mesh edges and a line.

    Parameters:
    - edge_coords: (nfaces, nmax, 2), with NaNs for unused vertices
    - line_coords: (N, 2)

    Returns:
    - intersects: (M, 2) array of intersection points
    - face_idx: (M,) array of face indices
    """

    intersects = []
    face_idx = []
    nfaces, nmax, _ = edge_coords.shape
    b = LineString(line_coords)

    for i in range(nfaces):
        # Extract non-NaN vertices for this face
        face_vertices = edge_coords[i]
        valid_mask = ~np.isnan(face_vertices[:, 0])
        valid_vertices = face_vertices[valid_mask]

        n_valid = valid_vertices.shape[0]
        if n_valid < 2:
            continue  # skip degenerate faces

        # Loop through valid edges
        for j in range(n_valid):
            a1 = valid_vertices[j]
            a2 = valid_vertices[(j + 1) % n_valid]  # wrap around
            a = LineString([a1, a2])

            try:
                intersect = a.intersection(b)
                if not intersect.is_empty:
                    coords = extract_coordinates([intersect])
                    if coords.size > 0:
                        intersects.extend(coords)
                        face_idx.extend([i] * len(coords))
            except:
                pass

    intersects = np.array(intersects)
    face_idx = np.asarray(face_idx)

    # Optional for debugging:
    # pd.DataFrame(np.column_stack((intersects[:,0], intersects[:,1], face_idx))).to_csv('intersects.csv')
    return intersects, face_idx


def extract_coordinates(geom_list) -> np.ndarray:
    coords = []
    for g in geom_list:
        if g.geom_type == "Point":
            coords.append([g.x, g.y])
        elif g.geom_type == "MultiPoint":
            coords.extend([[pt.x, pt.y] for pt in g.geoms])
        elif g.geom_type == "LineString":
            mid_idx = len(g.coords) // 2
            coords.append(list(g.coords[mid_idx]))
        elif g.geom_type == "GeometryCollection":
            for subg in g.geoms:
                coords.extend(extract_coordinates([subg]))
    return np.array(coords)


def calculate_intersect_distance(
    line_coords: np.ndarray, intersects: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    profile_distances: distance of intersection points along line.
    segment_idx: indices of the line segments where the intersection occurs (N,1)"""
    profile_distances, segment_idx = geometry.find_distances_to_points(
        line_coords, intersects
    )
    return profile_distances, segment_idx


def _order_intersection_points(
    intersects: np.ndarray,
    profile_distances: np.ndarray,
    segment_idx: np.ndarray,
    face_idx: np.ndarray,
    river_km: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Correctly orders the intersection points between a UGRID mesh and profile line.

    Parameters:
    intersects: Intersection points.
    profile_distances: Distances along the profile line.
    segment_idx: Segment indices of the profile line.
    face_idx: Face indices of mesh.
    river_km: Mx3 array with x, y, and chainage values (river km)

    Returns:
    tuple[np.ndarray, np.ndarray, np.ndarray]: Grouped rkm, segment indices, and face indices.
    """

    # 1. Sort along profile line
    sorted_data = [
        sort_a_by_b(a, profile_distances) for a in [intersects, segment_idx, face_idx]
    ]
    intersects, segment_idx, face_idx = sorted_data

    # 2. Group face indices
    face_idx, group_idx = group_duplicates(face_idx)
    segment_idx = segment_idx[group_idx]
    intersects = intersects[group_idx]

    # 3. Convert to rkm, in metres
    rkm = convert_to_rkm(intersects, river_km, 1000)

    # 4. Ensure the overall direction is downstream (so the first rkm < last rkm)
    if rkm[0] > rkm[-1]:
        rkm = rkm[::-1]
        intersects = intersects[::-1]
        segment_idx = segment_idx[::-1]
        face_idx = face_idx[::-1]

    # 5. strictly increasing sequence of rkm
    mask = np.empty_like(rkm, dtype=bool)
    mask[0] = True
    last_r = rkm[0]

    for i in range(1, len(rkm)):
        if rkm[i] >= last_r:
            mask[i] = True
            last_r = rkm[i]
        else:
            mask[i] = False

    rkm_ordered = rkm[mask]
    intersects_ordered = intersects[mask]
    segment_idx_ordered = segment_idx[mask]
    face_idx_ordered = face_idx[mask]

    # now this should be guaranteed non‐decreasing (strictly increasing)
    assert np.all(np.diff(rkm_ordered) >= 0)

    return rkm_ordered, intersects_ordered, segment_idx_ordered, face_idx_ordered


def convert_to_rkm(intersects, river_km, conversion_factor=1):
    """Converts an array of points to the corresponding rkm values

    Parameters:    intersects: Nx2 array of intersection points
    river_km: Mx3 array with x, y, and chainage values (river km)
    conversion_factor: optional, to convert km to another unit (default = 1)"""
    intersects_line = LineGeometry(intersects)
    rkm = intersects_line.intersect_with_line(river_km) * conversion_factor
    return rkm

