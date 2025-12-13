from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import xugrid as xu
from pandas import DataFrame

# import pandas as pd
from shapely import LineString
from xugrid import UgridDataArray, UgridDataset

from dfasttf.batch import geometry
from dfasttf.config import Config, get_output_files

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


def load_simulation_data(configuration: Config, section: str) -> list[UgridDataset]:
    """Load and preprocess simulation datasets."""
    datasets = []
    output_files = get_output_files(
        configuration.config, configuration.configdir, section
    )
    for file in output_files:
        ds = xu.open_dataset(file, chunks={"time": 1, "x": 100, "y": 100})

        if configuration.general.bbox is not None:
            ds = clip_simulation_data(ds, configuration.general.bbox)
        ds = extract_variables(ds)
        datasets.append(ds)
    return datasets


def clip_simulation_data(
    data: UgridDataArray | UgridDataset, bbox: list
) -> UgridDataArray | UgridDataset:
    # TODO: implement better bbox data structure based on keywords
    """Clips simulation data based on bounding box [xmin, xmax, ymin, ymax]"""
    return data.ugrid.sel(x=slice(bbox[0], bbox[1]), y=slice(bbox[2], bbox[3]))


def extract_variables(ds: xu.UgridDataset) -> xu.UgridDataset:
    """Extract and standardize variable names from a NetCDF dataset using lazy loading and Dask."""

    if "time" in ds.coords:
        ds = ds.isel(time=-1)
    else:
        bl = find_variable(ds, "altitude")
        wl = find_variable(ds, "sea_surface_height")
        uc = find_variable(ds, "sea_water_speed")
        ucx = find_variable(ds, "sea_water_x_velocity")
        ucy = find_variable(ds, "sea_water_y_velocity")

        ds[bl] = ds[bl].ugrid.to_face().mean("nmax")  # bed elevation on nodes to faces

        ds = ds.assign(
            mesh2d_waterdepth=ds[wl] - ds[bl],
            mesh2d_ucmag=ds[uc],
            mesh2d_ucx=ds[ucx],
            mesh2d_ucy=ds[ucy],
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
    profile_dataset: UgridDataset, variable_name: str, face_idx
) -> dict:
    profile_data = profile_dataset[variable_name].data[face_idx]
    return profile_data


def slice_ugrid(
    simulation_data: UgridDataset,
    profile_coords: np.ndarray,
    riverkm_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    edge_coords = extract_edge_coords(simulation_data, VARN_FACE_X_BND, VARN_FACE_Y_BND)
    sliced = slice_mesh_with_polyline(edge_coords, profile_coords, riverkm_coords)
    if sliced is None:
        return None
    rkm, path_distances, segment_idx, face_idx = sliced
    return rkm, path_distances, segment_idx, face_idx


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
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
    return pkm, path_distances, segment_idx, face_idx


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


def extract_coordinates(geom_list):
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
    river_km: x,y coordinates of river kilometers (rkm)

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


def sort_a_by_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sorts the array `a` by the argsort of `b`.

    Parameters:
    a (np.ndarray): Array to be sorted.
    b (np.ndarray): Array to sort by.

    Returns:
    np.ndarray: Sorted array `a`.
    """
    sort_idx = np.argsort(b)
    return (
        np.take_along_axis(a, sort_idx[:, np.newaxis], axis=0)
        if a.ndim > 1
        else a[sort_idx]
    )


def group_duplicates(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Groups duplicates in an array, preserving insertion order of first occurrences"""
    groups = OrderedDict()
    for idx, val in enumerate(array):
        if val not in groups:
            groups[val] = []
        groups[val].append(idx)

    group_indices = np.array([idx for indices in groups.values() for idx in indices])
    grouped_array = array[group_indices]
    return grouped_array, group_indices


def convert_to_rkm(intersects, river_km, conversion_factor=1):
    """Converts an array of points to the corresponding rkm values

    Parameters:
    intersects: intersection points
    river_km: chainage values
    conversion_factor: optional, to convert km to another unit (default = 1)"""
    rkm = geometry.project_km_on_line(intersects, river_km) * conversion_factor
    return rkm
