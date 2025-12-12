from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas
import shapely
from shapely import LineString


def vector_angle(u0: np.ndarray, v0: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(v0, u0))


def calculate_dx_dy(
    x_coords: np.ndarray, y_coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the differences in x and y coordinates between consecutive points.

    Parameters:
    x_coords (np.ndarray): Array of x coordinates.
    y_coords (np.ndarray): Array of y coordinates.

    Returns:
    tuple: Two arrays containing the differences in x and y coordinates.
    """
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)

    return dx, dy


def calculate_curve_distance(x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
    """Calculate the cumulative distance along a curve."""
    # Ensure the coordinates are of the same length
    if len(x_coords) != len(y_coords):
        raise ValueError("The arrays of x and y coordinates must have the same length.")

    dx, dy = calculate_dx_dy(x_coords, y_coords)

    # Calculate the distance between each pair of points
    distances = np.linalg.norm(np.column_stack((dx, dy)), axis=1)

    # Calculate the cumulative distance
    cumulative_distances = np.concatenate(([0], np.cumsum(distances)))

    return cumulative_distances


def find_distances_to_points(line_coords: np.ndarray, new_points: np.ndarray):
    """Vectorized function to find the distances along the line to new points
    array([], dtype=float64)
    Returns:
    profile_distances: distance of each point along the profile line, to start of line
    segment_indices: for each point, the index of the line segment the point is on"""
    cumulative_distances = calculate_curve_distance(
        line_coords[:, 0], line_coords[:, 1]
    )
    x1, y1 = line_coords[:-1].T
    x2, y2 = line_coords[1:].T

    # 1) Which segment’s bounding‐box each point falls into?
    mask = (
        (new_points[:, 0, None] >= np.minimum(x1, x2))
        & (new_points[:, 0, None] <= np.maximum(x1, x2))
        & (new_points[:, 1, None] >= np.minimum(y1, y2))
        & (new_points[:, 1, None] <= np.maximum(y1, y2))
    )

    # 2) Distance from each segment start to each point
    segment_distances = np.linalg.norm(new_points[:, None] - line_coords[:-1], axis=2)

    # 3) Add on the “already‐traveled” distance along the curve
    valid_distances = np.where(
        mask, cumulative_distances[:-1] + segment_distances, np.inf
    )

    # 4) For each point, pick the segment with minimal total distance
    segment_indices = np.argmin(valid_distances, axis=1)
    min_distances = np.min(valid_distances, axis=1)

    # 5) Clamp to the true number of segments
    n_segments = len(line_coords) - 1
    segment_indices = np.clip(segment_indices, 0, n_segments - 1)

    # 6) Mark points that never hit any segment
    no_hit = ~mask.any(axis=1)
    segment_indices[no_hit] = -1
    min_distances[no_hit] = np.nan
    profile_distances = min_distances

    return profile_distances, segment_indices


def read_xyc(filepath: Path, num_columns: int = 2) -> shapely.geometry.LineString:
    """
    Adapted from D-FAST BE: io.read_xyc()
    Read lines from a file.

    Arguments
    ---------
    filepath : Path
        Name of the file to be read.
    num_columns : int
        Number of columns to be read (2 or 3)

    Returns
    -------
    L : shapely.geometry.linestring.LineStringAdapter
        Line strings.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath.suffix.lower() == ".xyc":
        if num_columns == 3:
            column_names = ["Val", "X", "Y"]
        else:
            column_names = ["X", "Y"]
        point_coordinates = pandas.read_csv(
            filepath, names=column_names, skipinitialspace=True, sep=r"\s+"
        )
        num_points = len(point_coordinates.X)
        x = point_coordinates.X.to_numpy().reshape((num_points, 1))
        y = point_coordinates.Y.to_numpy().reshape((num_points, 1))
        if num_columns == 3:
            z = point_coordinates.Val.to_numpy().reshape((num_points, 1))
            coords = np.concatenate((x, y, z), axis=1)
        else:
            coords = np.concatenate((x, y), axis=1)
        line_string = shapely.geometry.LineString(coords)
    else:
        gdf = gpd.read_file(filepath)["geometry"]
        line_string = gdf[0]

    return line_string


def get_xy_km(km_file) -> shapely.geometry.linestring.LineString:
    """From D-FAST BE: io.get_xy_km()

    Returns
    -------
    xykm : shapely.geometry.linestring.LineStringAdapter
    """
    # get the chainage file
    # log_text("read_chainage", dict={"file": km_file})
    xy_km = read_xyc(km_file, num_columns=3)

    # make sure that chainage is increasing with node index
    if xy_km.coords[0][2] > xy_km.coords[1][2]:
        xy_km = LineString(xy_km.coords[::-1])

    return xy_km


def extract_coordinates(geometries: list) -> np.ndarray:
    """Extract coordinates from a list of Point and MultiPoint geometries."""
    coords = []
    for geom in geometries:
        if geom.geom_type == "Point":
            coords.append((geom.x, geom.y))
        elif geom.geom_type == "MultiPoint":
            coords.extend([(point.x, point.y) for point in geom.geoms])
    return np.array(coords)


@dataclass
class ProfileLines:
    """Class for handling profile lines"""

    filepath: Path
    dataframe: gpd.GeoDataFrame = field(init=False, default_factory=gpd.GeoDataFrame)

    def read_file(self) -> gpd.GeoDataFrame:
        """
        Read the profile lines from the file and return as a GeoDataFrame.
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        try:
            self.dataframe = gpd.read_file(self.filepath)
            first_col = self.dataframe.columns[0]
            self.dataframe.set_index(first_col, inplace=True)
            self.dataframe = self.get_exploded_df()
        except Exception as e:
            raise IOError(f"Error reading file {self.filepath}: {e}") from e
        return self.get_exploded_df()

    def get_angles(self):
        """
        Calculate angles for the geometries in the GeoDataFrame.
        """
        self.dataframe["angle"] = self.dataframe.geometry.apply(
            lambda geom: (
                calculate_angle(geom.coords) if hasattr(geom, "coords") else np.nan
            )
        )
        return self.dataframe["angle"]

    def get_exploded_df(self) -> gpd.GeoDataFrame:
        exploded_df = self.dataframe.explode()
        exploded_df.reset_index(drop=True, inplace=True)
        return exploded_df


# def merge_lines(lines: MultiLineString):
#     """"Merge individual line segments."""
#     return ops.linemerge(lines)


def calculate_angle(coords):
    vertices = np.array(coords)
    return np.degrees(np.arctan2(np.diff(vertices[:, 1]), np.diff(vertices[:, 0])))


def project_km_on_line(line_xy: np.ndarray, xykm_np: np.ndarray) -> np.ndarray:
    """
    From D-FAST BE: support.project_km_on_line

    Project chainage values from source line L1 onto another line L2.

    The chainage values are giving along a line L1 (xykm_np). For each node
    of the line L2 (line_xy) on which we would like to know the chainage, first
    the closest node (discrete set of nodes) on L1 is determined and
    subsequently the exact chainage isobtained by determining the closest point
    (continuous line) on L1 for which the chainage is determined using by means
    of interpolation.

    Arguments
    ---------
    line_xy : np.ndarray
        Array containing the x,y coordinates of a line.
    xykm_np : np.ndarray
        Array containing the x,y,chainage data.

    Results
    -------
    line_km : np.ndarray
        Array containing the chainage for every coordinate specified in line_xy.
    """
    # pre-allocate the array for the mapped chainage values
    line_km = np.zeros(line_xy.shape[0])

    # get an array with only the x,y coordinates of line L1
    xy_np = xykm_np[:, :2]
    last_xykm = xykm_np.shape[0] - 1

    # for each node rp on line L2 get the chainage ...
    for i, rp_np in enumerate(line_xy):
        # find the node on L1 closest to rp
        imin = np.argmin(((rp_np - xy_np) ** 2).sum(axis=1))
        p0 = xy_np[imin]

        # determine the distance between that node and rp
        dist2 = ((rp_np - p0) ** 2).sum()

        # chainage value of that node
        km = xykm_np[imin, 2]
        # print("chainage closest node: ", km)

        # if we didn't get the first node
        if imin > 0:
            # project rp onto the line segment before this node
            p1 = xy_np[imin - 1]
            alpha = (
                (p1[0] - p0[0]) * (rp_np[0] - p0[0])
                + (p1[1] - p0[1]) * (rp_np[1] - p0[1])
            ) / ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
            # if there is a closest point not coinciding with the nodes ...
            if alpha > 0 and alpha < 1:
                dist2link = (rp_np[0] - p0[0] - alpha * (p1[0] - p0[0])) ** 2 + (
                    rp_np[1] - p0[1] - alpha * (p1[1] - p0[1])
                ) ** 2
                # if it's actually closer than the node ...
                if dist2link < dist2:
                    # update the closest point information
                    dist2 = dist2link
                    km = xykm_np[imin, 2] + alpha * (
                        xykm_np[imin - 1, 2] - xykm_np[imin, 2]
                    )
                    # print("chainage of projection 1: ", km)

        # if we didn't get the last node
        if imin < last_xykm:
            # project rp onto the line segment after this node
            p1 = xy_np[imin + 1]
            alpha = (
                (p1[0] - p0[0]) * (rp_np[0] - p0[0])
                + (p1[1] - p0[1]) * (rp_np[1] - p0[1])
            ) / ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
            # if there is a closest point not coinciding with the nodes ...
            if alpha > 0 and alpha < 1:
                dist2link = (rp_np[0] - p0[0] - alpha * (p1[0] - p0[0])) ** 2 + (
                    rp_np[1] - p0[1] - alpha * (p1[1] - p0[1])
                ) ** 2
                # if it's actually closer than the previous value ...
                if dist2link < dist2:
                    # update the closest point information
                    dist2 = dist2link
                    km = xykm_np[imin, 2] + alpha * (
                        xykm_np[imin + 1, 2] - xykm_np[imin, 2]
                    )
                    # print("chainage of projection 2: ", km)

        # store the chainage value, loop ... and return
        line_km[i] = km
    return line_km
