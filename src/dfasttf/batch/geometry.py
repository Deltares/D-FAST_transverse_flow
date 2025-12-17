from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np

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
