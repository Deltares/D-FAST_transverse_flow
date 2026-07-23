from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xugrid as xu
import warnings
from xarray import DataArray
from xugrid import UgridDataArray

from dfasttf.batch import geometry, plotting, support
from dfasttf.batch.dflowfm import clip_simulation_data, regrid_data_array
from dfasttf.config import Config
from dfasttf.kernel import froude


def run_1d(
    uc: list[np.ndarray],
    ucx: list[np.ndarray],
    ucy: list[np.ndarray],
    profile_angles: np.ndarray,
    rkm: np.ndarray,
    configuration: Config,
    figfile: Path,
    outputfile: Path,
) -> None:

    COLUMN_LABELS = (
        "afstand (rkm)",
        "stroomsnelheid (m/s)",
        "stromingshoek (graden)",
        "profiellijn (graden)",
        "stromingshoek t.o.v. profiellijn (graden)",
    )

    velocity_magnitude = []
    velocity_angle = []
    angle_diff = []
    rkm_km = rkm / 1000

    for m, x, y in zip(uc, ucx, ucy):
        velocity_magnitude.append(m)
        flow_angle = geometry.vector_angle(x, y)
        velocity_angle.append(flow_angle)

    # shortest angular difference
    angle_diff = [
        (angles - profile_angles + 180) % 360 - 180 for angles in velocity_angle
    ]
    angle_diff = [np.where(angles > 90, angles - 180, angles) for angles in angle_diff]
    angle_diff = [np.where(angles < -90, angles + 180, angles) for angles in angle_diff]

    labels = ["Reference", "WithIntervention", "Difference"]
    data = [
        (velocity_magnitude[0], velocity_angle[0], angle_diff[0]),
        (
            (velocity_magnitude[1], velocity_angle[1], angle_diff[1])
            if len(velocity_magnitude) > 1
            else None
        ),
        (
            (
                velocity_magnitude[1] - velocity_magnitude[0],
                velocity_angle[1] - velocity_angle[0],
                angle_diff[1] - angle_diff[0],
            )
            if len(velocity_magnitude) > 1
            else None
        ),
    ]

    with pd.ExcelWriter(outputfile) as writer:
        for label, d in zip(labels, data):
            if d is not None:
                support.to_excel(
                    writer,
                    COLUMN_LABELS,
                    label,
                    rkm_km,
                    d[0],
                    d[1],
                    profile_angles,
                    d[2],
                )

    plotting.Ice1D().create_figure(
        rkm, velocity_magnitude, angle_diff, configuration, figfile
    )


def run_2d(
    water_depth: list[DataArray],
    flow_velocity: list[DataArray],
    configuration: Config,
    profile_line: gpd.GeoSeries | None,
    filenames: list[Path],
) -> None:

    riverkm = configuration.general.riverkm

    froude_number = []
    for idx, (h, u) in enumerate(zip(water_depth, flow_velocity)):
        fr = froude.calculate_froude_number(h, u)
        fr = correct_model_results(fr, h, configuration)
        froude_number.append(fr)
        plotting.Ice2D().create_map(fr, riverkm, profile_line, filenames[idx])

    if len(froude_number) > 1:
        #grids_match = _map_grids(froude_number)
        # if not grids_match:
        #     warnings.warn(f"The intervention and reference grids don't match: {filenames}.\n"
        #                   "Note that the intervention grid is regridded to match the reference grid in the difference maps.", 
        #                   RuntimeWarning)
        froude_number[1] = regrid_data_array(froude_number)
        plotting.Ice2D().create_diff_map(
            froude_number[0], froude_number[1], riverkm, profile_line, filenames[2]
        )

def _map_grids(arrays: list[UgridDataArray]) -> bool | None:
    """Check if the grids are equal.
    0: mismatch, 1: match."""
    if len(arrays) < 2:
        return
    
    # TODO: replace this by face-node connectivity:
    nfaces1 = int(arrays[0].sizes.get("mesh2d_nFaces", 0))
    nfaces2 = int(arrays[1].sizes.get("mesh2d_nFaces", 0))
    return nfaces1 == nfaces2

def correct_model_results(
    froude_number: DataArray,
    water_depth: DataArray, 
    configuration: Config
) -> DataArray:
    water_uplift = configuration.general.bool_flags["waterupliftcorrection"]
    bed_change = configuration.general.bool_flags["bedchangecorrection"]
    tide_flag = configuration.general.bool_flags["tide"]
    bed_change_file = configuration.general.bedchangefile
    bbox = configuration.general.bbox

    if bed_change:
        if bed_change_file is None:
            raise ValueError("No bed change file specified in configuration.")
        bedlevel_change = get_bedlevel_change(bed_change_file, bbox)
        froude_number = froude.bed_change(froude_number, bedlevel_change, water_depth)
    if water_uplift:
        froude_number = froude.water_uplift(froude_number)
    if tide_flag and "time" in froude_number.dims:
        froude_number = get_percentile(froude_number, percentile=50)
    return froude_number

def get_percentile(
    da: DataArray,
    percentile: int = 80
) -> DataArray:
    """Takes the specified percentile of values over time.

    Parameters
    ----------
    da : DataArray
        Values over time, dimension 'time' must be present
    percentile : int
        Desired percentile (0-100), default is 80

    Returns
    ----------
    UgridDataArray
        k-th percentile for each cell
    """
    if not (0 <= percentile <= 100):
        raise ValueError("percentile must be between 0 and 100")

    # Convert to fraction for xarray quantile
    q = percentile / 100.0

    # Assume time dimension is named "time"
    if "time" not in da.dims:
        raise ValueError("Input variable must have a 'time' dimension")

    result = da.quantile(q=q, dim="time", keep_attrs=True)

    return result
                    
def get_bedlevel_change(file: Path, bbox: list) -> UgridDataArray:
    ds = xu.open_dataset(file)
    dfast_name = "avgdzb"
    data_vars = list(ds.data_vars)
    if dfast_name in data_vars:
        da = ds[dfast_name]
    elif len(data_vars) == 1:
        da = ds[data_vars[0]]
    else:
        raise IOError(f"NetCDF file must contain {dfast_name} or exactly one variable.")

    return clip_simulation_data(da, bbox)
