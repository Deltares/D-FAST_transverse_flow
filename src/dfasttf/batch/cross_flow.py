from pathlib import Path

import numpy as np
import pandas as pd

from dfasttf.batch import operations, plotting, support
from dfasttf.config import Config
from dfasttf.kernel import flow


def run(
    ucx: list[np.ndarray],
    ucy: list[np.ndarray],
    water_depth: list[np.ndarray],
    path_distances: np.ndarray,
    profile_angles: np.ndarray,
    rkm: np.ndarray,
    configuration: Config,
    figfile: Path,
    outputfiles: Path,
) -> None:
    """
    Input:
    ucx: (n,)
        x-component of flow velocity
    ucy: (n,)
        y-component of flow velocity
    water_depth: (n,)
        water depth at intersection points
    path_distances: (n,)
        cumulative distance between intersection points
    profile_angles: (n,)
        angle of profile line segments
    rkm: (n,)
        projected riverkm values
    """

    SHEET_LABELS = ("Reference", "WithIntervention", "Difference")
    CRITERIA: tuple[float, float] = (0.15, 0.3)  # criteria for transverse velocity

    rkm_km = rkm / 1000

    # Transverse velocity:
    COLUMN_LABELS = ("raai (rkm)", "dwarsstroomsnelheid (m/s)")
    transverse_velocity = []
    for x, y, wd in zip(ucx, ucy, water_depth):
        trans_flow = flow.trans_velocity(x, y, profile_angles)
        repr_trans_flow = flow.repr_trans_velocity(wd, trans_flow, path_distances, configuration.ship_params.depth)
        transverse_velocity.append(repr_trans_flow)

    data = [
        transverse_velocity[0],
        transverse_velocity[1] if len(transverse_velocity) > 1 else None,
        (
            transverse_velocity[1] - transverse_velocity[0]
            if len(transverse_velocity) > 1
            else None
        ),
    ]

    with pd.ExcelWriter(outputfiles[0]) as writer:
        for label, d in zip(SHEET_LABELS, data):
            if d is not None:
                support.to_excel(writer, COLUMN_LABELS, label, rkm_km, d)

    # Transverse discharge:
    COLUMN_LABELS = (
        "start (rkm)",
        "eind (rkm)",
        "dwarsstroomdebiet (m3/s)",
        "max. dwarsstroomsnelheid magnitude (m3/s)",
        "criterium (m/s)",
        "overschrijding (0=FALSE,1=TRUE)",
    )
    discharges, crit_values, xy_blocks = TransverseDischarge().compute(
        rkm,
        path_distances,
        transverse_velocity,
        configuration.ship_params.depth,
        configuration.ship_params.length,
        CRITERIA,
    )

    data = []
    for i, discharge in enumerate(discharges):
        data.append(prepare_data_for_excel(xy_blocks[i], discharge, crit_values[i]))

    with pd.ExcelWriter(outputfiles[1]) as writer:
        for label, d in zip(SHEET_LABELS, data):
            if d is not None:
                support.to_excel(writer, COLUMN_LABELS, label, *d)

    plotter = plotting.CrossFlow()
    plotter.create_figure(
        rkm,
        transverse_velocity,
        xy_blocks,
        crit_values,
        configuration.general.bool_flags["invertxaxis"],
        figfile,
    )


def prepare_data_for_excel(xy_block, discharge, crit_value):
    CONVERT_M_TO_KM = 1000
    x_start = [xy[0][0] / CONVERT_M_TO_KM for xy in xy_block]
    x_end = [xy[0][-1] / CONVERT_M_TO_KM for xy in xy_block]
    y_max = [max(abs(xy[1])) for xy in xy_block]
    exceedance = y_max > abs(crit_value)
    return (x_start, x_end, discharge, y_max, crit_value, exceedance)


class TransverseDischarge:
    def prepare_data(
        self,
        rkm: np.ndarray,
        path_distance: np.ndarray,
        transverse_velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare data by densifying, inserting array roots and subsequently splitting into blocks."""
        # because ship length is 0.5 m precision we first densify distance such that diff(distance) <= 0.5 m:

        path_distance_interp = operations.densify_array(path_distance, 0.5)

        transverse_velocity_interp = np.interp(
            path_distance_interp, path_distance, transverse_velocity
        )
        rkm_interp = np.interp(path_distance_interp, path_distance, rkm)

        rkm_app, transverse_velocity_app, path_distance_app = (
            operations.insert_array_roots(
                rkm_interp, transverse_velocity_interp, path_distance_interp
            )
        )
        rkm_split, transverse_velocity_split, path_distance_split = (
            operations.split_into_blocks(
                rkm_app, transverse_velocity_app, path_distance_app
            )
        )

        return rkm_split, path_distance_split, transverse_velocity_split

    def compute(
        self,
        rkm: np.ndarray,
        path_distances: np.ndarray,
        transverse_velocity: list[np.ndarray],
        ship_depth: float,
        ship_length: float,
        criteria: tuple[float, float],
    ):
        """Computes the transverse discharge from transverse velocity, ship depth and ship length"""
        discharges = []
        crit_values = []
        xy_segments = []

        for tv in transverse_velocity:
            rkm_split, path_distances_split, tv_split = self.prepare_data(
                rkm, path_distances, tv
            )
            discharge_case = []
            crit_case = []
            xy_segments_case = []

            for xi, prof_distance, yi in zip(rkm_split, path_distances_split, tv_split):
                if not np.any(yi):
                    continue

                max_integral, max_indices = operations.max_rolling_integral(
                    prof_distance, yi, ship_length
                )
                discharge = flow.trans_discharge(max_integral, ship_depth)
                discharge_case.append(discharge)

                start_idx, end_idx = max_indices[0], max_indices[-1] + 1
                # indices_case.append((start_idx, end_idx))

                xi_segment = xi[start_idx:end_idx]
                yi_segment = yi[start_idx:end_idx]
                xy_segments_case.append((xi_segment, yi_segment))

                crit_case.append(criteria[1] if discharge < 50.0 else criteria[0])

            discharges.append(np.array(discharge_case))
            crit_values.append(np.array(crit_case))
            xy_segments.append(xy_segments_case)

        return discharges, crit_values, xy_segments
