from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from dfasttf.batch import plotting, support
from dfasttf.config import Config
from dfasttf.kernel import flow


@dataclass(frozen=True)
class TideInputs:
    ucx: list[np.ndarray]  # each (nt, n)
    ucy: list[np.ndarray]  # each (nt, n)
    h: list[np.ndarray]  # each (nt, n)
    time_list: list[np.ndarray | None]  # per case (nt,)
    fig_vel: Path
    fig_qmax: Path
    fig_upar: Path
    fig_max_tv: Path
    fig_directional_bankward: Path
    fig_directional_riverward: Path


def _validate_tide_inputs(
    tide: TideInputs,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], int, int, int]:
    """
    Validate tide input arrays and return normalized arrays.

    Returns
    -------
    tide_ucx, tide_ucy, tide_h, n_cases, nt, n
    """
    n_cases = len(tide.ucx)
    if n_cases == 0:
        return [], [], [], 0, 0, 0

    tide_ucx = [np.asarray(a) for a in tide.ucx]
    tide_ucy = [np.asarray(a) for a in tide.ucy]
    tide_h = [np.asarray(a) for a in tide.h]

    for k in range(n_cases):
        if tide_ucx[k].ndim != 2:
            raise ValueError(
                f"Tide inputs require (nt, n) arrays. Case {k}: ucx shape={tide_ucx[k].shape}"
            )
        if (
            tide_ucy[k].shape != tide_ucx[k].shape
            or tide_h[k].shape != tide_ucx[k].shape
        ):
            raise ValueError(
                f"Tide inputs shape mismatch. Case {k}: "
                f"ucx={tide_ucx[k].shape}, ucy={tide_ucy[k].shape}, h={tide_h[k].shape}"
            )

    nt, n = tide_ucx[0].shape
    for k in range(1, n_cases):
        if tide_ucx[k].shape != (nt, n):
            raise ValueError("Tide inputs must have identical (nt, n) across cases.")

    return tide_ucx, tide_ucy, tide_h, n_cases, nt, n


def _maxQ_for_tv_tn(
    tv_tn: np.ndarray,
    td,
    rkm: np.ndarray,
    path_distances: np.ndarray,
    ship_depth: float,
    ship_length: float,
    criteria: tuple[float, float],
    prepare_data_for_excel,
):
    qmax = -np.inf
    t_best = -1
    payload_best = None
    xy_best = None
    crit_best = None

    for t in range(tv_tn.shape[0]):
        discharges, crit_values, xy_blocks = td.compute(
            rkm, path_distances, [tv_tn[t]], ship_depth, ship_length, criteria
        )
        if discharges[0].size == 0:
            continue

        j = int(np.nanargmax(np.abs(discharges[0])))
        q_t = float(np.abs(discharges[0][j]))

        if q_t > qmax:
            qmax = q_t
            t_best = t
            payload_best = prepare_data_for_excel(
                xy_blocks[0], discharges[0], crit_values[0]
            )
            xy_best = xy_blocks
            crit_best = crit_values

    if qmax == -np.inf:
        return np.nan, -1, None, None, None

    return qmax, t_best, payload_best, xy_best, crit_best


def append_tide_results(
    tide: TideInputs | None,
    rkm: np.ndarray,
    path_distances: np.ndarray,
    profile_angles: np.ndarray,
    bankward_sign: np.ndarray,
    configuration: Config,
    outputfiles: list[Path],
    td,
    prepare_data_for_excel,
    plotter: plotting.CrossFlow,
) -> None:
    """
    Run tide-specific cross-flow analysis and append results to the original Excel files.
    """
    if tide is None:
        warnings.warn(
            "Tide=True but no MAP/time tide inputs provided (likely FOU input). "
            "Tide analysis skipped.",
            RuntimeWarning,
        )
        return

    tide_ucx, tide_ucy, tide_h, n_cases, nt, n = _validate_tide_inputs(tide)
    if n_cases == 0:
        return

    SHEET_LABELS = ("Reference", "WithIntervention", "Difference")
    CRITERIA: tuple[float, float] = (0.15, 0.3)

    ship_depth = configuration.ship_params.depth
    ship_length = configuration.ship_params.length
    invertx = configuration.general.bool_flags.get("invertxaxis", False)

    # ------------------------------------------------------------
    # Compute per-case tide metrics
    # ------------------------------------------------------------
    upar_series = []
    tv_series = []
    idx_ebb_list = []
    idx_flood_list = []
    tv_ebb_list = []
    tv_flood_list = []

    idx_bankward_list = []
    tv_bankward_max_list = []
    q_bankward_list = []

    idx_riverward_list = []
    tv_riverward_max_list = []
    q_riverward_list = []

    for k in range(n_cases):
        upar_tn, tv_tn = flow.tide_time_series(
            tide_ucx[k],
            tide_ucy[k],
            tide_h[k],
            path_distances,
            profile_angles,
            ship_depth,
        )

        tv_tn = flow.orient_transverse_by_bankward_sign(
            tv_tn,
            bankward_sign,
        )

        (
            idx_bankward,
            tv_bankward_max,
            idx_riverward,
            tv_riverward_max,
        ) = flow.directional_tide_maxima(tv_tn)

        q_bankward = discharge_at_directional_maxima(
            tv_tn=tv_tn,
            time_indices=idx_bankward,
            path_distances=path_distances,
            ship_length=ship_length,
            ship_depth=ship_depth,
        )

        q_riverward = discharge_at_directional_maxima(
            tv_tn=tv_tn,
            time_indices=idx_riverward,
            path_distances=path_distances,
            ship_length=ship_length,
            ship_depth=ship_depth,
        )
        # No sign correction needed: `tv_bankward_max`/`tv_riverward_max` and
        # `discharge_at_directional_maxima`'s output both follow the same
        # physically oriented convention (positive = towards river axis,
        # negative = towards bank), so velocity and discharge already agree
        # in sign for both directions.

        idx_bankward_list.append(idx_bankward)
        tv_bankward_max_list.append(tv_bankward_max)
        q_bankward_list.append(q_bankward)

        idx_riverward_list.append(idx_riverward)
        tv_riverward_max_list.append(tv_riverward_max)
        q_riverward_list.append(q_riverward)

        idx_ebb, idx_flood, tv_ebb, tv_flood = flow.tide_peaks_from_upar(upar_tn, tv_tn)

        upar_series.append(upar_tn)
        tv_series.append(tv_tn)
        idx_ebb_list.append(idx_ebb)
        idx_flood_list.append(idx_flood)
        tv_ebb_list.append(tv_ebb)
        tv_flood_list.append(tv_flood)

    if n_cases > 1:
        tv_bankward_max_list.append(tv_bankward_max_list[1] - tv_bankward_max_list[0])
        q_bankward_list.append(q_bankward_list[1] - q_bankward_list[0])

        tv_riverward_max_list.append(
            tv_riverward_max_list[1] - tv_riverward_max_list[0]
        )
        q_riverward_list.append(q_riverward_list[1] - q_riverward_list[0])
    else:
        # Keep the lists aligned with Reference and no second case.
        tv_bankward_max_list.append(None)
        q_bankward_list.append(None)

        tv_riverward_max_list.append(None)
        q_riverward_list.append(None)

    # difference ebb/flood
    if n_cases > 1:
        tv_ebb_list.append(tv_ebb_list[1] - tv_ebb_list[0])
        tv_flood_list.append(tv_flood_list[1] - tv_flood_list[0])
    else:
        tv_ebb_list.append(None)
        tv_flood_list.append(None)

    # max transverse per point
    idx_tvmax_list = []
    tv_max_list = []

    for upar_tn, tv_tn in zip(upar_series, tv_series):
        idx_tvmax, tv_max, _ = flow.tide_max_transverse_per_point(upar_tn, tv_tn)
        idx_tvmax_list.append(idx_tvmax)
        tv_max_list.append(tv_max)

    if n_cases > 1:
        tv_max_list.append(tv_max_list[1] - tv_max_list[0])
    else:
        tv_max_list.append(None)

    # maxQ
    maxQ_value = []
    maxQ_time_index = []
    maxQ_payload = []

    for tv_tn in tv_series:
        q, tbest, payload, _, _ = _maxQ_for_tv_tn(
            tv_tn,
            td,
            rkm,
            path_distances,
            ship_depth,
            ship_length,
            CRITERIA,
            prepare_data_for_excel,
        )
        maxQ_value.append(q)
        maxQ_time_index.append(tbest)
        maxQ_payload.append(payload)

    if n_cases > 1:
        tv_diff = tv_series[1] - tv_series[0]
        q, tbest, payload, _, _ = _maxQ_for_tv_tn(
            tv_diff,
            td,
            rkm,
            path_distances,
            ship_depth,
            ship_length,
            CRITERIA,
            prepare_data_for_excel,
        )
        maxQ_value.append(q)
        maxQ_time_index.append(tbest)
        maxQ_payload.append(payload)
    else:
        maxQ_value.append(np.nan)
        maxQ_time_index.append(-1)
        maxQ_payload.append(None)

    # ------------------------------------------------------------
    # Append tide sheets to original Excel files
    # ------------------------------------------------------------
    rkm_km = rkm / 1000.0

    with pd.ExcelWriter(
        outputfiles[0],
        mode="a",
        if_sheet_exists="replace",
        engine="openpyxl",
    ) as writer:
        velocity_column_labels = ("raai (rkm)", "dwarsstroomsnelheid (m/s)")

        for label, tv in zip(SHEET_LABELS, tv_ebb_list):
            if tv is not None:
                support.to_excel(
                    writer, velocity_column_labels, f"{label}_Ebb", rkm_km, tv
                )

        for label, tv in zip(SHEET_LABELS, tv_flood_list):
            if tv is not None:
                support.to_excel(
                    writer, velocity_column_labels, f"{label}_Flood", rkm_km, tv
                )

        for label, tv in zip(SHEET_LABELS, tv_max_list):
            if tv is not None:
                support.to_excel(
                    writer,
                    velocity_column_labels,
                    f"{label}_MaxTransverse",
                    rkm_km,
                    tv,
                )

        directional_column_labels = (
            "raai (rkm)",
            "max. representatieve dwarsstroomsnelheid (m/s)",
            "instantaan dwarsstroomdebiet (m3/s)",
        )

        bankward_sheet_names = (
            "Reference_BankMax",
            "Intervention_BankMax",
            "Difference_BankMax",
        )

        riverward_sheet_names = (
            "Reference_RiverMax",
            "Intervention_RiverMax",
            "Difference_RiverMax",
        )

        for sheet_name, tv_max, discharge in zip(
            bankward_sheet_names,
            tv_bankward_max_list,
            q_bankward_list,
        ):
            if tv_max is None or discharge is None:
                continue

            support.to_excel(
                writer,
                directional_column_labels,
                sheet_name,
                rkm_km,
                tv_max,
                discharge,
            )

        for sheet_name, tv_max, discharge in zip(
            riverward_sheet_names,
            tv_riverward_max_list,
            q_riverward_list,
        ):
            if tv_max is None or discharge is None:
                continue

            support.to_excel(
                writer,
                directional_column_labels,
                sheet_name,
                rkm_km,
                tv_max,
                discharge,
            )

    with pd.ExcelWriter(
        outputfiles[1],
        mode="a",
        if_sheet_exists="replace",
        engine="openpyxl",
    ) as writer:
        discharge_column_labels = (
            "start (rkm)",
            "eind (rkm)",
            "dwarsstroomdebiet (m3/s)",
            "max. dwarsstroomsnelheid magnitude (m/s)",
            "criterium (m/s)",
            "overschrijding (0=FALSE,1=TRUE)",
        )

        for label, payload in zip(SHEET_LABELS, maxQ_payload):
            if payload is None:
                continue

            support.to_excel(
                writer,
                discharge_column_labels,
                f"{label}_MaxQ",
                *payload,
            )

    # ------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------
    plotter.create_figure_tide_velocities(
        rkm,
        tv_ebb_list,
        tv_flood_list,
        invertx,
        tide.fig_vel,
        annotation=None,
    )

    # MaxQ snapshot plot
    if maxQ_time_index[0] >= 0:
        tv_qmax_plot = []
        xy_qmax_plot = []
        crit_qmax_plot = []

        # Reference
        t_ref = maxQ_time_index[0]
        tv_ref_plot = tv_series[0][t_ref]
        tv_qmax_plot.append(tv_ref_plot)

        discharges_ref, crit_values_ref, xy_blocks_ref = td.compute(
            rkm, path_distances, [tv_ref_plot], ship_depth, ship_length, CRITERIA
        )
        xy_qmax_plot.extend(xy_blocks_ref)
        crit_qmax_plot.extend(crit_values_ref)

        # WithIntervention
        if n_cases > 1 and maxQ_time_index[1] >= 0:
            t_wi = maxQ_time_index[1]
            tv_wi_plot = tv_series[1][t_wi]
            tv_qmax_plot.append(tv_wi_plot)

            discharges_wi, crit_values_wi, xy_blocks_wi = td.compute(
                rkm, path_distances, [tv_wi_plot], ship_depth, ship_length, CRITERIA
            )
            xy_qmax_plot.extend(xy_blocks_wi)
            crit_qmax_plot.extend(crit_values_wi)

        plotter.create_figure(
            rkm,
            tv_qmax_plot,
            xy_qmax_plot,
            crit_qmax_plot,
            invertx,
            tide.fig_qmax,
            include_difference=False,
        )

    time_ref = None
    if tide.time_list and tide.time_list[0] is not None:
        time_ref = np.asarray(tide.time_list[0])

    plotter.create_figure_alongstream_timeseries(
        time_ref,
        upar_series[0],
        rkm,
        tide.fig_upar,
        threshold=None,
        annotation=None,
        idx_ebb=idx_ebb_list[0],
        idx_flood=idx_flood_list[0],
    )

    plotter.create_figure_tide_max_transverse(
        rkm,
        tv_max_list,
        time_ref,
        upar_series[0],
        idx_tvmax_list[0],
        tv_series[0],
        tide.fig_max_tv,
        invertx,
    )

    plotter.create_figure_directional_maxima(
        rkm,
        tv_bankward_max_list,
        q_bankward_list,
        plotting.DirectionalMaximaConfig.BANKWARD_VELOCITY_TITLE,
        plotting.DirectionalMaximaConfig.BANKWARD_DISCHARGE_TITLE,
        invertx,
        tide.fig_directional_bankward,
    )

    plotter.create_figure_directional_maxima(
        rkm,
        tv_riverward_max_list,
        q_riverward_list,
        plotting.DirectionalMaximaConfig.RIVERWARD_VELOCITY_TITLE,
        plotting.DirectionalMaximaConfig.RIVERWARD_DISCHARGE_TITLE,
        invertx,
        tide.fig_directional_riverward,
    )


def discharge_at_directional_maxima(
    tv_tn: np.ndarray,
    time_indices: np.ndarray,
    path_distances: np.ndarray,
    ship_length: float,
    ship_depth: float,
) -> np.ndarray:
    """
    Calculate the instantaneous transverse discharge at the timestep of the
    directional velocity maximum for every profile position.

    Sign convention
    ---------------
    Returned values follow the same physically oriented sign convention as
    `tv_tn` (positive = towards the river axis, negative = towards the bank),
    which also matches `directional_tide_maxima`'s bankward/riverward output.
    No extra sign correction is needed when combining the two.

    Parameters
    ----------
    tv_tn : np.ndarray
        Oriented representative transverse velocity, shape (nt, n).
    time_indices : np.ndarray
        Timestep index of the directional maximum per position, shape (n,).
        A value of -1 indicates that no maximum exists for that direction.
    path_distances : np.ndarray
        Cumulative distance along the profile, shape (n,).
    ship_length : float
        Representative ship length [m].
    ship_depth : float
        Representative ship depth [m].

    Returns
    -------
    np.ndarray
        Instantaneous transverse discharge per position, shape (n,).
    """
    tv_tn = np.asarray(tv_tn, dtype=float)
    time_indices = np.asarray(time_indices, dtype=int)

    if tv_tn.ndim != 2:
        raise ValueError("tv_tn must have shape (nt, n).")

    if time_indices.shape != (tv_tn.shape[1],):
        raise ValueError("time_indices must contain one index per profile position.")

    discharge_at_maximum = np.full(tv_tn.shape[1], np.nan, dtype=float)

    unique_timesteps = np.unique(time_indices[time_indices >= 0])

    for time_index in unique_timesteps:
        instantaneous_discharge = flow.local_transverse_discharge(
            path_distances=path_distances,
            transverse_velocity=tv_tn[time_index, :],
            ship_length=ship_length,
            ship_depth=ship_depth,
        )

        positions = time_indices == time_index
        discharge_at_maximum[positions] = instantaneous_discharge[positions]

    return discharge_at_maximum
