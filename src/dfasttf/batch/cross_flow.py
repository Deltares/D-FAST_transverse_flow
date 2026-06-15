from pathlib import Path
import numpy as np
import pandas as pd
from dfasttf.batch import operations, plotting, support
from dfasttf.config import Config
from dfasttf.kernel import flow
from dataclasses import dataclass
import warnings


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


def _fail_if_fourier_inputs(
    ucx: list[np.ndarray], ucy: list[np.ndarray], h: list[np.ndarray]
) -> None:
    """
    Hard fail if input looks like Fourier/running-mean based data (no flow vectors).
    In practice: Fourier files never provide ucx/ucy arrays for the profile extraction,
    so ucx/ucy/h will not have expected numeric shapes.
    This guard is mainly here to give a clearer error if a Fourier file slips through.
    """
    # If caller passed something non-numeric or empty, fail fast with clear message.
    # (Your IO layer should already prevent Fourier, but this keeps behavior strict.)
    for name, arrs in (("ucx", ucx), ("ucy", ucy), ("h", h)):
        if not arrs or arrs[0] is None:
            raise RuntimeError(
                "FOURIER/RUNNING-MEAN input detected (no ucx/ucy/h vectors). "
                "Cross-flow/tide analysis requires a MAP or snapshot flow file with ucx/ucy/waterdepth."
            )


def prepare_data_for_excel(xy_block, discharge, crit_value):
    CONVERT_M_TO_KM = 1000
    x_start = [xy[0][0] / CONVERT_M_TO_KM for xy in xy_block]
    x_end = [xy[0][-1] / CONVERT_M_TO_KM for xy in xy_block]
    y_max = [max(abs(xy[1])) for xy in xy_block]
    exceedance = y_max > abs(crit_value)
    return (x_start, x_end, discharge, y_max, crit_value, exceedance)


def run(
    ucx: list[np.ndarray],
    ucy: list[np.ndarray],
    water_depth: list[np.ndarray],
    path_distances: np.ndarray,
    profile_angles: np.ndarray,
    rkm: np.ndarray,
    configuration: Config,
    figfile: Path,
    outputfiles: list[Path],
    profile_points_xy: np.ndarray,
    axis_point_xy: np.ndarray,
    tide: TideInputs | None = None,
) -> None:
    """
    Single public entry point:
      - Always executes snapshot cross-flow analysis.
      - If Tide=True:
          * MAP/time tide inputs present -> runs tide analysis
          * otherwise -> warning + skip tide
      - Fourier inputs -> hard fail (safety net)
    """
    _fail_if_fourier_inputs(ucx, ucy, water_depth)

    # ============================================================
    # Cross-flow analysis via Fourier or Map file
    # ============================================================
    SHEET_LABELS = ("Reference", "WithIntervention", "Difference")
    CRITERIA: tuple[float, float] = (0.15, 0.3)  # criteria for transverse velocity

    rkm_km = rkm / 1000

    # Transverse velocity:
    COLUMN_LABELS = ("raai (rkm)", "dwarsstroomsnelheid (m/s)")
    transverse_velocity = []

    for x, y, wd in zip(ucx, ucy, water_depth):
        trans_flow = flow.trans_velocity(x, y, profile_angles)

        # Reorient so that:
        # positive = toward bank
        # negative = toward river axis
        trans_flow = flow.orient_transverse_toward_bank(
            trans_flow,
            profile_angles,
            profile_points_xy,
            axis_point_xy,
        )

        repr_trans_flow = flow.repr_trans_velocity(
            wd, trans_flow, path_distances, configuration.ship_params.depth
        )
        transverse_velocity.append(repr_trans_flow)

    data = [
        transverse_velocity[0],
        transverse_velocity[1] if len(transverse_velocity) > 1 else None,
        (
            (transverse_velocity[1] - transverse_velocity[0])
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
        "max. dwarsstroomsnelheid magnitude (m/s)",
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

    # ============================================================
    # TIDE (optional)
    # ============================================================
    if not configuration.general.bool_flags.get("tide", False):
        return

    if tide is None:
        warnings.warn(
            "Tide=True but no MAP/time tide inputs provided (likely Fourier input). Tide analysis skipped.",
            RuntimeWarning,
        )
        return

    # --- tide sanity checks ---
    n_cases = len(tide.ucx)
    if n_cases == 0:
        return

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
                f"Tide inputs shape mismatch. Case {k}: ucx={tide_ucx[k].shape}, ucy={tide_ucy[k].shape}, h={tide_h[k].shape}"
            )

    nt, n = tide_ucx[0].shape
    for k in range(1, n_cases):
        if tide_ucx[k].shape != (nt, n):
            raise ValueError("Tide inputs must have identical (nt, n) across cases.")

    # Constants used for tide products
    ship_depth = configuration.ship_params.depth
    ship_length = configuration.ship_params.length
    invertx = configuration.general.bool_flags.get("invertxaxis", False)

    # --- Compute upar_tn and tv_tn once per case using kernel (flow.py) ---
    upar_series = []
    tv_series = []
    idx_ebb_list = []
    idx_flood_list = []
    tv_ebb_list = []
    tv_flood_list = []

    for k in range(n_cases):
        # (nt, n) time series for one case

        upar_tn, tv_tn = flow.tide_time_series(
            tide_ucx[k],
            tide_ucy[k],
            tide_h[k],
            path_distances,
            profile_angles,
            ship_depth,
        )

        # Reorient so that:
        # positive = toward bank
        # negative = toward river axis
        tv_tn = flow.orient_transverse_toward_bank(
            tv_tn,
            profile_angles,
            profile_points_xy,
            axis_point_xy,
        )

        idx_ebb, idx_flood, tv_ebb, tv_flood = flow.tide_peaks_from_upar(upar_tn, tv_tn)

        upar_series.append(upar_tn)
        tv_series.append(tv_tn)
        idx_ebb_list.append(idx_ebb)
        idx_flood_list.append(idx_flood)
        tv_ebb_list.append(tv_ebb)
        tv_flood_list.append(tv_flood)

    idx_tvmax_list = []
    tv_max_list = []

    for upar_tn, tv_tn in zip(upar_series, tv_series):
        idx_tvmax, tv_max, _ = flow.tide_max_transverse_per_point(
            upar_tn,
            tv_tn,
        )
        idx_tvmax_list.append(idx_tvmax)
        tv_max_list.append(tv_max)

    # Difference if intervention exists
    if n_cases > 1:
        tv_max_list.append(tv_max_list[1] - tv_max_list[0])
    else:
        tv_max_list.append(None)

    # Difference for ebb/flood (if intervention)
    if n_cases > 1:
        tv_ebb_list.append(tv_ebb_list[1] - tv_ebb_list[0])
        tv_flood_list.append(tv_flood_list[1] - tv_flood_list[0])
    else:
        tv_ebb_list.append(None)
        tv_flood_list.append(None)

    # --- Max transverse discharge over tide (use tv_series directly) ---
    td = TransverseDischarge()

    def _maxQ_for_tv_tn(tv_tn: np.ndarray):
        qmax = -np.inf
        t_best = -1
        payload_best = None
        xy_best = None
        crit_best = None

        for t in range(tv_tn.shape[0]):
            discharges, crit_values, xy_blocks = td.compute(
                rkm, path_distances, [tv_tn[t]], ship_depth, ship_length, CRITERIA
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

    maxQ_value = []
    maxQ_time_index = []
    maxQ_payload = []

    for tv_tn in tv_series:
        q, tbest, payload, xy, crit = _maxQ_for_tv_tn(tv_tn)
        maxQ_value.append(q)
        maxQ_time_index.append(tbest)
        maxQ_payload.append(payload)
    if n_cases > 1:
        tv_diff = tv_series[1] - tv_series[0]
        q, tbest, payload, xy, crit = _maxQ_for_tv_tn(tv_diff)
        maxQ_value.append(q)
        maxQ_time_index.append(tbest)
        maxQ_payload.append(payload)
    else:
        maxQ_value.append(np.nan)
        maxQ_time_index.append(-1)
        maxQ_payload.append(None)

    # --- Figures ---
    plotter.create_figure_tide_velocities(
        rkm,
        tv_ebb_list,
        tv_flood_list,
        invertx,
        tide.fig_vel,
        annotation=None,
    )

    
    # --- Append tide sheets to the original Excel files ---
    rkm_km = rkm / 1000.0
    
    # 1) Append tide velocity-type sheets to transverse_velocity.xlsx
    with pd.ExcelWriter(
        outputfiles[0],
        mode="a",
        if_sheet_exists="replace",
        engine="openpyxl",
    ) as writer:
        velocity_column_labels = ("raai (rkm)", "dwarsstroomsnelheid (m/s)")
    
        for label, tv in zip(SHEET_LABELS, tv_ebb_list):
            if tv is None:
                continue
            support.to_excel(
                writer,
                velocity_column_labels,
                f"{label}_Ebb",
                rkm_km,
                tv,
            )
    
        for label, tv in zip(SHEET_LABELS, tv_flood_list):
            if tv is None:
                continue
            support.to_excel(
                writer,
                velocity_column_labels,
                f"{label}_Flood",
                rkm_km,
                tv,
            )
    
        for label, tv in zip(SHEET_LABELS, tv_max_list):
            if tv is None:
                continue
            support.to_excel(
                writer,
                velocity_column_labels,
                f"{label}_MaxTransverse",
                rkm_km,
                tv,
            )
    
    # 2) Append tide discharge sheets to transverse_flow.xlsx
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
    
        for i, label in enumerate(SHEET_LABELS):
            if maxQ_payload[i] is None:
                continue
            support.to_excel(
                writer,
                discharge_column_labels,
                f"{label}_MaxQ",
                *maxQ_payload[i],
            )


    # MaxQ snapshot plot: just show reference (simple & robust)
    if maxQ_time_index[0] >= 0:
        t_plot = maxQ_time_index[0]
        tv_ref_plot = tv_series[0][t_plot]
        discharges, crit_values, xy_blocks = td.compute(
            rkm, path_distances, [tv_ref_plot], ship_depth, ship_length, CRITERIA
        )
        plotter.create_figure(
            rkm,
            [tv_ref_plot],
            xy_blocks,
            crit_values,
            invertx,
            tide.fig_qmax,
        )

    # Alongstream u_parallel time series (Reference) with per-rkm points
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
