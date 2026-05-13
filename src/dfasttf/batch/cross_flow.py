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
        repr_trans_flow = flow.repr_trans_velocity(
            wd, trans_flow, path_distances, configuration.ship_params.depth
        )
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


def _alongstream_velocity(
    u: np.ndarray, v: np.ndarray, angles_deg: np.ndarray
) -> np.ndarray:
    """Along-stream velocity component: u*cos(theta) + v*sin(theta)."""
    th = np.radians(angles_deg)
    return u * np.cos(th) + v * np.sin(th)


def run_tide(
    ucx: list[np.ndarray],  # each: (nt, n)
    ucy: list[np.ndarray],  # each: (nt, n)
    water_depth: list[np.ndarray],  # each: (nt, n)
    path_distances: np.ndarray,  # (n,)
    profile_angles: np.ndarray,  # (n,)
    rkm: np.ndarray,  # (n,)
    configuration,
    outputfile: Path,
    figfile_tide_vel: Path,
    figfile_tide_q: Path,
    figfile_tide_upar: Path,  # <-- nieuw
    time_list: list[np.ndarray | None],  # <-- nieuw
) -> None:
    """
    Tide analysis (only called when TideAnalysis=True and tide data is present).

    Produces:
      - transverse velocity at peak ebb per rKm (Reference / WithIntervention / Difference)
      - transverse velocity at peak flood per rKm (Reference / WithIntervention / Difference)
      - max transverse discharge during the tide (Reference / WithIntervention / Difference)
      - figure of ebb/flood transverse velocities
      - old-style crossflow figure at time of max discharge (driven by Difference if available)
    """
    SHEET_LABELS = ("Reference", "WithIntervention", "Difference")
    CRITERIA: tuple[float, float] = (0.15, 0.3)

    ship_depth = configuration.ship_params.depth
    ship_length = configuration.ship_params.length
    invertx = configuration.general.bool_flags.get("invertxaxis", False)

    # --- basic checks ---
    upar_series_list = []
    n_cases = len(ucx)
    if n_cases == 0:
        return

    # ensure arrays are numpy and correct shape
    ucx = [np.asarray(a) for a in ucx]
    ucy = [np.asarray(a) for a in ucy]
    water_depth = [np.asarray(a) for a in water_depth]

    for k in range(n_cases):
        if ucx[k].ndim != 2:
            raise ValueError(
                f"run_tide requires (nt, n) arrays. Case {k} ucx shape={ucx[k].shape}"
            )
        if ucy[k].shape != ucx[k].shape or water_depth[k].shape != ucx[k].shape:
            raise ValueError(
                f"run_tide: ucx/ucy/h shapes must match. Case {k} got ucx={ucx[k].shape}, ucy={ucy[k].shape}, h={water_depth[k].shape}"
            )

    nt, n = ucx[0].shape
    for k in range(1, n_cases):
        if ucx[k].shape[0] != nt or ucx[k].shape[1] != n:
            raise ValueError("run_tide: all cases must have same (nt, n) shape.")

    rkm_km = rkm / 1000.0

    # ============================================================
    # 1) Compute peak ebb/flood transverse velocity per rKm
    # ============================================================
    ebb_list = []
    flood_list = []  
    idx_ebb_list = []  
    idx_flood_list = []  
    ebb_valid_frac_list = []
    flood_valid_frac_list = []

    for k in range(n_cases):
        upar_pos = np.full(n, -np.inf)
        upar_neg = np.full(n, np.inf)
        idx_ebb = np.full(n, -1, dtype=int)
        idx_flood = np.full(n, -1, dtype=int)
        tv_ebb = np.full(n, np.nan, dtype=float)
        tv_flood = np.full(n, np.nan, dtype=float)
        upar_tn = np.empty((nt, n), dtype=float)

        for t in range(nt):
            u_t = ucx[k][t]
            v_t = ucy[k][t]
            h_t = water_depth[k][t]
            upar_tn[t] = _alongstream_velocity(u_t, v_t, profile_angles)

            # representative transverse velocity at this time
            w_t = flow.trans_velocity(u_t, v_t, profile_angles)
            tv_t = flow.repr_trans_velocity(h_t, w_t, path_distances, ship_depth)

            # along-stream for ebb/flood selection
            upar = _alongstream_velocity(u_t, v_t, profile_angles)

            # ebb: strongest downstream => maximum positive upar
            mpos = (upar > 0.0) & (upar > upar_pos)
            if np.any(mpos):
                upar_pos[mpos] = upar[mpos]
                idx_ebb[mpos] = t
                tv_ebb[mpos] = tv_t[mpos]

            # flood: strongest upstream => most negative upar
            mneg = (upar < 0.0) & (upar < upar_neg)
            if np.any(mneg):
                upar_neg[mneg] = upar[mneg]
                idx_flood[mneg] = t
                tv_flood[mneg] = tv_t[mneg]

        mean_u = np.nanmean(upar_tn, axis=0)
        max_u = np.nanmax(upar_tn, axis=0)
        min_u = np.nanmin(upar_tn, axis=0)
        amp_u = max_u - min_u

        frac_neg = np.mean(upar_tn < 0.0, axis=0)
        frac_pos = np.mean(upar_tn > 0.0, axis=0)

        ebb_delta = max_u - mean_u
        flood_delta = mean_u - min_u

        #TODO probably would be better to place these limits in the config section, for now hardcoded. 
        EPS_ABS = 0.05  # m/s: noise floor
        EPS_AMP = 0.10  # m/s: required swing for meaningful tide
        EPS_DELTA = 0.3  # m/s: peak distinctness vs mean
        MIN_FRAC = 0.10  # at least 10% sign presence

        ebb_valid = (
            (frac_pos >= MIN_FRAC)
            & (max_u >= EPS_ABS)
            & (amp_u >= EPS_AMP)
            & (ebb_delta >= EPS_DELTA)
        )
        flood_valid = (
            (frac_neg >= MIN_FRAC)
            & (min_u <= -EPS_ABS)
            & (amp_u >= EPS_AMP)
            & (flood_delta >= EPS_DELTA)
        )

        # apply masks (so invalid points become NaN and time index -1)
        tv_ebb[~ebb_valid] = np.nan
        tv_flood[~flood_valid] = np.nan
        idx_ebb[~ebb_valid] = -1
        idx_flood[~flood_valid] = -1

        # store results (masked)
        ebb_list.append(tv_ebb)
        flood_list.append(tv_flood)
        idx_ebb_list.append(idx_ebb)
        idx_flood_list.append(idx_flood)

        # store validity fractions for figure annotation
        ebb_valid_frac = float(
            np.mean(np.isfinite(tv_ebb))
        )  # fraction points with meaningful ebb
        flood_valid_frac = float(
            np.mean(np.isfinite(tv_flood))
        )  # fraction points with meaningful flood

        # keep per-case stats
        ebb_valid_frac_list.append(ebb_valid_frac)
        flood_valid_frac_list.append(flood_valid_frac)
        upar_series_list.append(upar_tn)

    # add Difference for ebb/flood if intervention exists
    if n_cases > 1:
        ebb_list.append(ebb_list[1] - ebb_list[0])
        flood_list.append(flood_list[1] - flood_list[0])
    else:
        ebb_list.append(None)
        flood_list.append(None)

    if n_cases > 1:
        ebb_valid_frac_list.append(float(np.mean(np.isfinite(ebb_list[2]))))
        flood_valid_frac_list.append(float(np.mean(np.isfinite(flood_list[2]))))
    else:
        ebb_valid_frac_list.append(np.nan)
        flood_valid_frac_list.append(np.nan)

    # ============================================================
    # 2) Max transverse discharge over the tide
    # ============================================================
    td = TransverseDischarge()
    maxQ_value = [np.nan] * 3  # ref, int, diff
    maxQ_time_index = [-1] * 3
    maxQ_payload = [None] * 3  # payload for excel table
    maxQ_xy_blocks = [None] * 3  # for plotting (old style)
    maxQ_crit_values = [None] * 3
    maxQ_tv_at_t = [None] * 3

    def _maxQ_for_tv_time_series(tv_tn: np.ndarray):
        """Return qmax, t_best, (payload, xy_blocks, crit_values, tv_at_tbest)."""
        qmax = -np.inf
        t_best = -1
        payload_best = None
        xy_best = None
        crit_best = None
        tv_best = None

        for t in range(tv_tn.shape[0]):
            tv_t = tv_tn[t]
            discharges, crit_values, xy_blocks = td.compute(
                rkm, path_distances, [tv_t], ship_depth, ship_length, CRITERIA
            )
            if discharges[0].size == 0:
                continue

            i = int(np.nanargmax(np.abs(discharges[0])))
            q_t = float(np.abs(discharges[0][i]))
            if q_t > qmax:
                qmax = q_t
                t_best = t
                payload_best = prepare_data_for_excel(
                    xy_blocks[0], discharges[0], crit_values[0]
                )
                xy_best = xy_blocks
                crit_best = crit_values
                tv_best = tv_t

        if qmax == -np.inf:
            return np.nan, -1, None, None, None, None
        return qmax, t_best, payload_best, xy_best, crit_best, tv_best

    # Build representative transverse velocity time series per case
    tv_series = []
    for k in range(n_cases):
        tv_tn = np.empty((nt, n), dtype=float)
        for t in range(nt):
            w_t = flow.trans_velocity(ucx[k][t], ucy[k][t], profile_angles)
            tv_tn[t] = flow.repr_trans_velocity(
                water_depth[k][t], w_t, path_distances, ship_depth
            )
        tv_series.append(tv_tn)

    # Compute maxQ for reference and intervention
    q0, t0, p0, xy0, c0, tv0 = _maxQ_for_tv_time_series(tv_series[0])
    maxQ_value[0], maxQ_time_index[0], maxQ_payload[0] = q0, t0, p0
    maxQ_xy_blocks[0], maxQ_crit_values[0], maxQ_tv_at_t[0] = xy0, c0, tv0

    if n_cases > 1:
        q1, t1, p1, xy1, c1, tv1 = _maxQ_for_tv_time_series(tv_series[1])
        maxQ_value[1], maxQ_time_index[1], maxQ_payload[1] = q1, t1, p1
        maxQ_xy_blocks[1], maxQ_crit_values[1], maxQ_tv_at_t[1] = xy1, c1, tv1

        # Difference time series and maxQ
        tv_diff = tv_series[1] - tv_series[0]
        qd, td_, pd_, xyd_, cd_, tvd_ = _maxQ_for_tv_time_series(tv_diff)
        maxQ_value[2], maxQ_time_index[2], maxQ_payload[2] = qd, td_, pd_
        maxQ_xy_blocks[2], maxQ_crit_values[2], maxQ_tv_at_t[2] = xyd_, cd_, tvd_

    # ============================================================
    # 3) Write Excel
    # ============================================================
    with pd.ExcelWriter(outputfile) as writer:
        # Ebb velocities
        for label, tv in zip(SHEET_LABELS, ebb_list):
            if tv is None:
                continue
            support.to_excel(
                writer,
                ("raai (rkm)", "dwarsstroomsnelheid_ebb (m/s)"),
                f"{label}_Ebb",
                rkm_km,
                tv,
            )

        # Flood velocities
        for label, tv in zip(SHEET_LABELS, flood_list):
            if tv is None:
                continue
            support.to_excel(
                writer,
                ("raai (rkm)", "dwarsstroomsnelheid_flood (m/s)"),
                f"{label}_Flood",
                rkm_km,
                tv,
            )

        support.to_excel(
            writer,
            ("raai (rkm)", "t_index_peak_ebb"),
            "Reference_tEbb",
            rkm_km,
            idx_ebb_list[0],
        )
        support.to_excel(
            writer,
            ("raai (rkm)", "t_index_peak_flood"),
            "Reference_tFlood",
            rkm_km,
            idx_flood_list[0],
        )
        if n_cases > 1:
            support.to_excel(
                writer,
                ("raai (rkm)", "t_index_peak_ebb"),
                "WithIntervention_tEbb",
                rkm_km,
                idx_ebb_list[1],
            )
            support.to_excel(
                writer,
                ("raai (rkm)", "t_index_peak_flood"),
                "WithIntervention_tFlood",
                rkm_km,
                idx_flood_list[1],
            )

        # Max discharge tables
        COLUMN_LABELS = (
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
                COLUMN_LABELS,
                f"{label}_MaxDischarge_t{maxQ_time_index[i]}",
                *maxQ_payload[i],
            )

        # Simple summary sheet
        summary_labels = ["Reference", "WithIntervention", "Difference"]
        summary_q = [maxQ_value[0], maxQ_value[1], maxQ_value[2]]
        summary_t = [maxQ_time_index[0], maxQ_time_index[1], maxQ_time_index[2]]
        df = pd.DataFrame(
            {"Case": summary_labels, "Qmax (m3/s)": summary_q, "t_index": summary_t}
        )
        df.to_excel(writer, sheet_name="Summary_MaxQ", index=False)

    # ============================================================
    # 4) Figures
    # ============================================================

    def _fmt_pct(x):
        return "n/a" if (x is None or not np.isfinite(x)) else f"{100*x:.0f}%"

    # same thresholds you used (keep consistent)
    thr_text = f"criteria: |u∥|≥{EPS_ABS:.2f} m/s, amp≥{EPS_AMP:.2f} m/s, Δpeak≥{EPS_DELTA:.2f} m/s, frac≥{MIN_FRAC:.2f}"

    ebb_text = f"valid ebb points: Ref {_fmt_pct(ebb_valid_frac_list[0])}"
    flood_text = f"valid flood points: Ref {_fmt_pct(flood_valid_frac_list[0])}"

    if n_cases > 1:
        ebb_text += f", Plan {_fmt_pct(ebb_valid_frac_list[1])}, Diff {_fmt_pct(ebb_valid_frac_list[2])}"
        flood_text += f", Plan {_fmt_pct(flood_valid_frac_list[1])}, Diff {_fmt_pct(flood_valid_frac_list[2])}"

    # extra warning when flood is basically absent
    warn = ""
    if np.isfinite(flood_valid_frac_list[0]) and flood_valid_frac_list[0] < 0.05:
        warn = "WARNING: flood signal largely absent (few/none negative u∥)."

    tide_annotation = f"{ebb_text} | {flood_text}\n{thr_text}"
    if warn:
        tide_annotation += f"\n{warn}"

    # Figure 1: ebb/flood transverse velocities
    plotter = plotting.CrossFlow()
    plotter.create_figure_tide_velocities(
        rkm,
        ebb_list,
        flood_list,
        invertx,
        figfile_tide_vel,
        annotation=tide_annotation,
    )

    if n_cases > 1 and maxQ_tv_at_t[2] is not None:
        tv_plot = [maxQ_tv_at_t[0], maxQ_tv_at_t[1], maxQ_tv_at_t[2]]
        xy_plot = maxQ_xy_blocks[
            2
        ]  
        crit_plot = maxQ_crit_values[2]

        plotter.create_figure(
            rkm,
            tv_plot,
            maxQ_xy_blocks[2],
            maxQ_crit_values[2],
            invertx,
            figfile_tide_q,
        )
    elif maxQ_tv_at_t[0] is not None:
        plotter.create_figure(
            rkm,
            [maxQ_tv_at_t[0]],
            maxQ_xy_blocks[0],
            maxQ_crit_values[0],
            invertx,
            figfile_tide_q,
        )

    # Reference time axis
    time_ref = None
    if time_list and time_list[0] is not None:
        time_ref = np.asarray(time_list[0])

    plotter.create_figure_alongstream_timeseries(
        time_ref,
        upar_series_list[0],  # u∥(t, rkm) reference
        rkm,
        figfile_tide_upar,
        threshold=EPS_ABS,
        annotation=(
            f"criteria: |u∥|≥{EPS_ABS:.2f} m/s, amp≥{EPS_AMP:.2f} m/s, "
            f"Δpeak≥{EPS_DELTA:.2f} m/s, frac≥{MIN_FRAC:.2f}"
        ),
        idx_ebb=idx_ebb_list[0],  # <-- exact per-rkm ebb times
        idx_flood=idx_flood_list[0],  # <-- exact per-rkm flood times
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
