import numpy as np
from scipy.ndimage import uniform_filter1d
from typing import Tuple


def trans_velocity(u0: np.ndarray, v0: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    Calculates the transversal (perpendicular) component of the flow velocity.
    u0: x-component of velocity
    v0: y-component of velocity
    angles: angles in degrees (0 degrees is to the right of the x-axis)
    """

    angles_rad = np.radians(angles)  # convert angles in degrees to radians
    w0 = u0 * (-np.sin(angles_rad)) + v0 * np.cos(angles_rad)
    return w0


def repr_trans_velocity(
    water_depth: np.ndarray,
    trans_flow: np.ndarray,
    path_distances: np.ndarray,
    ship_depth: float,
) -> np.ndarray:
    """
    Calculates the representative transversal velocity at intersection points according to RBK specifications.

    Input:
    water_depth: water depth at intersection points (n,)
    trans_flow: transversal flow velocity at intersection points (n,)
    path_distances: cumulative distance between intersection points (n,)
    ship_depth: depth of a representative ship

    Returns:
    u_repr: representative transversal velocity (n,)
    """

    n = water_depth.shape[0]
    if trans_flow.shape[0] != n or path_distances.shape[0] != n:
        raise ValueError(
            f"water_depth, trans_flow, and path_distances must have compatible shapes. "
            f"Got water_depth={n}, trans_flow={trans_flow.shape[0]}, path_distances={path_distances.shape[0]}."
        )

    trans_flow_rm = uniform_filter1d(
        trans_flow, size=2, mode='nearest'
    )  # rolling average
    trans_flow_rm[-1] = np.nan
    seg_len = np.diff(path_distances)  # segment lengths
    bad_segments_ind = ~np.isfinite(seg_len) | (
        seg_len <= 0
    )  # Guard: identify zero/negative lengths

    q_trans = trans_flow_rm[:-1] * seg_len * water_depth[:-1]  # transversal discharge
    repr_depth = np.fmax(water_depth[:-1], ship_depth)  # representative depth
    denom = seg_len * repr_depth  # denominator for representative velocity

    # Compute u_repr for segments, then align back to node indices
    with np.errstate(divide='ignore', invalid='ignore'):
        u_seg = q_trans / denom  # (n-1,)

    # Mask bad and any computed NaNs
    mask = bad_segments_ind | ~np.isfinite(u_seg)
    u_seg_filled = u_seg.copy()

    if mask.any():
        x = np.arange(u_seg.size)
        good = ~mask
        if good.any():
            # interpolate over internal gaps
            u_seg_filled[mask] = np.interp(x[mask], x[good], u_seg[good])

    # Prepend NaN for the first node (no previous segment)
    u_repr = np.empty(n, dtype=float)
    u_repr[1:] = u_seg_filled
    u_repr[0] = u_repr[1]

    return u_repr


def trans_discharge(u_integral: np.ndarray, ship_depth: float) -> np.ndarray:
    """
    Calculates the transversal discharge.
    u_integral = integral of flow velocity over cross-sectional width
    ship_depth: depth of the ship
    """

    q = u_integral * ship_depth
    return q


def tide_max_transverse_per_point(
    upar_tn: np.ndarray,   # (nt, n)
    tv_tn: np.ndarray,     # (nt, n)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Determine, for each profile point, when the absolute transverse velocity is maximal.

    Parameters
    ----------
    upar_tn : np.ndarray
        Along-stream velocity time series, shape (nt, n).
    tv_tn : np.ndarray
        Representative transverse velocity time series, shape (nt, n).

    Returns
    -------
    idx_tvmax : (n,) int ndarray
        Timestep index where |tv| is maximal for each profile point.
    tv_max : (n,) float ndarray
        Signed transverse velocity at that timestep.
    upar_at_tvmax : (n,) float ndarray
        Along-stream velocity at that timestep.
    """
    upar_tn = np.asarray(upar_tn)
    tv_tn = np.asarray(tv_tn)

    if upar_tn.ndim != 2 or tv_tn.ndim != 2 or upar_tn.shape != tv_tn.shape:
        raise ValueError(
            f"upar_tn and tv_tn must be same shape (nt, n). Got {upar_tn.shape} and {tv_tn.shape}"
        )

    idx_tvmax = np.nanargmax(np.abs(tv_tn), axis=0).astype(int)

    n = tv_tn.shape[1]
    i = np.arange(n)

    tv_max = tv_tn[idx_tvmax, i]
    upar_at_tvmax = upar_tn[idx_tvmax, i]

    return idx_tvmax, tv_max, upar_at_tvmax

def alongstream_velocity(
    u: np.ndarray, v: np.ndarray, angles_deg: np.ndarray
) -> np.ndarray:
    """
    Along-stream velocity component:
    u_parallel = u*cos(theta) + v*sin(theta)

    angles_deg: degrees, 0 deg along +x axis.
    """
    th = np.radians(angles_deg)
    return u * np.cos(th) + v * np.sin(th)



def tide_time_series(
    ucx_tn: np.ndarray,  # (nt, n)
    ucy_tn: np.ndarray,  # (nt, n)
    h_tn: np.ndarray,  # (nt, n)
    path_distances: np.ndarray,  # (n,)
    angles_deg: np.ndarray,  # (n,)
    ship_depth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute time series needed for tide analysis for ONE case.

    Returns
    -------
    upar_tn : (nt, n) ndarray
        Along-stream velocity time series.
    tv_tn : (nt, n) ndarray
        Representative transverse velocity time series.
    """
    ucx_tn = np.asarray(ucx_tn)
    ucy_tn = np.asarray(ucy_tn)
    h_tn = np.asarray(h_tn)

    if ucx_tn.ndim != 2 or ucy_tn.ndim != 2 or h_tn.ndim != 2:
        raise ValueError("ucx_tn, ucy_tn, h_tn must be 2D arrays with shape (nt, n).")
    if ucx_tn.shape != ucy_tn.shape or ucx_tn.shape != h_tn.shape:
        raise ValueError(
            f"shape mismatch: ucx_tn={ucx_tn.shape}, ucy_tn={ucy_tn.shape}, h_tn={h_tn.shape}"
        )

    nt, n = ucx_tn.shape
    if path_distances.shape[0] != n or angles_deg.shape[0] != n:
        raise ValueError(
            "path_distances and angles_deg must have length n (= number of profile points)."
        )

    
    upar_tn = np.empty((nt, n), dtype=float)
    tv_tn = np.empty((nt, n), dtype=float)
    
    for t in range(nt):
        u = ucx_tn[t]
        v = ucy_tn[t]
        h = h_tn[t]
    
        upar_tn[t] = alongstream_velocity(u, v, angles_deg)
    
        w = trans_velocity(u, v, angles_deg)
        tv_tn[t] = repr_trans_velocity(h, w, path_distances, ship_depth)
        
    return upar_tn, tv_tn


def tide_peaks_from_upar(
    upar_tn: np.ndarray,  # (nt, n)
    tv_tn: np.ndarray,  # (nt, n)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Given u_parallel(t,i) and tv(t,i), compute per-point peak ebb/flood indices and tv at those indices.

    Returns
    -------
    idx_ebb : (n,) int ndarray
    idx_flood : (n,) int ndarray
    tv_ebb : (n,) float ndarray
    tv_flood : (n,) float ndarray
    """
    upar_tn = np.asarray(upar_tn)
    tv_tn = np.asarray(tv_tn)
    if upar_tn.ndim != 2 or tv_tn.ndim != 2 or upar_tn.shape != tv_tn.shape:
        raise ValueError(
            f"upar_tn and tv_tn must be same shape (nt, n). Got {upar_tn.shape} and {tv_tn.shape}"
        )

    idx_ebb = np.nanargmax(upar_tn, axis=0).astype(int)
    idx_flood = np.nanargmin(upar_tn, axis=0).astype(int)

    n = upar_tn.shape[1]
    i = np.arange(n)
    tv_ebb = tv_tn[idx_ebb, i]
    tv_flood = tv_tn[idx_flood, i]

    return idx_ebb, idx_flood, tv_ebb, tv_flood


def orient_transverse_by_bankward_sign(
    transverse_velocity: np.ndarray,
    bankward_sign: np.ndarray,
) -> np.ndarray:
    """
    Orient transverse velocity such that:

    positive = towards river axis
    negative = towards bank

    Parameters
    ----------
    transverse_velocity : np.ndarray
        Shape (n,) for snapshot or (nt, n) for tide.
    bankward_sign : np.ndarray
        Shape (n,), values +1 or -1.

    Returns
    -------
    np.ndarray
        Oriented transverse velocity with the same shape as input.
    """
    transverse_velocity = np.asarray(transverse_velocity)
    bankward_sign = np.asarray(bankward_sign, dtype=float)

    if bankward_sign.ndim != 1:
        raise ValueError("bankward_sign must be one-dimensional.")

    if not np.all(np.isin(bankward_sign, (-1.0, 1.0))):
        raise ValueError("bankward_sign may only contain -1 and +1.")

    if transverse_velocity.ndim == 1:
        if transverse_velocity.shape[0] != bankward_sign.shape[0]:
            raise ValueError(
                "transverse_velocity and bankward_sign length mismatch."
            )
        return transverse_velocity * bankward_sign

    if transverse_velocity.ndim == 2:
        if transverse_velocity.shape[1] != bankward_sign.shape[0]:
            raise ValueError(
                "transverse_velocity and bankward_sign length mismatch."
            )
        return transverse_velocity * bankward_sign[np.newaxis, :]

    raise ValueError("transverse_velocity must be 1D or 2D.")


def directional_tide_maxima(
    tv_tn: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Determine the maximum bankward and riverward transverse velocity
    for every profile position.

    Sign convention
    ---------------
    tv_tn > 0 : towards the river axis
    tv_tn < 0 : towards the bank

    Parameters
    ----------
    tv_tn : np.ndarray
        Oriented representative transverse velocity, shape (nt, n).

    Returns
    -------
    idx_bankward : np.ndarray
        Timestep of maximum bankward velocity per position, shape (n,).
        Value -1 indicates that no bankward flow occurs.
    tv_bankward_max : np.ndarray
        Maximum bankward velocity per position, shape (n,). Negative values.
    idx_riverward : np.ndarray
        Timestep of maximum riverward velocity per position, shape (n,).
        Value -1 indicates that no riverward flow occurs.
    tv_riverward_max : np.ndarray
        Maximum riverward velocity per position, shape (n,). Positive values.
    """
    tv_tn = np.asarray(tv_tn, dtype=float)

    if tv_tn.ndim != 2:
        raise ValueError("tv_tn must have shape (nt, n).")

    _, n = tv_tn.shape

    idx_bankward = np.full(n, -1, dtype=int)
    idx_riverward = np.full(n, -1, dtype=int)

    tv_bankward_max = np.full(n, np.nan, dtype=float)
    tv_riverward_max = np.full(n, np.nan, dtype=float)

    for i in range(n):
        series = tv_tn[:, i]

        riverward_mask = np.isfinite(series) & (series > 0.0)
        if np.any(riverward_mask):
            valid_indices = np.flatnonzero(riverward_mask)
            local_index = int(np.argmax(series[riverward_mask]))
            time_index = valid_indices[local_index]

            idx_riverward[i] = time_index
            tv_riverward_max[i] = series[time_index]

        bankward_mask = np.isfinite(series) & (series < 0.0)
        if np.any(bankward_mask):
            valid_indices = np.flatnonzero(bankward_mask)
            local_index = int(np.argmin(series[bankward_mask]))
            time_index = valid_indices[local_index]

            idx_bankward[i] = time_index
            tv_bankward_max[i] = series[time_index]

    return (
        idx_bankward,
        tv_bankward_max,
        idx_riverward,
        tv_riverward_max,
    )


def local_transverse_discharge(
    path_distances: np.ndarray,
    transverse_velocity: np.ndarray,
    ship_length: float,
    ship_depth: float,
) -> np.ndarray:
    """
    Calculate the instantaneous transverse discharge over a ship-length
    window centred on every profile position.

    Positive discharge represents flow towards the river axis.
    Negative discharge represents flow towards the bank.

    Duplicate profile positions are combined by averaging their finite
    transverse velocities.
    """
    path_distances = np.asarray(path_distances, dtype=float)
    transverse_velocity = np.asarray(transverse_velocity, dtype=float)

    if path_distances.ndim != 1:
        raise ValueError("path_distances must be one-dimensional.")

    if transverse_velocity.ndim != 1:
        raise ValueError("transverse_velocity must be one-dimensional.")

    if path_distances.shape != transverse_velocity.shape:
        raise ValueError(
            "path_distances and transverse_velocity must have the same shape. "
            f"Got {path_distances.shape} and {transverse_velocity.shape}."
        )

    if path_distances.size < 2:
        raise ValueError("At least two profile positions are required.")

    if not np.all(np.isfinite(path_distances)):
        raise ValueError("path_distances must contain only finite values.")

    if np.any(np.diff(path_distances) < 0.0):
        raise ValueError("path_distances must be non-decreasing.")

    if ship_length <= 0.0:
        raise ValueError("ship_length must be larger than zero.")

    if ship_depth <= 0.0:
        raise ValueError("ship_depth must be larger than zero.")

    # Combine duplicate profile positions. The inverse index is used to map
    # the calculated discharge back to the original profile positions.
    unique_distances, inverse_indices = np.unique(
        path_distances,
        return_inverse=True,
    )

    if unique_distances.size < 2:
        raise ValueError(
            "At least two unique profile positions are required."
        )

    unique_velocity = np.full(unique_distances.shape, np.nan, dtype=float)

    for unique_index in range(unique_distances.size):
        values = transverse_velocity[inverse_indices == unique_index]
        finite_values = values[np.isfinite(values)]

        if finite_values.size > 0:
            unique_velocity[unique_index] = np.mean(finite_values)

    valid = np.isfinite(unique_velocity)

    if np.count_nonzero(valid) < 2:
        return np.full(path_distances.shape, np.nan, dtype=float)

    discharge_unique = np.full(unique_distances.shape, np.nan, dtype=float)
    half_ship_length = ship_length / 2.0

    for i, center_distance in enumerate(unique_distances):
        window_start = center_distance - half_ship_length
        window_stop = center_distance + half_ship_length

        # A complete ship-length window must fit within the profile.
        if (
            window_start < unique_distances[0]
            or window_stop > unique_distances[-1]
        ):
            continue

        internal = (
            valid
            & (unique_distances > window_start)
            & (unique_distances < window_stop)
        )

        integration_distance = np.concatenate(
            (
                [window_start],
                unique_distances[internal],
                [window_stop],
            )
        )

        integration_velocity = np.interp(
            integration_distance,
            unique_distances[valid],
            unique_velocity[valid],
        )

        velocity_integral = np.trapezoid(
            integration_velocity,
            integration_distance,
        )

        discharge_unique[i] = velocity_integral * ship_depth

    # Map the discharge back to the original profile positions.
    return discharge_unique[inverse_indices]
