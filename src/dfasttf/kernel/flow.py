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


def orient_transverse_by_profile_side(
    transverse_velocity: np.ndarray,
    profile_is_right: bool,
) -> np.ndarray:
    """
    Reorient transverse velocity based on whether the profile lies on the
    right or left side of the river center line.

    Convention
    ----------
    Positive values should represent flow toward the bank and negative values
    flow toward the river center.

    Assumption
    ----------
    The current transverse sign convention produced by `trans_velocity()` is
    assumed to correspond to:
    - right-side profiles  -> keep sign
    - left-side profiles   -> flip sign

    If validation shows the opposite convention is needed, simply swap the sign
    assignment below.

    Parameters
    ----------
    transverse_velocity : np.ndarray
        Transverse velocity array, shape (n,) or (nt, n).
    profile_is_right : bool
        True if the profile lies on the right side of the center line.

    Returns
    -------
    np.ndarray
        Reoriented transverse velocity with the same shape as input.
    """
    sign = 1.0 if profile_is_right else -1.0

    transverse_velocity = np.asarray(transverse_velocity)

    if transverse_velocity.ndim == 1:
        return transverse_velocity * sign

    if transverse_velocity.ndim == 2:
        return transverse_velocity * sign

    raise ValueError("transverse_velocity must be 1D or 2D")
