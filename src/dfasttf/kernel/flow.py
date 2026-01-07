import numpy as np
from scipy.ndimage import uniform_filter1d


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


def repr_trans_velocity(water_depth: np.ndarray,
                        trans_flow: np.ndarray,
                        path_distances: np.ndarray,
                        ship_depth: float) -> np.ndarray:
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
        raise ValueError(f"water_depth, trans_flow, and path_distances must have compatible shapes. "
                         f"Got water_depth={n}, trans_flow={trans_flow.shape[0]}, path_distances={path_distances.shape[0]}.")

    trans_flow_rm = uniform_filter1d(trans_flow, size=2, mode='nearest') # rolling average
    trans_flow_rm[-1] = np.nan
    seg_len = np.diff(path_distances) # segment lengths
    bad_segments_ind = ~np.isfinite(seg_len) | (seg_len <= 0) # Guard: identify zero/negative lengths

    q_trans = trans_flow_rm[:-1] * seg_len * water_depth[:-1]  # transversal discharge
    repr_depth = np.fmax(water_depth[:-1], ship_depth)  # representative depth
    denom = seg_len * repr_depth # denominator for representative velocity

    # Compute u_repr for segments, then align back to node indices
    with np.errstate(divide='ignore', invalid='ignore'):
        u_seg = q_trans / denom  # (n-1,)
    
    # Mask bad and any computed NaNs
    mask = (bad_segments_ind | ~np.isfinite(u_seg)).compute()
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
