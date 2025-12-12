import math

import numpy as np

# def append_array_roots(x: np.ndarray, y: np.ndarray) -> tuple:
#     """
#     Interpolate arrays and append the roots (zero-crossings)
#     """

#     s = np.abs(np.diff(np.sign(y))).astype(bool)
#     z = x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1) # x-position of zero-crossings, found by linear interpolation

#     x_appended = np.concatenate([x,z],axis=0)
#     y_appended = np.concatenate([y,np.zeros(len(z))],axis=0)
#     x_sorted = np.sort(x_appended)
#     sort_idx = np.argsort(x_appended)
#     y_sorted = np.take_along_axis(y_appended,sort_idx,axis=0)

#     # # Make sure there are zero crossings at the beginning and end
#     # if (y_appended[0] > almost_zero) | (y_appended[0] < -almost_zero):
#     #     y_appended = np.insert(y_appended,0,0,axis=0)
#     #     x_appended = np.insert(x_appended,0,x_appended[0],axis=0)

#     # if (y_appended[-1] > almost_zero) | (y_appended[-1] < -almost_zero):
#     #     y_appended = np.insert(y_appended,-1,0,axis=0)
#     #     x_appended = np.insert(x_appended,-1,x_appended[-1],axis=0)

#     return x_sorted, y_sorted


def insert_array_roots(
    x: np.ndarray, y: np.ndarray, x2: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Interpolate arrays and append the roots (zero-crossings)
    """
    z = find_roots(x, y)
    # Insert zero-crossings to the original arrays
    # TODO: there's a small bug where the zero-crossing is inserted in front of an element with the same x
    idx = x.searchsorted(z, side="right")
    x_mod = np.insert(x, idx, z)
    y_mod = np.insert(y, idx, 0)
    if x2 is not None:
        z1 = find_roots(x2, y)
        x2_mod = np.insert(x2, idx, z1)
    return x_mod, y_mod, x2_mod


def find_roots(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Finds the x-position of zero-crossings"""
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return x[:-1][s] + np.diff(x)[s] / (np.abs(y[1:][s] / y[:-1][s]) + 1)


def split_into_blocks(
    x: np.ndarray, y: np.ndarray, x2: np.ndarray | None = None
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray] | None]:
    """Splits x and y into blocks, separated by 0 in y."""
    x_split = []
    y_split = []
    if x2 is not None:
        x2_split = []
    zero_ind = np.where(y == 0)[0]

    for i in range(len(zero_ind) - 1):
        start = zero_ind[i]
        end = zero_ind[i + 1] + 1
        x_split.append(x[start:end])
        y_split.append(y[start:end])
        if x2 is not None:
            x2_split.append(x2[start:end])

    return x_split, y_split, x2_split


def max_rolling_integral(x: np.ndarray, y: np.ndarray, window: float) -> tuple:
    """
    Maximum absolute integral over forward rolling windows with width at most window,
    without interpolation, and allowing duplicates in x.

    The window for start index i includes indices [i, j] where j is the largest
    index such that x[j] - x[i] <= window (i.e., the width is not yet exceeded).
    The integral is computed via the trapezoidal rule using *only* the fully
    included segments (no partial last segment interpolation).

    If the full range x[-1] - x[0] < window, the integral is taken over the entire arrays.

    Parameters
    ----------
    x : (n,) array_like
        x-coordinates (can be non-strictly increasing; duplicates allowed).
    y : (n,) array_like
        values at x
    window : float
        window width in x-units (must be > 0).

    Returns
    -------
    max_abs_area : float
        Maximum absolute integral over any window with width <= window (or full array if range < window).
    best_i : int
        Start index of the window achieving max_abs_area.
    best_j : int
        End index of the window achieving max_abs_area.

    Notes
    -----
    - No interpolation is performed. The end index j is chosen such that
      x[j] - x[i] <= window and including j+1 would exceed window.
    - Duplicates in x cause zero-width segments; they add no area and do not
      affect the width, but they can be included if they fit within the bound.
    - If n < 2, area is 0 and (0, 0) is returned for indices.
    - Time complexity: O(n). Memory: O(n).
    """

    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1-D arrays of the same length.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must be finite.")
    if window <= 0:
        raise ValueError("Window width window must be positive.")

    n = x.size
    if n <= 1:
        return 0.0, [0, 0]

    # If whole range is smaller than window, integrate over all data
    full_range = x[-1] - x[0]
    dx = np.diff(x)
    # (Strictly smaller than window per requirement; if equal, the algorithm naturally picks [0, n-1])
    if full_range < window:
        seg = 0.5 * (y[1:] + y[:-1]) * dx
        total_area = np.sum(seg)
        return abs(float(total_area)), [0, n - 1]

    # Precompute cumulative trapezoidal integral: F[k] = ∫ from x[0] to x[k] (using full segments)
    seg = 0.5 * (y[1:] + y[:-1]) * dx  # zero when dx == 0
    F = np.concatenate(([0.0], np.cumsum(seg)))

    max_abs_area = -np.inf
    best_i = 0
    best_j = 0

    j = 0  # right boundary pointer
    for i in range(n):
        j = max(j, i)
        # Advance j while including next point does NOT exceed width window
        # i.e., keep x[j+1] - x[i] <= window
        while j + 1 < n and (x[j + 1] - x[i]) <= window:
            j += 1

        # Area over [i, j] via cumulative difference; no partial last segment
        area_signed = F[j] - F[i]
        area_abs = abs(area_signed)

        if area_abs > max_abs_area:
            max_abs_area = area_abs
            best_i = i
            best_j = j

    return float(max_abs_area), [int(best_i), int(best_j)]


def densify_array(x: np.ndarray, max_step: float) -> np.ndarray:
    """
    Return a densified x array such that np.diff(x_new) <= max_step by inserting
    intermediate points between neighbors with too-large gaps.

    Parameters
    ----------
    x : (n,) array_like
        Input x-values. Intended to be non-decreasing; duplicates allowed.
    max_step : float, optional
        Maximum allowed step between consecutive x-values (default 0.5). Must be > 0.

    Returns
    -------
    x_new : (m,) ndarray
        Densified x array including all original points (unless keep_duplicates=False),
        with np.diff(x_new) <= max_step (up to tiny float roundoff).

    Notes
    -----
    - Duplicates (gap == 0) are fine; they don't add area and don't require inserts.
    - For each gap g = x[i+1] - x[i] > 0, we insert n_add = ceil(g/max_step) - 1 points,
      placed with np.linspace so the new max sub-step is <= max_step.
    - Time complexity: O(n); memory: O(n + #inserted).
    """
    if x.ndim != 1:
        raise ValueError("x must be a 1-D array.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values.")
    if max_step <= 0:
        raise ValueError("max_step must be positive.")

    if x.size <= 1:
        return x.copy()

    parts = [x[0:1]]
    for a, b in zip(x[:-1], x[1:]):
        gap = b - a
        if gap <= 0:
            # zero or negative (duplicate or decreasing segment) — just append b
            parts.append(np.array([b]))
            continue

        # Number of extra points to insert to ensure sub-step <= max_step
        n_add = max(0, int(math.ceil(gap / max_step) - 1))
        if n_add > 0:
            # Place interior points evenly, excluding endpoints a and b
            mids = np.linspace(a, b, n_add + 2, endpoint=True)[1:-1]
            parts.append(mids)
        parts.append(np.array([b]))

    x_new = np.concatenate(parts)
    return x_new
