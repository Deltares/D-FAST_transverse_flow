import numpy as np


def sort_ref_by_chainage(ref_coords: np.ndarray) -> np.ndarray:
    """
    Sort reference line by increasing chainage.

    Expected input:
    - shape (n, 3): x, y, chainage
    - shape (n, 2): x, y, assumed already in downstream order
    """
    ref_coords = np.asarray(ref_coords, dtype=float)

    if ref_coords.shape[1] >= 3:
        order = np.argsort(ref_coords[:, 2])
        return ref_coords[order, :2]

    return ref_coords[:, :2]


def project_points_to_polyline(
    points_xy: np.ndarray,
    ref_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project each point to the nearest segment of a reference polyline.

    Returns
    -------
    projected_xy : np.ndarray
        Coordinates of nearest projected points on the reference line, shape (n, 2).
    segment_idx : np.ndarray
        Index of nearest reference segment for each point.
    """
    points_xy = np.asarray(points_xy, dtype=float)
    ref_xy = np.asarray(ref_xy, dtype=float)

    if ref_xy.shape[0] < 2:
        raise ValueError("Reference line must contain at least two points.")

    seg_start = ref_xy[:-1]
    seg_end = ref_xy[1:]
    seg_vec = seg_end - seg_start
    seg_len2 = np.sum(seg_vec * seg_vec, axis=1)

    if np.any(seg_len2 == 0):
        raise ValueError("Reference line contains zero-length segments.")

    projected_xy = np.empty_like(points_xy, dtype=float)
    segment_idx = np.empty(points_xy.shape[0], dtype=int)

    for i, p in enumerate(points_xy):
        rel = p - seg_start
        frac = np.sum(rel * seg_vec, axis=1) / seg_len2
        frac = np.clip(frac, 0.0, 1.0)

        proj = seg_start + frac[:, np.newaxis] * seg_vec
        dist2 = np.sum((proj - p) ** 2, axis=1)

        j = int(np.argmin(dist2))
        projected_xy[i] = proj[j]
        segment_idx[i] = j

    return projected_xy, segment_idx


def bankward_normal_sign(
    profile_angles: np.ndarray,
    sample_points_xy: np.ndarray,
    riverkm_coords: np.ndarray,
) -> np.ndarray:
    """
    Determine the sign needed to orient transverse velocity such that:

    positive = towards river axis
    negative = towards bank

    The check is performed at the same points where transverse velocity is
    evaluated: the ordered mesh/profile intersection points.

    Downstream direction is defined as increasing river kilometre / chainage.

    Returns
    -------
    np.ndarray
        Sign array with shape (n,). Multiply raw transverse velocity by this sign.
    """
    profile_angles = np.asarray(profile_angles, dtype=float)
    sample_points_xy = np.asarray(sample_points_xy, dtype=float)

    if sample_points_xy.shape[0] != profile_angles.shape[0]:
        raise ValueError(
            "profile_angles and sample_points_xy must have the same length."
        )

    ref_xy = sort_ref_by_chainage(riverkm_coords)
    projected_xy, _ = project_points_to_polyline(sample_points_xy, ref_xy)

    # Vector from river axis to sample point.
    # This is the local bankward direction.
    bank_vec = sample_points_xy - projected_xy

    # Positive normal used by flow.trans_velocity():
    # w = u * (-sin(theta)) + v * cos(theta)
    theta = np.radians(profile_angles)
    normal_xy = np.column_stack((-np.sin(theta), np.cos(theta)))

    dot = np.sum(normal_xy * bank_vec, axis=1)

    # Negate: `dot` aligns with the bankward direction, but the convention
    # used throughout the tool is positive = towards the river axis.
    sign = -np.sign(dot)
    sign[sign == 0] = -1.0

    return sign
