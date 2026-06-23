
import math
import numpy as np


def on_right_side(line_xy: np.ndarray, ref_xy: np.ndarray) -> bool:
    """
    Determine whether line_xy is to the right of ref_xy.

    Left and right are defined relative to the direction of ref_xy from
    its first node to its last node.

    Assumptions
    -----------
    - line_xy and ref_xy do not cross each other
    - neither line crosses itself
    - line_xy lies alongside ref_xy, not before or after it

    Parameters
    ----------
    line_xy : np.ndarray
        Array of shape (N, 2) containing x,y coordinates of the line.
    ref_xy : np.ndarray
        Array of shape (M, 2) containing x,y coordinates of the reference line.

    Returns
    -------
    bool
        True if line_xy lies on the right side of ref_xy, False otherwise.
    """
    ref_npnt = ref_xy.shape[0]
    npnt = line_xy.shape[0]

    if ref_npnt < 2:
        raise ValueError("Reference line must contain at least two points.")

    # determine the reference point based on the line with the fewest points
    if ref_npnt < npnt:
        if ref_npnt == 2:
            imin = 0
            imind = 0
            iminu = 1
            p0 = (ref_xy[0] + ref_xy[1]) / 2
        else:
            imin = int(ref_npnt / 2)
            imind = imin - 1
            iminu = imin + 1
            p0 = ref_xy[imin]

        # find the node on line_xy closest to p0
        hpnt = np.argmin(((p0 - line_xy) ** 2).sum(axis=1))
        hpxy = line_xy[hpnt]
    else:
        # determine the mid-point hpxy of line_xy
        hpnt = int(npnt / 2)
        hpxy = line_xy[hpnt]

        # find the node on ref_xy closest to hpxy
        imin = np.argmin(((hpxy - ref_xy) ** 2).sum(axis=1))
        imind = imin - 1
        iminu = imin + 1
        p0 = ref_xy[imin]

    # direction to the midpoint of line_xy
    theta = math.atan2(hpxy[1] - p0[1], hpxy[0] - p0[0])

    # direction from which ref_xy comes
    if imin > 0:
        phi1 = math.atan2(ref_xy[imind, 1] - p0[1], ref_xy[imind, 0] - p0[0])
        # direction to which ref_xy goes
        if imin < ref_xy.shape[0] - 1:
            phi2 = math.atan2(ref_xy[iminu, 1] - p0[1], ref_xy[iminu, 0] - p0[0])
        else:
            phi2 = -phi1
    else:
        phi2 = math.atan2(ref_xy[iminu, 1] - p0[1], ref_xy[iminu, 0] - p0[0])
        phi1 = -phi2

    # adjust directions of ref_xy such that both are larger than theta
    if phi1 < theta:
        phi1 += 2 * math.pi
    if phi2 < theta:
        phi2 += 2 * math.pi

    # theta points to the right if we encounter phi2 before phi1
    # when rotating counter-clockwise starting from theta
    right_side = phi2 < phi1

    return right_side
