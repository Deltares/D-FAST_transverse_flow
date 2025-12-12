import numpy as np


def face_len(face_x_bnd: np.ndarray, face_y_bnd: np.ndarray) -> np.ndarray:
    """
    Calculates the length of mesh faces.
    face_x_bnd: x-coordinates of the face boundaries
    face_y_bnd: y-coordinates of the face boundaries
    """
    dx = np.gradient(face_x_bnd.values, axis=0)
    dy = np.gradient(face_y_bnd.values, axis=0)
    face_len = np.sqrt(dx**2 + dy**2)  # length of faces
    return face_len


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


# def representative_trans_velocity(face_len: np.ndarray,
#                         water_depth: np.ndarray,
#                         trans_velocity: np.ndarray,
#                         SHIP_DEPTH: float) -> np.ndarray:
#     """
#     Calculates the representative transversal velocity at mesh faces according to RBK specifications.
#     """

#     Q_trans =  water_depth * face_len * trans_velocity # transversal discharge
#     urepr = Q_trans / (face_len * np.fmax(water_depth, SHIP_DEPTH)) # representative transversal velocity

#     return urepr


def trans_discharge(u_integral: np.ndarray, SHIP_DEPTH: float) -> np.ndarray:
    """
    Calculates the transversal discharge.
    u_integral = integral of flow velocity over cross-sectional width
    SHIP_DEPTH: depth of the ship
    """

    q = u_integral * SHIP_DEPTH
    return q
