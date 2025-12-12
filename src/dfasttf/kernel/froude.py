"""Performs corrections on the modelled Froude number specific for the discharge of ice """

import numpy as np
from xarray import DataArray


def calculate_froude_number(
    water_depth: DataArray, flow_velocity: DataArray, grav_constant=9.81
) -> DataArray:
    """Calculates the Froude number from flow velocity and water depth"""
    froude_number = flow_velocity / np.sqrt(grav_constant * water_depth)
    return froude_number


def water_uplift(froude_number: DataArray) -> DataArray:
    """correction from water level uplift due to downstream ice cover"""
    froude_corrected = froude_number / np.sqrt(2)
    return froude_corrected


def bed_change(
    froude_number: DataArray, bedlevel_change: DataArray, water_depth: DataArray
) -> DataArray:
    """correction from bed level change due to the measure (as calculated by D-FAST-MI)"""
    # requires change in bed level calculated with d-fast-mi
    correction_term = (1 - bedlevel_change / water_depth) ** (-1.5)
    froude_corrected = froude_number * correction_term
    return froude_corrected
