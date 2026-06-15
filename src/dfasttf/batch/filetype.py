
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class FileKind(Enum):
    MAP = auto()       # has time dimension
    FOU = auto()       # no time dimension, but supports original analysis
    INVALID = auto()   # no time, missing required variables


@dataclass(frozen=True)
class FileInfo:
    kind: FileKind
    has_time: bool
    has_ucxy: bool
    has_depth: bool
    missing: tuple[str, ...] = ()
    vars_sample: tuple[str, ...] = ()


def _standard_names_present(ds) -> set[str]:
    """Return the set of CF standard_name values present in the dataset."""
    return {
        ds[var].attrs.get("standard_name")
        for var in ds.data_vars
        if ds[var].attrs.get("standard_name") is not None
    }


def detect_file_info(ds) -> FileInfo:
    """
    Detect file type based on dataset content (not filename).

    Rules
    -----
    - If dataset has a 'time' dimension -> MAP
    - If dataset has no 'time' dimension, it must still contain the variables
      required for the original analysis:
        * sea_water_x_velocity
        * sea_water_y_velocity
        * sea_surface_height
        * altitude
      Otherwise -> INVALID
    """
    has_time = "time" in ds.dims
    std_names = _standard_names_present(ds)
    vars_sample = tuple(sorted(ds.data_vars)[:30])

    required_ucxy = {
        "sea_water_x_velocity",
        "sea_water_y_velocity",
    }
    required_depth = {
        "sea_surface_height",
        "altitude",
    }

    has_ucxy = required_ucxy.issubset(std_names)
    has_depth = required_depth.issubset(std_names)

    missing = []
    if not has_ucxy:
        missing.append("sea_water_x_velocity + sea_water_y_velocity")
    if not has_depth:
        missing.append("sea_surface_height + altitude")

    if has_time:
        return FileInfo(
            kind=FileKind.MAP,
            has_time=True,
            has_ucxy=has_ucxy,
            has_depth=has_depth,
            missing=(),
            vars_sample=vars_sample,
        )

    if missing:
        return FileInfo(
            kind=FileKind.INVALID,
            has_time=False,
            has_ucxy=has_ucxy,
            has_depth=has_depth,
            missing=tuple(missing),
            vars_sample=vars_sample,
        )

    return FileInfo(
        kind=FileKind.FOU,
        has_time=False,
        has_ucxy=True,
        has_depth=True,
        missing=(),
        vars_sample=vars_sample,
    )