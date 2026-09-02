"""Styling/config dataclasses for the dfasttf plot modules.

Centralizes labels, colors, tick spacing and other constants used across the
1D, 2D, ice and cross-flow plotting modules so styling only needs to change
in one place.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Plot1DConfig:
    XLABEL: str = "raai km"
    DELTARES_BLUE = "#0D38DF"
    DELTARES_DARKGREEN = "#00B389"
    COLORS = (
        "k",
        DELTARES_BLUE,
        DELTARES_DARKGREEN,
    )  # reference, intervention, difference
    LABELS = ["Referentie", "Plansituatie", "Verschil"]


@dataclass
class FlowfieldConfig:
    VELOCITY_YLABEL: str = "stroomsnelheid\nmagnitude [m/s]"
    VELOCITY_DIFF_YLABEL: str = "verschil plansituatie\n-referentie [m/s]"
    VELOCITY_YLIM: tuple = (0.0, 0.5)
    VELOCITY_YTICKS_MAJOR: float = 0.1
    VELOCITY_YTICKS_MINOR: float = 0.05
    ANGLE_YTICKS_MAJOR: float = 30.0
    ANGLE_YTICKS_MINOR: float = 10.0
    ANGLE_YLIM: tuple = (-90.0, 90.0)
    ANGLE_PRIMARY_YLABEL: str = "stromingshoek t.o.v.\nprofiellijn [graden]"
    ANGLE_DIFF_YLABEL: str = "verschil plansituatie\n-referentie [graden]"
    FRACTION: float = 5.0


@dataclass
class FroudeConfig:
    profile_line_color: str = "black"
    legend_title: str = "Verschil tussen plan-\nsituatie en referentie"

    class Abs:
        colorbar_label: str = "Froude getal"
        levels: tuple = (0, 0.08, 0.1, 0.15)
        colormap: str = "RdBu"

    class Diff:
        bins: list = [0, 0.08, 0.1, 0.15, np.inf]
        colors = ("green", "red", "blue")
        labels: list[str] = [
            f"Fr van < {bins[1]} naar >= {bins[1]}",
            f"Fr van > {bins[1]} naar <= {bins[1]}",
            "droog in referentie,\nnat in plansituatie"
        ]


@dataclass
class CrossFlowConfig:
    XLABEL = Plot1DConfig.XLABEL
    YLABEL: str = "representatieve dwars-\nstroomsnelheid [m/s]"
    DIFF_YLABEL: str = FlowfieldConfig.VELOCITY_DIFF_YLABEL
    EBB_TITLE: str = "Bij piek ebstroming (per cel)"
    FLOOD_TITLE: str = "Bij piek vloedstroming (per cel)"
    CRIT_LABEL: str = "Lokaal criterium"
    YLIM: tuple = (-0.3, 0.3)
    FRACTION: int = 3
    YTICKS_MAJOR = 0.15
    YTICKS_MINOR = 0.05


@dataclass
class DirectionalMaximaConfig:
    """Styling for the bankward/riverward directional-maxima figures.

    Per profile position these figures show the maximum transverse velocity
    in one direction (top row) together with the instantaneous transverse
    discharge at that same moment (bottom row) (see RBK review: velocity and
    discharge must be taken at the same phase to be physically comparable).
    Bankward and riverward are rendered as two separate figures.
    """
    XLABEL = Plot1DConfig.XLABEL
    VELOCITY_YLABEL: str = "max. representatieve\ndwarsstroomsnelheid [m/s]"
    DISCHARGE_YLABEL: str = "instantaan dwars-\nstroomdebiet [m³/s]"
    VELOCITY_DIFF_YLABEL: str = "verschil plansituatie\n-referentie [m/s]"
    DISCHARGE_DIFF_YLABEL: str = "verschil plansituatie\n-referentie [m³/s]"
    BANKWARD_VELOCITY_TITLE: str = "Max. snelheid naar de oever (per cel)"
    BANKWARD_DISCHARGE_TITLE: str = "Debiet bij max. snelheid naar de oever (per cel)"
    RIVERWARD_VELOCITY_TITLE: str = "Max. snelheid naar de rivier (per cel)"
    RIVERWARD_DISCHARGE_TITLE: str = "Debiet bij max. snelheid naar de rivier (per cel)"

