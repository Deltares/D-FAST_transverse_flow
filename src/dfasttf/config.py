from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path

from dfastio import xyc
from shapely import LineString

from dfastmi.batch.core import _get_output_dir
from dfastmi.batch.PlotOptions import PlotOptions
from dfastmi.config.ConfigFileOperations import ConfigFileOperations
from dfastmi.io.DFastAnalysisConfigFileParser import DFastAnalysisConfigFileParser

GENERAL_SECTION = "General"
BOUNDING_BOX_SECTION = "BoundingBox"
BEDCHANGEFILE_KEY = "BedChangeFile"
WITHINTERVENTION_KEY = "WithIntervention"


@dataclass
class Ship:
    length: float
    depth: float

    @classmethod
    def from_config(cls, reach: str, ships_file: Path) -> "Ship":
        config = ConfigParser()
        config.read(ships_file)
        try:
            length = float(config[reach]["Length"])
            depth = float(config[reach]["Depth"])
        except KeyError as e:
            raise ValueError(f"Missing key in ships file for reach '{reach}': {e}")
        return cls(length=length, depth=depth)


class Config:
    """
    Loads and manages configuration for D-FAST analysis.
    """

    def __init__(self, config_file: str, ships_file: str):
        configfile = Path(config_file).resolve()
        self.configdir = configfile.parent

        self.config = ConfigFileOperations.load_configuration_file(str(config_file))
        self.keys = self.config.keys

        self.data = DFastAnalysisConfigFileParser(self.config)
        self.general = GeneralSettings.from_config(
            self.data, self.config, self.configdir
        )
        self.outputdir = _get_output_dir(str(self.configdir), True, self.data)

        shipsfile = Path(ships_file).resolve()
        self.ship_params = Ship.from_config(self.general.reach, shipsfile)

        self.plotsettings = PlotSettings(self.configdir, self.data)


def get_output_files(config: ConfigParser, configdir: Path, section: str):
    """
    Adds output files from config file section to configuration.
    """
    output_files = []

    reference_file = config.get(section, "Reference")
    output_files.append(
        ConfigFileOperations._get_absolute_path_from_relative_path(
            str(configdir), reference_file
        )
    )

    if WITHINTERVENTION_KEY in config[section]:
        with_intervention = config.get(section, "WithIntervention")
        output_files.append(
            ConfigFileOperations._get_absolute_path_from_relative_path(
                str(configdir), with_intervention
            )
        )
    return output_files


@dataclass
class GeneralSettings:
    """Sets the general settings"""

    branch: str
    reach: str
    bool_flags: dict
    riverkm: LineString | None
    profiles_file: Path | None
    bedchangefile: Path | None
    bbox: list | None

    @classmethod
    def from_config(
        cls, data: DFastAnalysisConfigFileParser, config: ConfigParser, configdir: Path
    ) -> "GeneralSettings":
        reach = data.getstring(GENERAL_SECTION, "Reach")
        branch = data.getstring(GENERAL_SECTION, "Branch")

        bool_flags = {
            flag.lower(): data.getboolean(GENERAL_SECTION, flag, fallback=False)
            for flag in ["InvertXAxis", "WaterUpliftCorrection", "BedChangeCorrection"]
        }

        riverkm = None
        riverkm_file = data.getstring(GENERAL_SECTION, "RiverKM")
        riverkm = xyc.models.XYCModel.read(riverkm_file, num_columns=3)

        profiles_file = None
        profiles_file = Path(
            ConfigFileOperations._get_absolute_path_from_relative_path(
                str(configdir), data.getstring(GENERAL_SECTION, "ProfileLines")
            )
        )

        bedchangefile = None
        if BEDCHANGEFILE_KEY in config[GENERAL_SECTION]:
            bedchangefile = Path(
                ConfigFileOperations._get_absolute_path_from_relative_path(
                    str(configdir), data.getstring(GENERAL_SECTION, BEDCHANGEFILE_KEY)
                )
            )

        bbox = None
        if BOUNDING_BOX_SECTION in config:
            bbox = [
                float(config[BOUNDING_BOX_SECTION][key])
                for key in config[BOUNDING_BOX_SECTION]
            ]

        return cls(
            branch=branch,
            reach=reach,
            bool_flags=bool_flags,
            riverkm=riverkm,
            profiles_file=profiles_file,
            bedchangefile=bedchangefile,
            bbox=bbox,
        )


class PlotSettings:
    def __init__(self, config_dir: Path, data: DFastAnalysisConfigFileParser):
        self.type = data.getstring(GENERAL_SECTION, "PlotType", "both")
        self.options = PlotOptions()
        self.options.set_plotting_flags(config_dir, False, data)
