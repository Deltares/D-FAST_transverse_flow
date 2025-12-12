import logging

from dfastrbk.src.batch.core import preprocess_1d, run_analysis
from dfastrbk.src.batch.dflowfm import Variables
from dfastrbk.src.config import Config

logging.basicConfig(filename="dfastrbk.log", level=logging.INFO)
logger = logging.getLogger(__name__)


# TODO: make figfiles optional, now depends on SavePlots=True
def run(config_file: str, ships_file: str) -> None:
    """Main entry point for running the analysis."""
    logger.info("Running analysis...")

    configuration = Config(config_file, ships_file)

    variables = Variables(
        h="mesh2d_waterdepth",
        uc="mesh2d_ucmag",
        ucx="mesh2d_ucx",
        ucy="mesh2d_ucy",
        bl="mesh2d_flowelem_bl",
    )

    prof_line_df = None
    prof_line_df, riverkm = preprocess_1d(configuration)

    for section in configuration.keys():
        if "Reference" in configuration.config[section]:
            run_analysis(configuration, section, variables, prof_line_df, riverkm)

    logger.info("Finished analysis.")
