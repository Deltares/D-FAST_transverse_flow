import sys
import argparse
from pathlib import Path
from dfasttf.cmd import run
from dfasttf import __version__
from dfasttf import __path__

DATA_PATH = Path(__path__[0]) / "data"

DEFAULT_SHIP_DIMENSIONS = DATA_PATH / "ship_dimensions.ini"
DUTCH_RIVERS_INI = DATA_PATH / "dutch_rivers.ini"


def parse_arguments() -> tuple:
    """
    Parse the command line arguments.

    Arguments
    ---------
    None

    Returns
    -------
    config_name : Optional[str]
        Name of the analysis configuration file.
    rivers_file : str
        Name of rivers configuration file.
    """

    parser = argparse.ArgumentParser(description="D-FAST-RBK")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    parser.add_argument(
        "--config",
        default="unspecified",
        help="name of analysis configuration file ('%(default)s' is default)",
    )

    parser.add_argument(
        "--rivers",
        default=DUTCH_RIVERS_INI,
        help="name of rivers configuration file ('Dutch_rivers_v3.ini' is default)",
    )

    parser.add_argument(
        "--ships",
        default=DEFAULT_SHIP_DIMENSIONS,
        help="name of ship dimensions file ('ship_dimensions.ini' is default)",
    )

    parser.set_defaults(reduced_output=False)
    args = parser.parse_args()

    return args


def validate_args(args):
    config_file = args.config
    rivers_file = args.rivers
    ships_file = args.ships

    if config_file == "unspecified":
        config_file = "examples/c01 - Waal/config.ini"

    return config_file, rivers_file, ships_file

def main(argv: list[str] | None = None) -> int:
    args = parse_arguments()
    config_file, rivers_file, ships_file = validate_args(args)
    run(config_file, ships_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())