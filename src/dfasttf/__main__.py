import argparse

import dfastrbk.src.cmd


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
        "--config",
        default="unspecified",
        help="name of analysis configuration file ('%(default)s' is default)",
    )

    parser.add_argument(
        "--rivers",
        default="unspecified",
        help="name of rivers configuration file ('Dutch_rivers_v3.ini' is default)",
    )

    parser.add_argument(
        "--ships",
        default="unspecified",
        help="name of ship dimensions file ('ship_dimensions.ini' is default)",
    )

    parser.set_defaults(reduced_output=False)
    args = parser.parse_args()

    config_file = args.__dict__["config"]
    rivers_file = args.__dict__["rivers"]
    ships_file = args.__dict__["ships"]
    if config_file == "unspecified":
        config_file = "examples/c01 - Waal/config.ini"
    if rivers_file == "unspecified":
        rivers_file = "Dutch_rivers_v3.ini"
    if ships_file == "unspecified":
        # TODO: fix this path
        ships_file = "dfastrbk/src/ship_dimensions.ini"

    return config_file, rivers_file, ships_file


if __name__ == "__main__":
    config_file, rivers_file, ships_file = parse_arguments()
    dfastrbk.src.cmd.run(config_file, ships_file)
