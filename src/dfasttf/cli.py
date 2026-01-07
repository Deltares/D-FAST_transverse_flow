import sys
import argparse
from pathlib import Path
from dfasttf.cmd import run
from dfasttf import __version__
from dfasttf import __path__

DATA_PATH = Path(__path__[0]) / "data"

DEFAULT_SHIP_DIMENSIONS = DATA_PATH / "ship_dimensions.ini"
DUTCH_RIVERS_INI = DATA_PATH / "dutch_rivers.ini"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse the command line arguments.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command line arguments to parse. If None, uses sys.argv.
        This is useful for testing.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing:
        - config : str
            Path to the analysis configuration file
        - rivers : str
            Path to the rivers configuration file (currently unused)
        - ships : str
            Path to the ship dimensions file
    """
    parser = argparse.ArgumentParser(
        prog="dfasttf",
        description="D-FAST Transverse Flow - Analyze transverse flow patterns in rivers"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="path to analysis configuration file (required)",
    )

    parser.add_argument(
        "--rivers",
        default=DUTCH_RIVERS_INI,
        help=f"path to rivers configuration file (default: {DUTCH_RIVERS_INI.name})",
    )

    parser.add_argument(
        "--ships",
        default=DEFAULT_SHIP_DIMENSIONS,
        help=f"path to ship dimensions file (default: {DEFAULT_SHIP_DIMENSIONS.name})",
    )

    args = parser.parse_args(argv)
    return args


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """
    Validate command line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments

    Returns
    -------
    tuple[Path, Path, Path]
        Tuple of (config_file, rivers_file, ships_file) as Path objects

    Raises
    ------
    FileNotFoundError
        If any of the required files do not exist
    ValueError
        If any file paths are invalid
    """
    # Validate config file
    config_file = Path(args.config)
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_file.resolve()}"
        )
    if not config_file.is_file():
        raise ValueError(
            f"Configuration path is not a file: {config_file.resolve()}"
        )

    # Validate rivers file
    rivers_file = Path(args.rivers)
    if not rivers_file.exists():
        raise FileNotFoundError(
            f"Rivers configuration file not found: {rivers_file.resolve()}"
        )
    if not rivers_file.is_file():
        raise ValueError(
            f"Rivers configuration path is not a file: {rivers_file.resolve()}"
        )

    # Validate ships file
    ships_file = Path(args.ships)
    if not ships_file.exists():
        raise FileNotFoundError(
            f"Ship dimensions file not found: {ships_file.resolve()}"
        )
    if not ships_file.is_file():
        raise ValueError(
            f"Ship dimensions path is not a file: {ships_file.resolve()}"
        )

    return config_file, rivers_file, ships_file


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for the D-FAST Transverse Flow CLI.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command line arguments to parse. If None, uses sys.argv.
        This is useful for testing.

    Returns
    -------
    int
        Exit code: 0 for success, non-zero for failure
    """
    try:
        args = parse_arguments(argv)
        config_file, rivers_file, ships_file = validate_args(args)

        run(config_file, ships_file)

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())