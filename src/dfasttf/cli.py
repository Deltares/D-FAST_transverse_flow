import sys
import argparse
from pathlib import Path

import matplotlib

# Force a non-interactive backend before any submodule imports matplotlib.pyplot,
# since this is a batch/report-generation tool with no GUI. Without this, the
# default backend on some systems can attempt to open a window per saved figure.
matplotlib.use("Agg")

from dfasttf.cmd import run
from dfasttf import __version__
from dfasttf import __path__

DATA_PATH = Path(__path__[0]) / "data"

DEFAULT_SHIP_DIMENSIONS = DATA_PATH / "ship_dimensions.ini"
DUTCH_RIVERS_INI = DATA_PATH / "dutch_rivers.ini"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command line arguments for D-FAST Transverse Flow analysis.

    This function parses command line arguments to configure a transverse flow
    analysis session. It requires a configuration file path and optionally accepts
    paths to rivers and ship dimensions configuration files.

    Args:
        argv: Command line arguments to parse. If None, uses sys.argv.
            Useful for testing and programmatic invocation. Should contain
            arguments in the format: ['--config', 'path/to/config.ini', ...]

    Returns:
        Parsed arguments as a namespace object with the following attributes:
            - config (str): Path to the analysis configuration file
            - rivers (str | Path): Path to the rivers configuration file
            - ships (str | Path): Path to the ship dimensions file

    Examples:
        - Parse arguments with only required config file:
            >>> args = parse_arguments(['--config', 'examples/c04-nvo-maas/config.ini'])
            >>> args.config
            'examples/c04-nvo-maas/config.ini'

        - Check default ship dimensions file path contains expected filename:
            >>> args = parse_arguments(['--config', 'examples/c04-nvo-maas/config.ini'])
            >>> from pathlib import Path
            >>> Path(str(args.ships)).name
            'ship_dimensions.ini'

        - Check default rivers configuration file path contains expected filename:
            >>> args = parse_arguments(['--config', 'examples/c04-nvo-maas/config.ini'])
            >>> from pathlib import Path
            >>> Path(str(args.rivers)).name
            'dutch_rivers.ini'

        - Parse arguments with custom ships file:
            >>> args = parse_arguments([
            ...     '--config', 'examples/c04-nvo-maas/config.ini',
            ...     '--ships', 'custom_ships.ini'
            ... ])
            >>> args.ships
            'custom_ships.ini'

        - Parse arguments with custom rivers file:
            >>> args = parse_arguments([
            ...     '--config', 'examples/c04-nvo-maas/config.ini',
            ...     '--rivers', 'custom_rivers.ini'
            ... ])
            >>> args.rivers
            'custom_rivers.ini'

        - Verify all three arguments are present:
            >>> args = parse_arguments(['--config', 'test.ini'])
            >>> hasattr(args, 'config') and hasattr(args, 'rivers') and hasattr(args, 'ships')
            True

    See Also:
        validate_args: Validates the parsed arguments and checks file existence.
        main: Main entry point that uses this parser.
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
    Validate command line arguments and ensure all required files exist.

    This function checks that all file paths provided in the arguments exist
    and are valid files (not directories). It converts string paths to Path
    objects for further processing.

    Args:
        args: Parsed command line arguments from parse_arguments containing
            config, rivers, and ships file paths as strings.

    Returns:
        A tuple of three Path objects in the order:
            (config_file, rivers_file, ships_file)

    Raises:
        FileNotFoundError: If any of the required files do not exist.
            The error message includes the absolute path of the missing file.
        ValueError: If any of the paths exist but are not files (e.g., directories).
            The error message includes the absolute path of the invalid path.

    Examples:
        - Validate arguments for the NVO Maas example:
            >>> from pathlib import Path
            >>> args = parse_arguments(['--config', 'examples/c04-nvo-maas/config.ini'])
            >>> config, rivers, ships = validate_args(args)
            >>> isinstance(config, Path)
            True
            >>> isinstance(rivers, Path)
            True
            >>> isinstance(ships, Path)
            True

        - Check that config file path is correct:
            >>> args = parse_arguments(['--config', 'examples/c04-nvo-maas/config.ini'])
            >>> config, rivers, ships = validate_args(args)
            >>> config.name
            'config.ini'

        - Check that default ships file is validated:
            >>> args = parse_arguments(['--config', 'examples/c04-nvo-maas/config.ini'])
            >>> config, rivers, ships = validate_args(args)
            >>> ships.name
            'ship_dimensions.ini'

        - Check that all returned paths are Path objects:
            >>> args = parse_arguments(['--config', 'examples/c04-nvo-maas/config.ini'])
            >>> config, rivers, ships = validate_args(args)
            >>> all(isinstance(p, Path) for p in [config, rivers, ships])
            True

        - Verify files exist and are files:
            >>> args = parse_arguments(['--config', 'examples/c04-nvo-maas/config.ini'])
            >>> config, rivers, ships = validate_args(args)
            >>> config.is_file() and rivers.is_file() and ships.is_file()
            True

    See Also:
        parse_arguments: Parses command line arguments before validation.
        main: Main function that calls both parse_arguments and validate_args.
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
    r"""
    Main entry point for the D-FAST Transverse Flow CLI.

    This function orchestrates the entire command-line interface workflow:
    parsing arguments, validating file paths, and executing the analysis.
    It handles errors gracefully and returns appropriate exit codes.

    Args:
        argv: Command line arguments to parse. If None, uses sys.argv.
            Useful for testing and programmatic invocation. Should follow
            the format: ['--config', 'path/to/config.ini', ...]

    Returns:
        Exit code indicating the result of execution:
            - 0: Success - analysis completed without errors
            - 1: Failure - file not found, validation error, or unexpected error
            - 130: Cancelled - user interrupted with Ctrl+C (SIGINT)

    Examples:
        - Run analysis with the NVO Maas example configuration:
            ```bash
            >>> dfasttf --config "examples/c04-nvo-maas/config.ini" # doctest: +SKIP
            Still no message found for overwrite_dir
            geometry:   0%           0/2 [00:00<?, ?it/s]
            simulation data:   0%           0/2 [00:00<?, ?it/s]
            simulation data:  50%#####      1/2 [00:01<00:01,  1.92s/it]
            simulation data: 100%########## 2/2 [00:03<00:00,  1.92s/it]
            saving figure C:\...\figures\C1_profile0_velocity_angle.png
            saving figure C:\...\figures\C1_profile0_transverse_flow.png
            saving figure C:\...\figures\profile0_location.png
            ...
            ```

        - Test with missing config file returns error code:
            ```bash
            >>> dfasttf --config nonexistent.ini  # doctest: +SKIP
            Error: Configuration file not found: C:\...\nonexistent.ini
            ```

        - Check version information:
            ```bash
            >>> dfasttf --version  # doctest: +SKIP
            dfasttf 0.1.0

            ```
        - Use custom ships and rivers configuration files:
            ```bash
            >>> dfasttf --config config.ini --ships custom_ships.ini --rivers custom_rivers.ini  # doctest: +SKIP

            Still no message found for overwrite_dir
            geometry:   0%           0/2 [00:00<?, ?it/s]
            simulation data:   0%           0/2 [00:00<?, ?it/s]
            ...
            ```

    See Also:
        parse_arguments: Parses command line arguments.
        validate_args: Validates the parsed arguments.
        run: Executes the actual transverse flow analysis.
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