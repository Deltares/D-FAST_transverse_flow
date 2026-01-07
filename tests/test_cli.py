"""
Run the dfasttf module to produce output files for the 'NVO Maas' example.

Usage:
    1) Activate the appropriate venv.
    2) Run this script with pytest.
"""
import sys
import subprocess
from pathlib import Path
from dfasttf import __path__


def test_cli():
    """Test the CLI command with the NVO Maas example."""
    config = "examples/c04-nvo-maas/config.ini"

    ship_dimensions = Path(__path__[0]) / "data" / "ship_dimensions.ini"

    cmd = [
        sys.executable,
        "-m",
        "dfasttf",
        "--config", str(config),
        "--ships", str(ship_dimensions),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    assert result.returncode == 0, f"Command failed: {result.stderr}"
