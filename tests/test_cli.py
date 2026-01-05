"""
Run the dfastrbk module to produce output files for the 'NVO Maas' example.

Usage:
    1) Activate the py_3_10-dfastmi venv.
    2) Run this script.
"""
import sys
import subprocess
from pathlib import Path
from dfasttf import __path__


def test_cli():
    config = "examples/c04-nvo-maas/config.ini"
    ship_dimensions  = Path(__path__[0]) / "ship_dimensions.ini"
    
    cmd = [
        sys.executable,
        "-m",
        "dfasttf",
        "--config", str(config),
        "--ships", str(ship_dimensions),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    assert result.returncode == 0, f"Command failed: {result.stderr}"
