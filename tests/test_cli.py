"""
Run the dfasttf module to produce output files for the 'NVO Maas' example.

Usage:
    1) Activate the appropriate venv.
    2) Run this script with pytest.
"""
import sys
import subprocess
import pytest


@pytest.mark.e2e
def test_cli():
    """Test the CLI command with the NVO Maas example."""
    config = "examples/c04-nvo-maas/config.ini"

    cmd = [
        sys.executable,
        "-m",
        "dfasttf",
        "--config", str(config),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    assert result.returncode == 0, f"Command failed: {result.stderr}"
