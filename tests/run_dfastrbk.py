
#!/usr/bin/env python3
"""
Run the dfastrbk module to produce output files for the 'NVO Maas' example.

Usage:
    1) Activate the py_3_10-dfastmi venv.
    2) Run this script.
"""

import subprocess
import sys
from pathlib import Path

def main():
    module_root = Path("src").resolve()

    config = module_root / "dfasttf" / "examples" / "c04 - NVO Maas" / "config.ini"
    ship_dimensions  = module_root / "dfasttf" / "ship_dimensions.ini"
    
    cmd = [
        sys.executable,
        "-m", "dfasttf",
        "--config", str(config),
        "--ships", str(ship_dimensions),
    ]
    
    subprocess.run(cmd, cwd=module_root, check=True)

if __name__ == "__main__":
    main()
