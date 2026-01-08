# D-FAST Transverse Flow

[![License: LGPL v2.1](https://img.shields.io/badge/License-LGPL_v2.1-blue.svg)](https://www.gnu.org/licenses/lgpl-2.1)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

A tool to perform bank erosion analysis based on D-Flow FM simulation results. D-FAST Transverse Flow analyzes transverse flow patterns and velocity angles at cross-sections to assess potential bank erosion risks.

## Features

- **Transverse Flow Analysis**: Calculate and visualize transverse flow components across river cross-sections
- **Velocity Angle Analysis**: Analyze flow angles relative to channel orientation
- **Multi-Discharge Analysis**: Compare conditions at multiple discharge levels
- **Before/After Comparison**: Evaluate impacts of river interventions (dredging, groins, etc.)
- **Bank Erosion Assessment**: Integrate with ship-induced wave calculations for comprehensive bank erosion risk analysis
- **Automated Reporting**: Generate Excel spreadsheets and publication-quality figures

## Installation

### Prerequisites

- Python 3.11
- [Poetry](https://python-poetry.org/) (recommended) for dependency management

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/Deltares/D-FAST_transverse_flow.git
cd D-FAST_transverse_flow

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

## Quick Start

Run the analysis with the provided example:

```bash
dfasttf --config examples/c04-nvo-maas/config.ini
```

This will:
- Process D-Flow FM simulation results for the Maas river
- Analyze transverse flow at two discharge levels (1300 m³/s and 2100 m³/s)
- Generate Excel files with cross-section data in `examples/c04-nvo-maas/output/`
- Create visualization figures in `examples/c04-nvo-maas/figures/`

## Usage

### Command-Line Interface

```bash
dfasttf --config <path/to/config.ini> [--ships <ships.ini>] [--rivers <rivers.ini>]
```

**Arguments:**
- `--config`: Path to analysis configuration file (required)
- `--ships`: Path to ship dimensions file (optional, uses bundled default)
- `--rivers`: Path to rivers configuration file (optional, uses bundled default)
- `--version`: Show version information
- `--help`: Display help message

### Configuration File

Create a configuration file (`config.ini`) with analysis parameters:

```ini
[General]
  Version          = 3.0
  CaseDescription  = River Analysis
  Branch           = Maas
  Reach            = Upper Reach
  OutputDir        = output
  Plotting         = True
  SavePlots        = True
  FigureDir        = figures
  RiverKM          = river_chainage.xyc
  ProfileLines     = cross_sections.geojson

[BoundingBox]
  xmin = 169000
  xmax = 171000
  ymin = 425000
  ymax = 427000

[C1]
  Discharge        = 1300.0
  Reference        = ref/Q1300.nc
  WithIntervention = with_measure/Q1300.nc

[C2]
  Discharge        = 2100.0
  Reference        = ref/Q2100.nc
  WithIntervention = with_measure/Q2100.nc
```

## Input Files

### Required Files

1. **Configuration File** (`config.ini`): Analysis parameters and file paths
2. **River Chainage File** (`.xyc`): River axis coordinates and chainage
3. **Cross-Section Profile File** (`.geojson`): Profile line definitions
4. **D-Flow FM Results** (`.nc`): NetCDF files with velocity data from simulations

### Optional Files

- **Ship Dimensions File** (`.ini`): Vessel specifications for bank erosion calculations
- **Rivers Configuration File** (`.ini`): Hydraulic parameters for Dutch rivers
- **Bed Change File** (`.nc`): Morphological changes from D-FAST MI analysis

## Output Files

### Excel Files

Cross-section data for each case and profile:
- `{case}_{profile}_transverse_flow.xlsx`: Transverse flow magnitude
- `{case}_{profile}_velocity_angle.xlsx`: Flow angle relative to cross-section
- `{case}_{profile}_transverse_velocity.xlsx`: Transverse velocity component

### Figures

Publication-quality PNG images:
- `{profile}_location.png`: Profile location map with mesh
- `{case}_{profile}_transverse_flow.png`: Transverse flow along profile
- `{case}_{profile}_velocity_angle.png`: Velocity angle distribution

## Documentation

Comprehensive documentation is available:

- **[CLI Guide](docs/mkdocs/user_docs/cli.md)**: Detailed command-line usage and examples
- **[Poetry Guide](docs/mkdocs/guides/poetry.md)**: Development setup with Poetry
- **[Change Log](docs/mkdocs/change-log.md)**: Version history and updates

## Development

### Setting Up Development Environment

```bash
# Install with development dependencies
poetry install --with dev,docs

# Install pre-commit hooks
poetry run pre-commit install

# Run tests
poetry run pytest

# Run tests excluding end-to-end tests
poetry run pytest -m "not e2e"
```

### Running Tests

```bash
# Run all tests with coverage
poetry run pytest

# Run specific test markers
poetry run pytest -m unit          # Unit tests only
poetry run pytest -m integration   # Integration tests
poetry run pytest -m e2e           # End-to-end tests

# Generate coverage report
poetry run pytest --cov --cov-report=html
```

### Code Quality

The project uses:
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **pytest**: Testing framework
- **pre-commit**: Git hooks for code quality

## Example Workflow

1. **Prepare Input Files**
   - D-Flow FM simulation results (`.nc` files)
   - River chainage file defining the river axis
   - GeoJSON file with cross-section profiles

2. **Create Configuration**
   - Define analysis parameters in `config.ini`
   - Specify discharge levels and file paths
   - Set output directories

3. **Run Analysis**
   ```bash
   dfasttf --config config.ini
   ```

4. **Review Results**
   - Excel files with tabulated data
   - Figures showing transverse flow patterns
   - Log file with processing details

## Related Tools

D-FAST Transverse Flow is part of the D-FAST toolkit:

- **[D-FAST Morphological Impact](https://github.com/Deltares/D-FAST_Morphological_Impact)**: Bed level change analysis
- **[D-FAST Bank Erosion](https://github.com/Deltares/D-FAST_Bank_Erosion)**: Comprehensive bank erosion assessment
- **[D-Flow FM](https://www.deltares.nl/en/software/delft3d-flexible-mesh-suite/)**: Hydrodynamic simulation software

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes using [Conventional Commits](https://www.conventionalcommits.org/) format
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Support

For questions, issues, or feature requests:

- **GitHub Issues**: [https://github.com/Deltares/D-FAST_transverse_flow/issues](https://github.com/Deltares/D-FAST_transverse_flow/issues)
- **Email**: delft3d.support@deltares.nl

## License

This project is licensed under the GNU Lesser General Public License v2.1 - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

Developed by [Deltares](https://www.deltares.nl/) for river management and analysis applications.

---

**Note**: This tool requires D-Flow FM simulation results as input. Ensure you have completed hydrodynamic simulations before running the transverse flow analysis.
