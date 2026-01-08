# Command Line Interface (CLI)

## Overview

D-FAST Transverse Flow provides a command-line interface for analyzing transverse flow patterns and bank erosion based on D-Flow FM simulation results. The tool processes velocity data from multiple simulations and generates visualizations and Excel reports.

## Installation

After installing the package using Poetry, the CLI tool becomes available as the `dfasttf` command:

```bash
poetry install
poetry run dfasttf
```

## Basic Usage

The tool operates using a configuration file that specifies all analysis parameters:

```bash
dfasttf --config path/to/config.ini
```

Optionally, you can specify custom ship dimensions and river configuration files:

```bash
dfasttf --config path/to/config.ini --ships custom_ships.ini --rivers custom_rivers.ini
```

## Command-Line Arguments

The CLI accepts the following arguments:

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `--config` | Path to analysis configuration file (INI format) | Yes | - |
| `--ships` | Path to ship dimensions file (INI format) | No | `ship_dimensions.ini` (bundled) |
| `--rivers` | Path to rivers configuration file (INI format) | No | `dutch_rivers.ini` (bundled) |
| `--version` | Show program version and exit | No | - |
| `-h`, `--help` | Show help message and exit | No | - |

### Getting Help

Display the help message with all available options:

```bash
dfasttf --help
```

Expected output:
```
usage: dfasttf [-h] [--version] --config CONFIG [--rivers RIVERS] [--ships SHIPS]

D-FAST Transverse Flow - Analyze transverse flow patterns in rivers

options:
  -h, --help       show this help message and exit
  --version        show program's version number and exit
  --config CONFIG  path to analysis configuration file (required)
  --rivers RIVERS  path to rivers configuration file (default: dutch_rivers.ini)
  --ships SHIPS    path to ship dimensions file (default: ship_dimensions.ini)
```

### Check Version

Display the installed version:

```bash
dfasttf --version
```

## Configuration File Format

The configuration file uses INI format with sections for different aspects of the analysis. All paths in the configuration file are relative to the location of the configuration file itself.

### Complete Configuration Example

Here's a complete example from the NVO Maas test case:

```ini
[General]
  Version          = 3.0
  CaseDescription  = Maas - analysis
  Branch           = Maas
  Reach            = Maas
  OutputDir        = output
  Plotting         = True
  PlotType         = 1D
  SavePlots        = True
  InvertXAxis      = True
  FigureDir        = figures
  RiverKM          = Meuse_rivkm_20m.xyc
  ProfileLines     = normprofiel.geojson
  WaterUpliftCorrection = True
  BedChangeCorrection = True
  BedChangeFile    = nvo\dfastmi_results.nc

[BoundingBox]
  xmin = 169094
  xmax = 170532
  ymin = 425838
  ymax = 426960

[C1]
  Discharge        = 1300.0
  Reference        = ref/S_1300/Maas_0000_fou.nc
  WithIntervention = nvo/S_1300/Maas_0000_fou.nc

[C2]
  Discharge        = 2100.0
  Reference        = ref/S_2100/Maas_0000_fou.nc
  WithIntervention = nvo/S_2100/Maas_0000_fou.nc
```

### Section Descriptions

#### [General]

Core settings for the analysis:

- **Version**: Configuration file format version (use `3.0`)
- **CaseDescription**: Descriptive name for the analysis case
- **Branch**: River branch name (used for looking up ship dimensions and river properties)
- **Reach**: Specific reach within the branch
- **OutputDir**: Directory where Excel output files will be saved (relative to config file)
- **Plotting**: Enable/disable figure generation (`True`/`False`)
- **PlotType**: Type of plots to generate (`1D` for cross-section plots)
- **SavePlots**: Save plots to files instead of displaying (`True`/`False`)
- **InvertXAxis**: Invert the x-axis in plots (`True`/`False`)
- **FigureDir**: Directory where PNG figures will be saved (relative to config file)
- **RiverKM**: Path to river chainage file (.xyc format) defining river axis
- **ProfileLines**: Path to profile lines file (.geojson format) defining cross-sections
- **WaterUpliftCorrection**: Apply water uplift correction (`True`/`False`)
- **BedChangeCorrection**: Apply bed change correction (`True`/`False`)
- **BedChangeFile**: Path to NetCDF file with bed change data from D-FAST MI

#### [BoundingBox]

Spatial extent for analysis and visualization:

- **xmin, xmax**: X-coordinate bounds in meters (local coordinate system)
- **ymin, ymax**: Y-coordinate bounds in meters (local coordinate system)

This bounding box is used to:
- Limit the computational domain for efficiency
- Define the extent of location maps
- Filter relevant mesh cells

#### [C1], [C2], [C3], ...

Case definitions for different discharge levels. Each case compares reference conditions with intervention scenarios:

- **Discharge**: Discharge level in m³/s
- **Reference**: Path to D-Flow FM output file for reference condition (without intervention)
- **WithIntervention**: Path to D-Flow FM output file with intervention/measures applied

You can define as many cases as needed (C1, C2, C3, ...). Each case will generate separate output files and figures.


## Usage Examples

### Example 1: Basic Analysis with Configuration File

Analyze transverse flow at cross-sections using the provided example configuration:

```bash
dfasttf --config examples/c04-nvo-maas/config.ini
```

**Expected output**:
- Excel files in `examples/c04-nvo-maas/output/`
- PNG figures in `examples/c04-nvo-maas/figures/`
- Log messages showing progress

**Console output** (excerpt):
```
geometry:   0%           0/2 [00:00<?, ?it/s]
simulation data:   0%           0/2 [00:00<?, ?it/s]
simulation data:  50%#####      1/2 [00:01<00:01,  1.92s/it]
simulation data: 100%########## 2/2 [00:03<00:00,  1.92s/it]
saving figure C:\...\figures\C1_profile0_velocity_angle.png
saving figure C:\...\figures\C1_profile0_transverse_flow.png
saving figure C:\...\figures\profile0_location.png
...
```

### Example 2: Using Custom Ship Dimensions

The tool includes a default `ship_dimensions.ini` file with predefined vessel dimensions for Dutch rivers. You can provide a custom ship dimensions file if needed:

```bash
dfasttf --config examples/c04-nvo-maas/config.ini --ships path/to/custom_ships.ini
```

**Note**: The example uses the `Branch = Maas` setting in the config file, which references the "Maas" section in the bundled `ship_dimensions.ini`:

```ini
[Maas]
    Length = 193.0
    Depth = 3.5
```

The `Branch` parameter in your configuration file must match a section name in the ships file.

### Example 3: Multiple Discharge Levels

The provided example demonstrates analysis at two discharge levels (1300 m³/s and 2100 m³/s). You can add more discharge cases by extending the configuration:

```bash
dfasttf --config examples/c04-nvo-maas/config.ini
```

The configuration file defines multiple cases:

```ini
[C1]
  Discharge        = 1300.0
  Reference        = ref/S_1300/Maas_0000_fou.nc
  WithIntervention = nvo/S_1300/Maas_0000_fou.nc

[C2]
  Discharge        = 2100.0
  Reference        = ref/S_2100/Maas_0000_fou.nc
  WithIntervention = nvo/S_2100/Maas_0000_fou.nc
```

Each case generates separate output files and figures for both discharge conditions.

**Note**: The tool uses the bundled `dutch_rivers.ini` file which contains hydraulic parameters for Dutch river systems including the Maas. The `--rivers` argument is currently accepted but not used in the analysis.


## Error Messages and Troubleshooting

### Common Errors

**"Configuration file not found"**
```
Error: Configuration file not found: C:\path\to\config.ini
```
**Solution**: Verify the file path is correct and the file exists.

---

**"Ship dimensions file not found"**
```
Error: Ship dimensions file not found: C:\path\to\ships.ini
```
**Solution**: Check the path to your custom ships file, or omit `--ships` to use the default bundled file.

---

**"Rivers configuration file not found"**
```
Error: Rivers configuration file not found: C:\path\to\rivers.ini
```
**Solution**: Check the path to your custom rivers file, or omit `--rivers` to use the default bundled file.

---

### Report Issues

For bugs or feature requests, visit:
- **GitHub**: https://github.com/Deltares/D-FAST_transverse_flow/issues
- **Email**: delft3d.support@deltares.nl