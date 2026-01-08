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

## Input File Formats

### River Chainage File (.xyc)

The river chainage file defines the river axis in a curvilinear coordinate system.

**Format**: Plain text file with three columns (x, y, chainage)

**Example** (Meuse_rivkm_20m.xyc):
```
# River: Meuse
# Spacing: 20m
# Format: X Y Chainage
186250.0 352750.0 0.0
186270.0 352755.0 0.02
186290.0 352760.0 0.04
186310.0 352765.0 0.06
...
```

**Specifications**:
- Whitespace-separated values (spaces or tabs)
- X, Y: Coordinates in meters (RD/local coordinate system)
- Chainage: Distance along river in kilometers
- Lines starting with `#` are treated as comments
- Typically spaced every 20-100m along the river

### Normal Profile File (.geojson)

The normal profile file defines cross-sections where transverse flow analysis is performed.

**Format**: GeoJSON FeatureCollection with LineString features

**Example** (normprofiel.geojson):
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "LineString",
        "coordinates": [
          [186500.0, 353000.0],
          [186500.0, 352500.0]
        ]
      },
      "properties": {
        "name": "profile0",
        "chainage": 12.5,
        "description": "Cross-section at km 12.5"
      }
    },
    {
      "type": "Feature",
      "geometry": {
        "type": "LineString",
        "coordinates": [
          [187000.0, 353200.0],
          [187000.0, 352700.0]
        ]
      },
      "properties": {
        "name": "profile1",
        "chainage": 15.8,
        "description": "Cross-section at km 15.8"
      }
    }
  ]
}
```

**Specifications**:
- GeoJSON format with LineString geometries
- Each LineString represents one cross-section
- Properties:
  - `name`: Unique identifier for the profile (used in output filenames)
  - `chainage`: River kilometer location (optional, for reference)
  - `description`: Human-readable description (optional)
- Coordinates: Same coordinate system as river chainage file
- Profile orientation: Perpendicular to flow direction recommended

### D-Flow FM Output Files (.nc)

The tool accepts D-Flow FM fourier analysis output files in NetCDF format.

**Expected file**: `*_fou.nc` files from D-Flow FM simulations

**Required variables**:
- **ucxa, ucya**: Time-averaged velocity components (m/s)
- **mesh2d_face_x, mesh2d_face_y**: Cell center coordinates
- Mesh topology information (UGRID conventions)

**Notes**:
- Files are typically produced by D-FAST Morphological Impact tool
- Must be on the same computational grid for comparison
- Supports both 2D and 3D simulations (uses depth-averaged data)

### Ship Dimensions File (.ini)

The ship dimensions file defines vessel characteristics for each river branch. This is used for ship-induced wave calculations and bank erosion assessments.

**Format**: INI file with sections for each river branch

**Default file**: `ship_dimensions.ini` (bundled with the package)

**Example**:
```ini
[General]
    Version = 1.0

[Bovenrijn]
    Length = 269.5
    Depth = 4.5

[Waal]
    Length = 269.5
    Depth = 4.5

[Maas]
    Length = 193.0
    Depth = 3.5

[IJssel]
    Length = 110.0
    Depth = 3.5
```

**Parameters**:
- **Length**: Design vessel length in meters
- **Depth**: Design vessel draft in meters

**Usage**: The `Branch` parameter in your configuration file's `[General]` section should match a section name in this file.

### Rivers Configuration File (.ini)

The rivers configuration file defines hydraulic parameters and reach characteristics for Dutch rivers. This includes discharge-depth relationships and river geometry.

**Format**: INI file with sections for each river system

**Default file**: `dutch_rivers.ini` (bundled with the package)

**Example**:
```ini
[General]
    Version    = 3.0
    UCrit      = 0.3
    CelerForm  = 1

[Bovenrijn & Waal]
    QLocation  = Lobith
    HydroQ     = 1300 2000 3000 4000 6000 8000
    AutoTime   = True
    QStagnant  = 800
    QFit       = 800  1280
    
    Reach1     = Bovenrijn                    km  859-867
    NWidth1    = 340
    PropQ1     = 1300 2000 3000 4000 6000 8000
    PropC1     = 0.42 0.98 1.86 2.63 4.79 8.76
    
    Reach2     = Boven-Waal                   km  868-886
    NWidth2    = 260
    PropQ2     = 1300 2000 3000 4000 6000 8000
    PropC2     = 0.63 0.97 1.51 2.06 3.16 5.35

[Maas]
    QLocation  = Borgharen
    HydroQ     = 500 1000 1500 2000 2500 3000
    AutoTime   = True
    QStagnant  = 300
    QFit       = 300 800
    
    Reach1     = Upper Maas                   km  0-50
    NWidth1    = 150
    PropQ1     = 500 1000 1500 2000 2500 3000
    PropC1     = 0.3 0.8 1.2 1.8 2.5 3.5
```

**Parameters**:
- **QLocation**: Reference location for discharge measurements
- **HydroQ**: Discharge levels (m³/s) for which relationships are defined
- **AutoTime**: Automatic time step calculation (`True`/`False`)
- **QStagnant**: Discharge below which flow is considered stagnant (m³/s)
- **QFit**: Discharge range for fitting relationships (m³/s)
- **Reach{n}**: Description of river reach with kilometer range
- **NWidth{n}**: Normal width of the river reach (m)
- **PropQ{n}**: Discharge values for propeller wash calculations (m³/s)
- **PropC{n}**: Propeller wash coefficients corresponding to discharge values

**Usage**: The `Branch` parameter in your configuration file's `[General]` section should match a main section name in this file.

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

Provide a custom ship dimensions file for a different vessel:

```bash
dfasttf --config config.ini --ships my_custom_ships.ini
```

**Custom ship dimensions file format** (`my_custom_ships.ini`):
```ini
[General]
    Version = 1.0

[MyCustomRiver]
    Length = 200.0
    Depth = 4.0
```

Make sure the river name in the `[General]` section of your config matches a section in the ships file.

### Example 3: Using Custom River Configuration

Provide a custom river configuration file with different hydraulic parameters:

```bash
dfasttf --config config.ini --rivers my_rivers.ini
```

**Custom rivers file format** (`my_rivers.ini`):
```ini
[General]
    Version    = 3.0
    UCrit      = 0.3
    CelerForm  = 1

[MyRiver]
    QLocation  = MyLocation
    HydroQ     = 1000 1500 2000 2500
    AutoTime   = True
    QStagnant  = 600
    QFit       = 600 1000
    
    Reach1     = Upper reach                  km  0-10
    NWidth1    = 300
    PropQ1     = 1000 1500 2000 2500
    PropC1     = 0.5 1.0 1.5 2.0
```

## Output Files

### Excel Files

For each profile and case combination, the tool generates Excel files:

**Filename pattern**: `{case}_{profile}_{variable}.xlsx`

**Example files**:
- `C1_profile0_transverse_flow.xlsx`: Transverse flow intensity for Case 1
- `C1_profile0_velocity_angle.xlsx`: Flow angle relative to cross-section
- `C1_profile0_transverse_velocity.xlsx`: Transverse velocity component

**Excel structure**:
- **Column A**: Distance along profile (m)
- **Column B**: Variable value at each point
- Header row with column names and units

### Figure Files

PNG images showing visualizations of the analysis results:

**Filename pattern**: `{case}_{profile}_{variable}.png`

**Figure types**:

1. **Location map** (`{profile}_location.png`)
   - Shows profile location relative to river axis
   - Includes computational mesh visualization

2. **Transverse flow** (`{case}_{profile}_transverse_flow.png`)
   - Plot of transverse flow intensity along cross-section
   - Reference vs. with-measures comparison (if applicable)

3. **Velocity angle** (`{case}_{profile}_velocity_angle.png`)
   - Flow angle deviation from perpendicular
   - Shows flow alignment with cross-section

**Figure specifications**:
- Format: PNG
- Resolution: 300 DPI (print quality)
- Size: 10" × 6" default
- Style: Professional with grid, labels, and legend

### Log File

**Filename**: `dfasttf.log`

**Content**:
- Execution timestamp
- Configuration parameters
- Processing progress
- Warnings and errors
- Performance metrics

**Example log excerpt**:
```
2026-01-08 10:30:15 - INFO - Starting D-FAST Transverse Flow v0.1.0
2026-01-08 10:30:15 - INFO - Reading configuration from: config.ini
2026-01-08 10:30:15 - INFO - Mode: nvo
2026-01-08 10:30:16 - INFO - Loading river chainage: Meuse_rivkm_20m.xyc
2026-01-08 10:30:16 - INFO - Loading normal profiles: normprofiel.geojson
2026-01-08 10:30:17 - INFO - Processing profile: profile0
2026-01-08 10:30:18 - INFO - Processing profile: profile1
2026-01-08 10:30:20 - INFO - Analysis complete
```


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