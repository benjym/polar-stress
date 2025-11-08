# photoelastimetry

Package for processing polarised images to measure stress in granular media

## Installation

To install the package, run the following command in the terminal:

```bash
pip install photoelastimetry
```

## Usage

After installation, two command line scripts are available:

### image-to-stress

Converts photoelastic images to stress maps using the stress-optic law and polarization analysis.

```bash
image-to-stress <json_filename> [--output OUTPUT] [--polariser-angle ANGLE]
```

**Arguments:**
- `json_filename`: Path to the JSON5 parameter file containing configuration (required)
- `--output`: Path to save the output stress map image (optional)
- `--polariser-angle`: Polariser angle in degrees relative to the 0 degree camera axis (default: 0.0)

**Example:**
```bash
image-to-stress params.json5 --output stress_map.png --polariser-angle 45.0
```

The JSON5 parameter file should contain:
- `folderName`: Path to folder containing raw photoelastic images
- `C`: Stress-optic coefficient in 1/Pa
- `thickness`: Sample thickness in meters
- `wavelengths`: List of wavelengths in nanometers
- `crop` (optional): Crop region as [y1, y2, x1, x2]
- `debug` (optional): If true, display all channels for debugging

### stress-to-image

Converts stress field data to photoelastic fringe pattern images.

```bash
stress-to-image <json_filename>
```

**Arguments:**
- `json_filename`: Path to the JSON5 parameter file containing configuration (required)

**Example:**
```bash
stress-to-image params.json5
```

The JSON5 parameter file should contain:
- `p_filename`: Path to the photoelastimetry parameter file
- `stress_filename`: Path to the stress field data file
- `t`: Thickness of the photoelastic material
- `lambda_light`: Wavelength of light used in the experiment
- `C`: Stress-optic coefficient of the material
- `scattering` (optional): Gaussian filter sigma for scattering simulation
- `output_filename` (optional): Path for the output image (default: "output.png")

## Development

To set up the development environment, clone the repository and install the package in editable mode:

```bash
git clone https://github.com/benjym/photoelastimetry.git
cd photoelastimetry
pip install -e .
```

## Authors

- [Benjy Marks](benjy.marks@sydney.edu.au)
- [Qianyu Fang](qianyu.fang@sydney.edu.au)
