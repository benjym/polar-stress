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

Converts photoelastic images to stress maps using the stress-optic law and polarisation analysis.

```bash
image-to-stress <json_filename> [--output OUTPUT]
```

**Arguments:**

- `json_filename`: Path to the JSON5 parameter file containing configuration (required)
- `--output`: Path to save the output stress map image (optional)

**Example:**

```bash
image-to-stress params.json5 --output stress_map.png
```

The JSON5 parameter file should contain:

- `folderName`: Path to folder containing raw photoelastic images
- `C`: Stress-optic coefficient in 1/Pa
- `thickness`: Sample thickness in meters
- `wavelengths`: List of wavelengths in nanometers
- `polariser_angle` (optional): Polariser angle in degrees relative to the 0 degree camera axis (default: 0.0)
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

### demosaic-raw

De-mosaics a raw polarimetric image from a camera with a 4x4 superpixel pattern into separate colour and polarisation channels.

```bash
demosaic-raw <input_file> [--width WIDTH] [--height HEIGHT] [--dtype DTYPE] [--output-prefix PREFIX] [--format FORMAT] [--all]
```

**Arguments:**

- `input_file`: Path to the raw image file, or directory when using `--all` (required)
- `--width`: Image width in pixels (default: 4096)
- `--height`: Image height in pixels (default: 3000)
- `--dtype`: Data type, either 'uint8' or 'uint16' (auto-detected if not specified)
- `--output-prefix`: Prefix for output files (default: input filename without extension)
- `--format`: Output format, either 'tiff' or 'png' (default: 'tiff')
- `--all`: Recursively process all .raw files in the input directory and subdirectories

**Examples:**

```bash
# Save as a single TIFF stack
demosaic-raw image.raw --width 2448 --height 2048 --dtype uint16 --format tiff

# Save as four separate PNG files (one per polarisation angle)
demosaic-raw image.raw --width 2448 --height 2048 --format png --output-prefix output

# Process all raw files in a directory recursively
demosaic-raw images/ --format png --all
```

**Output formats:**

- `tiff`: Creates a single TIFF file with shape [H/4, W/4, 4, 4] containing all colour channels (R, G1, G2, B) and polarisation angles (0°, 45°, 90°, 135°)
- `png`: Creates 4 PNG files (one per polarisation angle), each containing all colour channels as a composite image

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
