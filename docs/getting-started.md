# Getting Started

## Installation

To install the package, run the following command in the terminal:

```bash
pip install photoelastimetry
```

## Quick Start

After installation, three command line scripts are available for different photoelasticity workflows:

### image-to-stress

Convert photoelastic images to stress maps:

```bash
image-to-stress params.json5 --output stress_map.png
```

### stress-to-image

Generate photoelastic fringe patterns from stress fields:

```bash
stress-to-image params.json5
```

### demosaic-raw

Process raw polarimetric images from specialized cameras:

```bash
demosaic-raw image.raw --width 2448 --height 2048 --format tiff
```

## Configuration

All tools use JSON5 parameter files for configuration. See the [User Guide](user-guide.md) for detailed parameter descriptions.

## Next Steps

- Read the [User Guide](user-guide.md) for detailed usage instructions
- Check the [Examples](examples.md) for practical demonstrations
- Browse the [API Reference](reference/index.md) for module documentation
