# solver

Main solver module with high-level API.

The solver subpackage provides multiple approaches for recovering stress fields from polarimetric images. This module exports the main functions from all solver submodules for convenient access.

## Overview

Three complementary stress inversion methods are available:

1. **Stokes-based solver** - Uses normalized Stokes components for pixel-wise inversion (recommended for most cases)
2. **Intensity-based solver** - Works directly with raw polarization intensities
3. **Equilibrium solver** - Global inversion enforcing mechanical equilibrium via Airy stress function

::: photoelastimetry.solver
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
