"""
Photoelastimetry stress inversion solvers.

This package provides multiple approaches for recovering stress fields from
polarimetric images:

- stokes_solver: Pixelwise inversion using normalized Stokes components
- intensity_solver: Pixelwise inversion using raw polarization intensities
- equilibrium_solver: Global inversion enforcing mechanical equilibrium via Airy function
"""

from photoelastimetry.solver.stokes_solver import (
    compute_stokes_components,
    compute_normalized_stokes,
    predict_stokes,
    compute_residual,
    recover_stress_tensor,
    compute_solid_fraction,
    recover_stress_map as recover_stress_map_stokes,
)

from photoelastimetry.image import (
    compute_retardance,
    compute_principal_angle,
    mueller_matrix,
)

from photoelastimetry.solver.equilibrium_solver import (
    build_finite_difference_operators,
    airy_to_stress,
    compute_global_residual,
    recover_stress_field_global,
    recover_stress_field_global_iterative,
    compare_local_vs_global,
)

from photoelastimetry.solver.intensity_solver import (
    predict_intensity,
    compute_intensity_residual,
    recover_stress_tensor_intensity,
    recover_stress_map as recover_stress_map_intensity,
    compare_stokes_vs_intensity,
)

__all__ = [
    # Stokes-based solver
    "compute_stokes_components",
    "compute_normalized_stokes",
    "compute_retardance",
    "compute_principal_angle",
    "mueller_matrix",
    "predict_stokes",
    "compute_residual",
    "recover_stress_tensor",
    "compute_solid_fraction",
    "recover_stress_map_stokes",
    # Intensity-based solver
    "predict_intensity",
    "compute_intensity_residual",
    "recover_stress_tensor_intensity",
    "recover_stress_map_intensity",
    "compare_stokes_vs_intensity",
    # Equilibrium solver
    "build_finite_difference_operators",
    "airy_to_stress",
    "compute_global_residual",
    "recover_stress_field_global",
    "recover_stress_field_global_iterative",
    "compare_local_vs_global",
]
