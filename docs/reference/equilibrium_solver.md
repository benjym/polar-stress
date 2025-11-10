# solver.equilibrium_solver

Global equilibrium-based stress inversion.

This module implements a global stress field recovery method that enforces mechanical equilibrium constraints using an Airy stress function representation. This approach ensures the recovered stress field satisfies equilibrium equations.

## Key Functions

- `build_finite_difference_operators()` - Construct finite difference matrices
- `airy_to_stress()` - Convert Airy function to stress components
- `recover_stress_field_global()` - Main function for equilibrium-based recovery
- `compare_local_vs_global()` - Compare pixel-wise vs equilibrium methods

::: photoelastimetry.solver.equilibrium_solver
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
