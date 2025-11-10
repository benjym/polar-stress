# Contributing

Thank you for your interest in contributing to Photoelastimetry! This guide will help you get started.

## Development Setup

1. **Clone the repository:**

```bash
git clone https://github.com/benjym/photoelastimetry.git
cd photoelastimetry
```

2. **Install in editable mode:**

```bash
pip install -e .
```

This installs the package in development mode, allowing you to make changes and test them immediately.

## Code Style

The project uses automated code formatting and linting:

### Black

Code is formatted using [Black](https://github.com/psf/black) with a line length of 110 characters:

```bash
black photoelastimetry/
```

### Flake8

Code is checked using [Flake8](https://flake8.pycqa.org/):

```bash
flake8 photoelastimetry/
```

### Pre-commit Hooks

Pre-commit hooks are set up to automatically format code before commits:

```bash
pre-commit install
```

This will run Black and Flake8 automatically on your changes before each commit.

## Testing

Before submitting changes, ensure your code works correctly:

```bash
# Run any existing tests
python test_local.py

# Test command-line tools
image-to-stress --help
stress-to-image --help
demosaic-raw --help
```

## Documentation

### Building Documentation Locally

The documentation is built using MkDocs:

```bash
mkdocs serve
```

This starts a local server at `http://127.0.0.1:8000` where you can preview your documentation changes.

To build the documentation:

```bash
mkdocs build
```

### Writing Documentation

- **Docstrings**: Use NumPy-style docstrings for all public functions and classes
- **Examples**: Include examples in docstrings where appropriate
- **User Guides**: Update user guides when adding new features

Example docstring format:

```python
def compute_stress(I, C, t, wavelength):
    """
    Compute stress from intensity measurements.
    
    Parameters
    ----------
    I : ndarray
        Intensity measurements with shape (H, W, 4)
    C : float
        Stress-optic coefficient in 1/Pa
    t : float
        Sample thickness in meters
    wavelength : float
        Light wavelength in meters
        
    Returns
    -------
    stress : ndarray
        Stress tensor with shape (H, W, 3) containing
        [sigma_xx, sigma_yy, sigma_xy]
        
    Examples
    --------
    >>> I = np.random.rand(100, 100, 4)
    >>> stress = compute_stress(I, 5e-11, 0.005, 550e-9)
    >>> stress.shape
    (100, 100, 3)
    """
    # Implementation
    pass
```

## Submitting Changes

1. **Create a new branch:**

```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes** and commit them:

```bash
git add .
git commit -m "Add feature: description of your changes"
```

3. **Push to GitHub:**

```bash
git push origin feature/your-feature-name
```

4. **Create a Pull Request** on GitHub with a clear description of your changes.

## Pull Request Guidelines

- Write clear, descriptive commit messages
- Update documentation for new features
- Add examples if appropriate
- Ensure code passes Black and Flake8 checks
- Test your changes thoroughly

## Reporting Issues

When reporting issues, please include:

- Python version
- Operating system
- Steps to reproduce the issue
- Expected vs actual behavior
- Error messages and stack traces

## Questions?

If you have questions about contributing, feel free to:

- Open an issue on GitHub
- Contact the maintainers:
  - [Benjy Marks](mailto:benjy.marks@sydney.edu.au)
  - [Qianyu Fang](mailto:qianyu.fang@sydney.edu.au)

## Code of Conduct

Please be respectful and constructive in all interactions with the community. We aim to maintain a welcoming environment for all contributors.

## License

By contributing to Photoelastimetry, you agree that your contributions will be licensed under the GPL-3.0-or-later license.
