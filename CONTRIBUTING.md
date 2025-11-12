# Contributing to Photoelastimetry

Thank you for your interest in contributing to photoelastimetry! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/photoelastimetry.git
cd photoelastimetry
```

2. Create a virtual environment and install development dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

## Testing

### Running Tests

We use `pytest` for testing. Tests are located in the `tests/` directory and organized by module:

- `tests/test_stokes_solver_pytest.py` - Tests for Stokes-based stress recovery
- `tests/test_intensity_solver.py` - Tests for intensity-based stress recovery
- `tests/test_equilibrium_solver.py` - Tests for equilibrium-constrained recovery
- `tests/test_disk.py` - Tests for disk simulations
- `tests/test_image_io.py` - Tests for image processing and I/O

Run all tests:
```bash
pytest
```

Run tests with verbose output:
```bash
pytest -v
```

Run a specific test file:
```bash
pytest tests/test_stokes_solver_pytest.py
```

Run a specific test function:
```bash
pytest tests/test_stokes_solver_pytest.py::TestStokesComponents::test_compute_stokes_components
```

### Code Coverage

We aim for >70% test coverage. Check coverage with:

```bash
pytest --cov=photoelastimetry --cov-report=html
```

View the detailed coverage report:
```bash
open htmlcov/index.html  # On macOS
xdg-open htmlcov/index.html  # On Linux
start htmlcov/index.html  # On Windows
```

### Writing Tests

When adding new features, please include tests:

1. **Unit tests**: Test individual functions in isolation
2. **Integration tests**: Test how components work together
3. **Edge cases**: Test boundary conditions and error handling

Example test structure:
```python
import pytest
import numpy as np
from photoelastimetry.solver.stokes_solver import function_to_test

@pytest.fixture
def sample_data():
    """Fixture providing test data."""
    return {
        'param1': value1,
        'param2': value2,
    }

class TestFeature:
    """Test class for a specific feature."""
    
    def test_basic_functionality(self, sample_data):
        """Test basic use case."""
        result = function_to_test(**sample_data)
        assert result is not None
        assert np.isfinite(result)
    
    def test_edge_case(self):
        """Test edge case or error handling."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
```

## Code Style

We follow PEP 8 style guidelines with some modifications:

- Line length: 110 characters (configured in `pyproject.toml`)
- Use `black` for automatic formatting
- Use `flake8` for linting

Format your code:
```bash
black photoelastimetry tests
```

Check code style:
```bash
flake8 photoelastimetry --max-line-length=110
```

## Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes and add tests:**
   - Write clear, documented code
   - Add tests for new functionality
   - Update documentation if needed

3. **Ensure all tests pass:**
   ```bash
   pytest
   black photoelastimetry tests
   flake8 photoelastimetry
   ```

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

5. **Push to your fork and create a pull request:**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **In your PR description, include:**
   - What changes you made and why
   - Any relevant issue numbers (#123)
   - Screenshots if applicable
   - Confirmation that tests pass

## Continuous Integration

Our GitHub Actions workflow automatically:
- Runs tests on Python 3.9, 3.10, 3.11, and 3.12
- Checks code formatting with `black`
- Runs linting with `flake8`
- Uploads coverage reports to Codecov

Your PR must pass all CI checks before it can be merged.

## Documentation

When adding new features:

1. **Add docstrings** following NumPy style:
   ```python
   def my_function(param1, param2):
       """
       Brief description of the function.
       
       Parameters
       ----------
       param1 : type
           Description of param1.
       param2 : type
           Description of param2.
           
       Returns
       -------
       return_type
           Description of return value.
           
       Examples
       --------
       >>> result = my_function(1, 2)
       >>> print(result)
       3
       """
       return param1 + param2
   ```

2. **Update docs/** if adding major features
3. **Add examples** to `docs/examples.md` if relevant

## Questions?

If you have questions or need help:
- Open an issue on GitHub
- Email the maintainers (see README.md)

Thank you for contributing!