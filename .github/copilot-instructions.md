# Project Overview

This project is a 3D computer vision framework that provides a modular pipeline for processing RGB-D data, performing object detection, pose estimation, and point cloud manipulation. It is designed to be extensible and easy to integrate with various 3D vision tasks using state of art deep learning models.

Refer to the main [README](../README.md) for a comprehensive overview of the project and update it when necessary.

## Project Structure

```
regenbogen/
├── regenbogen/           # Main package
│   ├── core/            # Core pipeline and node infrastructure
│   │   ├── pipeline.py       # Sequential pipeline implementation
│   │   ├── graph_pipeline.py # DAG-based pipeline with branching
│   │   └── node.py           # Base Node class
│   ├── interfaces/      # Data structures for inter-node communication
│   │   └── __init__.py       # Frame, ObjectModel, Pose, etc.
│   ├── nodes/           # Pipeline node implementations
│   │   ├── depth_anything.py
│   │   ├── sam2.py
│   │   ├── bop_dataset.py
│   │   └── ...
│   └── utils/           # Utility functions
│       └── rerun_logger.py
├── tests/               # Test suite
├── examples/            # Usage examples
├── docs/                # Sphinx documentation
└── pyproject.toml       # Project configuration
```

## Libraries and Frameworks

- **uv**: ALWAYS use uv for dependency management and running in virtual environment by doing `uv run python <script>`
- **pytest**: Use for testing by doing `uv run pytest`
- **ruff**: Use for linting and code formatting
- **rerun-sdk**: Use for visualizing intermediate results in the pipeline
- **numpy**: Core numerical operations and arrays
- **open3d**: 3D geometry processing and ICP
- **torch/transformers**: Deep learning models (DepthAnything, SAM2)
- **pyrender**: 3D rendering for template generation
- Use pydantic for data validation and typing when needed
- Use python type hints including annotations from `__future__` for forward compatibility

## Development Workflow

### Setting Up Development Environment

Note: The uv environment is already set up before Copilot session starts. If you need to manually set up:

```bash
# Clone the repository
git clone git@github.com:onosamo/regenbogen.git
cd regenbogen

# Install all dependencies including dev and full groups
uv sync --group dev --group full

# Verify installation
uv run pytest tests/test_basic.py -v
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_basic.py -v

# Run with coverage
uv run pytest --cov=regenbogen tests/

# Run specific test function
uv run pytest tests/test_basic.py::test_frame_interface -v
```

### Linting and Formatting

```bash
# Check code with ruff
uv run ruff check .

# Format code with ruff
uv run ruff format .

# Check specific files
uv run ruff check regenbogen/nodes/new_node.py
```

### Running Examples

```bash
# Run examples using uv
uv run python examples/complete_pipeline_example.py
uv run python examples/pytorch_style_example.py
uv run python examples/bop_dataset_example.py
```

## Coding Standards

### General Principles

- Follow Google style guide for Python code
- Use type hints for all function signatures
- Write docstrings for all public classes and functions
- Use logging for debug/info/warning messages instead of print statements
- Write unit tests for all new functionality
- Always prefer minimal code changes that achieve the goal
- Stick to Python Zen principles, e.g. "Simple is better than complex" and "Readability counts"
- Draw inspiration from established libraries like numpy, pytorch, and scikit-learn for API design and coding style
- Always check implemented core functionality and reuse existing standard library features and interfaces
- Don't add comments that state what the code is doing, only add comments that explain why something is done a certain way if it's not obvious

### Code Style

- Line length: 88 characters (Black default)
- Use descriptive variable names
- Prefer composition over inheritance
- Keep functions small and focused
- Use pydantic models for data structures (preferred over standard dataclasses)

### Type Hints

```python
from __future__ import annotations
import numpy.typing as npt
import numpy as np

def process_frame(
    frame: Frame,
    intrinsics: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
    """Process frame and return results."""
    ...
```

## Architecture Patterns

### Node Implementation

All nodes inherit from the base `Node` class and implement the `process()` method:

```python
from regenbogen import Node
from regenbogen.interfaces import Frame

class MyNode(Node):
    """Brief description of what this node does."""
    
    def __init__(self, param1: str, param2: int = 10):
        super().__init__(name="MyNode")
        self.param1 = param1
        self.param2 = param2
    
    def process(self, frame: Frame) -> Frame:
        """Process the frame and return modified frame."""
        # Implementation
        return frame
```

### Pipeline Usage

Two pipeline patterns are supported:

**Sequential Pipeline:**
```python
from regenbogen import Pipeline
pipeline = Pipeline(name="MyPipeline")
pipeline.add_node(NodeA())
pipeline.add_node(NodeB())
result = pipeline.process(data)
```

**Graph Pipeline (PyTorch-style):**
```python
from regenbogen import execute
node_a = NodeA()
node_b = NodeB()(node_a)  # Connect nodes
node_c = NodeC()(node_a)  # Branch from node_a
for result in execute(node_b, streaming=True):
    process(result)
```

### Data Interfaces

Use standardized interfaces for data exchange (note: interfaces are still evolving):
- `Frame`: RGB-D images with camera intrinsics/extrinsics
- `ObjectModel`: 3D mesh or pointcloud models
- `Pose`: Rotation matrix + translation vector
- `BoundingBoxes`: Object detection results
- `ErrorMetrics`: Evaluation results

### Rerun Visualization

Rerun logging should be enabled by default for development and visualization (not just debugging):
```python
pipeline = Pipeline(name="MyPipeline", enable_rerun_logging=True)
```
This is the standard mode of operation for the framework.

## Testing Guidelines

### Test Structure

- Place tests in `tests/` directory
- Name test files as `test_<module>.py`
- Name test functions as `test_<functionality>`
- Use descriptive test names that explain what is being tested

### Test Patterns

```python
def test_node_basic_functionality():
    """Test basic node operation with valid input."""
    node = MyNode(param1="test")
    frame = Frame(rgb=np.zeros((480, 640, 3), dtype=np.uint8))
    result = node.process(frame)
    assert result.rgb.shape == (480, 640, 3)

def test_node_with_invalid_input():
    """Test node handles invalid input gracefully."""
    node = MyNode(param1="test")
    with pytest.raises(ValueError):
        node.process(None)
```

### Mock External Dependencies

For nodes that download models or require GPU:
- Mock HuggingFace model loading in tests
- Use small test data (e.g., 64x64 images)
- Skip GPU tests with `@pytest.mark.skipif(not torch.cuda.is_available())`

## Documentation

- Main docs are in `docs/` using Sphinx
- Build docs: `cd docs && make html`
- Update docstrings in code - they auto-generate API docs
- Add usage examples to `docs/source/` for major features
- Keep README.md up to date with new capabilities

## CI/CD

The project uses GitHub Actions:
- **Lint**: Runs `ruff check` on all Python files
- **Test**: Runs pytest on Python 3.11
- Tests run on push to `main` and `develop` branches
- Tests also run on all pull requests

## Useful Commands

```bash
# Install dependencies
uv sync --group dev --group full

# Run tests
uv run pytest

# Lint code
uv run ruff check .

# Format code
uv run ruff format .

# Run example
uv run python examples/complete_pipeline_example.py

# Build documentation
cd docs && make html

# Clean generated files
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```