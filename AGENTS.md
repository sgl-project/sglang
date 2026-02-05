# SGLang Agent Guidelines

This document provides essential guidelines for agentic coding in the SGLang repository.

## Quick Commands

### Python Development
```bash
# Install development dependencies
cd python && pip install -e ".[dev,test]"

# Run linting and formatting
ruff check python/ --select F401,F821
black python/
isort python/

# Run tests (single test)
python -m pytest test/srt/test_specific_file.py::TestClass::test_method -v
python -m unittest test.srt.test_specific_file.TestClass.test_method -v

# Run full test suite
python -m pytest test/srt/
python -m unittest discover test/srt/
```

### Rust Development
```bash
# Build model gateway
cd sgl-model-gateway
cargo build --release
cargo test

# Format and lint
cargo fmt
cargo clippy -- -D warnings
```

### Build Commands
```bash
# Full project build
cd python && python -m pip install -e ".[dev]"
cd sgl-model-gateway && cargo build --release

# Pre-commit hooks (recommended)
pre-commit run --all-files
```

## Code Style Guidelines

### General Formatting
- **Indentation**: 4 spaces (configured in .editorconfig)
- **Line endings**: LF
- **Encoding**: UTF-8
- **Max line length**: 88 characters (Black default)

### Python Specific

#### Import Organization
```python
# 1. Standard library
import asyncio
import logging
import os
from typing import Dict, List, Optional

# 2. Third-party libraries
import torch
import transformers
from fastapi import FastAPI

# 3. Local imports (sglang package)
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.utils import configure_logger
```

#### Logging Pattern
```python
import logging

logger = logging.getLogger(__name__)

def my_function():
    logger.info("Starting operation")
    logger.debug("Detailed debug info: %s", data)
    logger.error("Error occurred: %s", str(e))
```

#### Error Handling
```python
# Validation errors
raise ValueError(f"Invalid parameter '{param_name}': {value}")
raise TypeError(f"Expected {expected_type}, got {type(value).__name__}")

# Runtime errors
raise RuntimeError(f"NCCL operation failed: {error_str}")
raise RuntimeError("This should not happen")  # Defensive programming

# Feature support
raise NotImplementedError(f"Unsupported backend: {backend_name}")
```

#### Docstring Format
```python
"""Brief description of the function/class/module.

Detailed explanation spanning multiple lines if needed. Include usage
examples and important behavioral notes.

Args:
    param1: Description of first parameter
    param2: Description of second parameter

Returns:
    Description of return value

Raises:
    ValueError: When parameter validation fails
    RuntimeError: When operation cannot be completed
"""
```

#### Type Hints
```python
from typing import Dict, List, Optional, Union, AsyncIterator

def process_data(
    input_data: Dict[str, Any],
    config: Optional[Dict[str, str]] = None
) -> List[str]:
    """Process input data and return results."""
    pass
```

### Rust Specific

#### Error Handling
```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

fn load_config() -> Result<Config, ConfigError> {
    // Implementation
}
```

#### CLI Pattern
```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "sgl-model-gateway")]
#[command(about = "SGLang Model Gateway")]
struct Cli {
    #[arg(short, long, default_value = "8080")]
    port: u16,
    
    #[command(subcommand)]
    command: Option<Commands>,
}
```

## Testing Guidelines

### Python Test Structure
```python
import unittest
from sglang.srt.engine import Engine

class TestMyFeature(unittest.TestCase):
    """Test suite for my feature.
    
    This test suite validates the core functionality of the feature
    including edge cases and error conditions.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.engine = Engine(model_path="test_model")
    
    def test_basic_functionality(self):
        """Test basic functionality with valid inputs."""
        result = self.engine.process("test input")
        self.assertIsNotNone(result)
        self.assertEqual(result.status, "success")
    
    def test_error_handling(self):
        """Test proper error handling for invalid inputs."""
        with self.assertRaises(ValueError):
            self.engine.process("")
```

### Running Tests
```bash
# Single test file
python -m pytest test/srt/test_my_feature.py -v

# Specific test method
python -m pytest test/srt/test_my_feature.py::TestMyFeature::test_basic_functionality -v

# Run with coverage
python -m pytest test/srt/ --cov=sglang --cov-report=html

# Rust tests
cargo test --package sgl-model-gateway
```

## File Organization

### Python Package Structure
```
python/sglang/
├── __init__.py              # Public API exports
├── lang/                    # Frontend language APIs
│   ├── __init__.py
│   ├── chat_template.py
│   └── compilation.py
├── srt/                     # Backend runtime
│   ├── __init__.py
│   ├── managers/            # Engine managers
│   ├── layers/              # Model layers
│   └── model_loader/        # Model loading
├── utils/                   # Utilities
│   ├── __init__.py
│   └── logger.py
└── jit_kernel/              # JIT compilation
    └── __init__.py
```

## Development Workflow

1. **Set up environment**: Install pre-commit hooks
2. **Make changes**: Follow style guidelines
3. **Run linting**: `pre-commit run --all-files`
4. **Add tests**: Include comprehensive test coverage
5. **Run tests**: Verify all tests pass
6. **Build**: Ensure both Python and Rust components build successfully

## Common Patterns

### Lazy Imports for Heavy Dependencies
```python
from sglang.utils import LazyImport

# For optional or heavy dependencies
Anthropic = LazyImport("sglang.lang.backend.anthropic", "Anthropic")
```

### Configuration Management
```python
@dataclass
class Config:
    model_path: str
    tp_size: int = 1
    mem_fraction: float = 0.9
    
    def __post_init__(self):
        if self.tp_size < 1:
            raise ValueError("tp_size must be >= 1")
```

### Async Patterns
```python
async def process_stream(
    self, inputs: List[str]
) -> AsyncIterator[Dict[str, Any]]:
    """Process inputs and yield results asynchronously."""
    for input_text in inputs:
        result = await self._process_single(input_text)
        yield {"text": result, "finished": False}
    yield {"finished": True}
```

## Tool Configuration

The project uses these tools with specific configurations:
- **Black**: Code formatting (88 char line length)
- **Ruff**: Linting (F401, F821 rules focus)
- **isort**: Import sorting (black profile)
- **rustfmt**: Rust formatting
- **clippy**: Rust linting (warnings as errors)

Always run `pre-commit run --all-files` before submitting changes to ensure consistency with the existing codebase.