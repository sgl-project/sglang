# SGLang Router Python Bindings

This directory contains the Python bindings for the SGLang Router, built using [maturin](https://github.com/PyO3/maturin) and [PyO3](https://github.com/PyO3/pyo3).

## Directory Structure

```
bindings/python/
├── sglang_router/          # Python source code
│   ├── __init__.py
│   ├── version.py
│   ├── launch_server.py
│   ├── launch_router.py
│   ├── router.py
│   ├── router_args.py
│   └── mini_lb.py
├── pyproject.toml          # Python package configuration
├── setup.py                # Setup configuration
├── MANIFEST.in             # Package manifest
├── .coveragerc             # Test coverage configuration
└── README.md               # This file
```

## Building

### Development Build

```bash
# Install maturin
pip install maturin

# Build and install in development mode
cd sgl-router/bindings/python
maturin develop --manifest-path ../../Cargo.toml --features vendored-openssl
```

### Production Build

```bash
# Build wheel
cd sgl-router/bindings/python
maturin build --manifest-path ../../Cargo.toml --release --out dist --features vendored-openssl

# Install the built wheel
pip install dist/sglang_router-*.whl
```

## Testing

```bash
# Run Python tests
cd sgl-router
pytest py_test/
```

## Configuration

- **pyproject.toml**: Defines package metadata, dependencies, and build configuration
- **python-source**: Set to "." to indicate Python source is in the same directory as pyproject.toml
- **module-name**: `sglang_router.sglang_router_rs` - the Rust extension module name

## Notes

- The Rust source code is located in `../../src/`
- Cargo.toml and build.rs are located in `../../` (sgl-router root)
- The package includes both Python code and Rust extensions built with PyO3
