# Targeted Python validation

- The system `python3` on macOS may be older than SGLang's supported Python. Use
  an explicit Python 3.12 virtual environment for focused connector tests.
- Editable installs now select Rust extensions by default. For Python-only CPU
  tests, install the build prerequisites (`setuptools`, `setuptools-rust`,
  `setuptools-scm`, `wheel`, and the `torch` version in `python/pyproject.toml`),
  then use `SGLANG_BUILD_RUST_EXTS=none python -m pip install --no-build-isolation
  --no-deps -e ./python`. `--no-deps` does not install runtime/test dependencies;
  provision those separately. Do not treat this as a serving installation.
- On macOS arm64, SGLang installs its own Triton/MPS compatibility stubs. Use
  the existing `maybe_stub_sgl_kernel()` helper for CPU scheduler tests. These
  tests cannot establish CUDA transfer correctness. Verify GPU/container
  availability separately before claiming device validation.
- Importing the scheduler on macOS arm64 also imports the MLX scheduler mixin;
  install `mlx` and `mlx-lm` even for CPU-only scheduler tests on that platform.
- `ServerArgs(...)` and `prepare_server_args(...)` return raw records. Argument
  validation tests must call `resolve_once()` (or publish a runtime context);
  construction alone no longer executes the resolution pipeline.
- If pip-based pre-commit environment creation repeatedly fails downloading
  dependencies while uv works, `uvx --python 3.12 --with pre-commit-uv pre-commit
  run --files ...` installs and runs the same pinned hooks using uv.
