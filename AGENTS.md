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

# External connector GPU validation

- Pin the serving image by digest and inspect its entrypoint, Python, torch,
  CUDA, and compiler versions before installing either project. Use a separate
  container with explicitly selected idle GPUs and a writable data volume.
- A cached serving image may predate the checkout's dependency pins. Compare
  the installed distributions against `python/pyproject.toml` and install the
  checkout's runtime dependencies; a `--no-deps` editable install alone is not
  sufficient for GPU serving. Keep the FlashInfer/kernel version checks enabled.
  FlashInfer's optional `flashinfer-cubin` and `flashinfer-jit-cache` packages
  must also match `flashinfer-python`; upgrading Python dependencies does not
  upgrade these image-provided artifacts. Use the indexes in `docker/Dockerfile`.
- The Ubuntu SGLang image marks system Python as externally managed. Inside
  the disposable test container, `UV_BREAK_SYSTEM_PACKAGES=1` is needed for
  `uv pip install --system`; this does not apply to host installations.
- Build LMCache for H200 with `TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=1`. Verify
  `cusparse.h` exists before building; CUDA 13 images may put headers under
  `site-packages/nvidia/cu13/include` as well as the toolkit include directory.
- Exclude `rust/target` and Python caches when transferring a checkout between
  macOS and Linux. Run long builds/tests detached in the container and preserve
  logs on the mounted volume so an SSH disconnect cannot lose the evidence.
- When the working directory contains a checkout named `sglang`, Python may
  resolve it as a namespace package before an editable install's finder runs.
  Put the checkout's `python/` directory first on `PYTHONPATH` and verify
  `sglang.__file__` resolves to that checkout before serving.
- A repeated request can hit SGLang's local tree without retrieving external
  KV. Require a successful local flush, `device=0`, `storage>0`, completed
  backend retrieve operations, and matching generated tokens against a cold
  baseline before claiming an external cache hit.
