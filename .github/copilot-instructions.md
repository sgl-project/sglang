# SGLang Development Guide

SGLang is a fast serving framework for large language models with three main components: Python runtime, sgl-kernel (C++/CUDA), and sgl-router (Rust). This project supports training, serving, and inference for LLMs and vision language models.

**ALWAYS reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Prerequisites and Environment Setup
- Install prerequisites: `sudo apt-get update && sudo apt-get install -y protobuf-compiler git libnuma-dev`
- Create virtual environment: `python3 -m venv sglang-venv && source sglang-venv/bin/activate`
- Install package managers: `pip install --upgrade pip && pip install uv`
- Set UV environment variable: `export UV_SYSTEM_PYTHON=true`

### Core Installation and Build Process
**NEVER CANCEL builds or installs - they require long timeouts due to network and compilation complexity.**

- **Install SGLang Python package:** `uv pip install -e "python[dev]" --index-strategy unsafe-best-match --timeout=600` 
  - **NEVER CANCEL: Takes 15-30 minutes with network delays. Set timeout to 45+ minutes.**
  - If network timeouts occur, retry with longer `--timeout` values
  - Use `pip install -e "python[dev]" --timeout=600` as fallback if uv fails

- **Build sgl-kernel (C++/CUDA components):** 
  ```bash
  cd sgl-kernel
  make submodule  # Initialize git submodules
  make build      # Build and install wheel package
  ```
  - **NEVER CANCEL: Takes 20-45 minutes depending on CUDA availability. Set timeout to 60+ minutes.**
  - Requires CUDA toolkit for GPU support
  - Contains 62 CUDA files, 18 C++ files - substantial compilation time

- **Build sgl-router (Rust component):**
  ```bash
  cd sgl-router
  cargo build --release  # Production build
  ```
  - **NEVER CANCEL: Takes 5-7 minutes. Set timeout to 15+ minutes.**
  - Requires protobuf-compiler: `sudo apt-get install protobuf-compiler`
  - Use `cargo check` for faster validation (1 minute)

### Pre-commit and Linting Workflow
- Install pre-commit: `pip install pre-commit`
- Setup hooks: `pre-commit install`
- **Run all checks:** `pre-commit run --all-files`
  - **NEVER CANCEL: Takes 5-15 minutes first run. Set timeout to 30+ minutes.**
  - Subsequent runs are faster (1-3 minutes)
  - May require network access to install hook environments
- **Format code manually:** `make format` (root directory)

### Testing
**NEVER CANCEL test suites - they are designed for thorough validation.**

- **Frontend tests:** `cd test/lang && python3 run_suite.py --suite per-commit`
  - **NEVER CANCEL: Takes ~10 minutes. Set timeout to 20+ minutes.**

- **Backend tests:** `cd test/srt && python3 run_suite.py --suite per-commit`
  - **NEVER CANCEL: Takes 30+ minutes per partition. Set timeout to 60+ minutes.**
  - CI runs 10 partitions in parallel, each taking ~30 minutes
  - Use `--auto-partition-id` and `--auto-partition-size` for subset testing

- **Component-specific tests:**
  - sgl-router: `cd sgl-router && cargo test` (~2 minutes)
  - sgl-kernel: `cd sgl-kernel && make test` (~5-10 minutes)

### Development Workflow Shortcuts
- **All component development setup:** `cd sgl-router && make dev-setup` (build + test)
- **Quick validation:** `cd sgl-router && make check` (cargo check + clippy)
- **Format Rust code:** `cd sgl-router && make fmt`
- **Clean builds:** `make clean` in respective component directories

## Validation

### Manual Testing Requirements
**ALWAYS run through complete end-to-end scenarios after making changes.**

- **Basic server functionality:** Launch a server and send test requests
  ```bash
  # Terminal 1: Launch server (requires model download)
  python3 -m sglang.launch_server --model Qwen/Qwen2-7B-Instruct
  
  # Terminal 2: Test client
  python3 -c "import sglang as sgl; print(sgl.__version__)"
  ```

- **Frontend testing:** Execute language constructs and prompt workflows
- **Backend testing:** Verify model loading, batching, and inference paths
- **Router testing:** Validate request routing and load balancing

### CI Validation Requirements
- **ALWAYS run format and lint checks:** `pre-commit run --all-files`
  - Must pass before PR submission or CI will fail
- **Test critical paths:** Run at least `per-commit` test suites
- **Build all components:** Ensure Rust, C++/CUDA, and Python builds succeed

## Common Tasks

### Performance and Benchmarking
- **Quick benchmarks:** `cd sgl-router && make bench-quick` (~5 minutes)
- **Full benchmark suite:** `cd sgl-router && make bench` (~15-30 minutes)
- **Benchmark reporting:** `make bench-report` (opens HTML report)

### Documentation
- **Build docs:** `cd docs && make compile && make html`
  - **NEVER CANCEL: Notebook compilation takes 10+ minutes. Set timeout to 30+ minutes.**
- **Serve docs locally:** `cd docs && make serve` (auto-rebuilding server)
- **Clean notebook outputs:** `find docs -name '*.ipynb' -exec nbstripout {} \;`

### Key Project Structure
```
├── python/sglang/          # Main Python package
├── sgl-kernel/             # C++/CUDA kernels (62 .cu, 18 .cpp files)
├── sgl-router/             # Rust routing component
├── test/                   # Test suites
│   ├── lang/              # Frontend language tests
│   └── srt/               # Backend runtime tests
├── docs/                   # Documentation and notebooks
├── docker/                # Container configurations
└── scripts/ci/            # CI automation scripts
```

### Environment Variables
- `CUDA_HOME`: Set to CUDA installation path (e.g., `/usr/local/cuda-12.6`)
- `UV_SYSTEM_PYTHON=true`: Required for uv package manager
- `SGLANG_DISABLE_REQUEST_LOGGING`: Disable verbose logging (default: false)
- See `docs/references/environment_variables.md` for complete list

### Build Troubleshooting
- **Network timeouts:** Increase pip/uv timeout values to 600+ seconds. PyPI access may be limited in some environments.
- **Pre-commit installation fails:** Network issues can prevent hook environment setup. Use `git commit --no-verify` if needed for urgent commits, but address before PR submission.
- **CUDA not found:** Install CUDA toolkit or use CPU-only builds
- **Protobuf missing:** Install with `sudo apt-get install protobuf-compiler`
- **Memory issues:** Reduce parallel build jobs with `export MAKEFLAGS='-j2'`
- **Permission issues:** Use virtual environment instead of system Python

### Git and Contribution Workflow
- **NEVER commit to main branch directly**
- Create feature branch: `git checkout -b feature/description`
- Format before commit: `pre-commit run --all-files`
- Test before PR: Run relevant test suites for your changes
- Update documentation if changing APIs or adding features

## Critical Timing Expectations

**ALWAYS use these timeout values - builds and tests take substantial time:**

| Operation | Expected Time | Recommended Timeout |
|-----------|---------------|-------------------|
| Python package install | 15-30 minutes | 45+ minutes |
| sgl-kernel build | 20-45 minutes | 60+ minutes |
| sgl-router check | ~30 seconds | 2+ minutes |
| sgl-router build | 5-7 minutes | 15+ minutes |
| sgl-router tests | ~5 minutes | 15+ minutes |
| Pre-commit first run | 5-15 minutes | 30+ minutes |
| Frontend tests | ~10 minutes | 20+ minutes |
| Backend tests (full) | 30+ minutes | 60+ minutes |
| Documentation compile | 10+ minutes | 30+ minutes |

**NEVER CANCEL long-running operations. If unsure, wait longer rather than interrupting builds.**