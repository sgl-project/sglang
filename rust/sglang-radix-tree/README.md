# sglang-radix-tree

Rust tree core for the Unified Radix Cache, covering Full attention, sliding window attention, and Mamba components. It implements the tree side of the `UnifiedTreeCoreInterface` split — match/insert walks, node arena, locks, eviction walks, HiCache backup/load-back specs, and KV events — behind a PyO3 binding, while the cache orchestration stays in Python.

## Usage

Rust is the default tree core. The centralized tree-core registry falls back to
Python in these cases:

- Session-aware caching.
- C128 or other unsupported components.
- Custom component overrides.
- Non-Linux platforms.
- PyTorch versions outside 2.11 through 2.14.
- Devices other than CPU or CUDA.
- Installations containing neither the Rust extension nor its sources.
- Source builds with a missing or unusable Rust toolchain.

This policy also applies when Rust is explicitly selected.
Build, import, and runtime failures in supported configurations remain errors.
The Rust bindings report unexpected native panics as `RuntimeError` so Python's
crash handlers can report them and coordinate shutdown. A panic during a core
operation poisons its mutex, and subsequent calls refuse to reuse that state.

T-LRU accepts integer and floating-point `threshold` and `next_prompt_estimate`
parameters. In mixed integer/float configurations, the integer estimate must fit
in i128; larger values raise `OverflowError` at initialization. Native integer
addition is checked, and overflow is reported as `RuntimeError`.

Select a backend explicitly with:

```bash
SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=rust
# Use the Python implementation instead:
SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
```

Standard SGLang wheels bundle the production extension; some platform
distributions omit it. A source checkout falls back to
the shared fingerprinted Rust-extension cache; it never writes a shared object
into the Python package. LibTorch and the Python headers come from the running
interpreter's PyTorch install. PyTorch 2.11 through 2.14 are accepted explicitly,
and `torch_compat.h` covers two alignment APIs removed in PyTorch 2.13 plus
the `cholesky` and `qr` aliases removed in 2.14. Torch 2.12 and later declare
C++20, so the builds below pass `-std=c++20` over torch-sys's own `-std=c++17`.
Source checkouts need working `cargo` and `rustc` to use Rust, including for
fingerprinted cache lookup. If either tool is unavailable, the registry selects
Python. Trusted bundled extensions do not require a Rust compiler.

## Development

Alternative backends implement `UnifiedTreeCoreInterface` and register through
`register_tree_core_backend`. For built-in Mamba and SWA internal-state write-back,
the core returns a backup request and the controller performs transfers and waits
for acknowledgment.
The controller then calls `finish_mamba_state_eviction` or
`finish_swa_state_eviction` to resume eviction. This keeps I/O outside the Rust
tree's mutex; both tree cores implement the same contract.

Native code compiled into this extension can construct a core with
`UnifiedTreeCore::with_component_factory(params, component_types, factory)`.
The factory receives each `ComponentType` and `&CacheInitParams`, returns a native
component, and can capture additional construction settings. Use
`create_tree_component` for component kinds that retain their default behavior.
Factories are supplied per construction; the core retains the resulting components.

Both concrete binding classes also provide a Rust-only `with_component_factory`
constructor, preserving binding validation and Python error conversion. Python
constructor signatures and class-based component overrides remain unchanged.
This hook customizes implementations of the existing component kinds; it does
not introduce a global registry or Python callbacks in tree operations.

```bash
# Build (libtorch from the installed torch package):
cd rust/sglang-radix-tree
LIBTORCH_USE_PYTORCH=1 \
  LIBTORCH_BYPASS_VERSION_CHECK=1 \
  CXXFLAGS="-std=c++20 -include $PWD/torch_compat.h" \
  cargo build --release --locked --features python-extension

# Native tests do not enable pyo3's extension-module feature:
TORCH_ROOT=$(python3 -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)')
LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 \
  CXXFLAGS="-std=c++20 -include $PWD/torch_compat.h" \
  LD_LIBRARY_PATH="$TORCH_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  cargo test --locked
```

The `inspection` Cargo feature adds white-box methods for the shared Python/Rust
cache suite. Production wheels do not enable it.

Unit tests live in `src/tests/`, mirroring the source layout one file per module (wired via `#[cfg(test)] #[path = ...]`), so implementation files stay free of inline test blocks.

Supported component sets are `[Full]`, `[Full, SWA]`, `[Full, Mamba]`, and `[Full, SWA, Mamba]`.
