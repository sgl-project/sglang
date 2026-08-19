# mem-cache

Rust tree core for the Unified Radix Cache, covering Full attention, sliding window attention, and Mamba components. It implements the tree side of the `UnifiedTreeCoreInterface` split — match/insert walks, node arena, locks, eviction walks, HiCache backup/load-back specs, and KV events — behind a PyO3 binding, while the cache orchestration stays in Python.

## Usage

Select the backend with:

```bash
SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=rust
```

The extension builds itself with cargo on first use (see `python/sglang/srt/mem_cache/rust_tree_core/extension.py`); libtorch and the Python headers come from the running interpreter's torch install. The build injects `torch_2_13_compat.h` for two alignment APIs removed in PyTorch 2.13.

## Development

```bash
cd rust/mem-cache

# Build (libtorch from the installed torch package):
LIBTORCH=$(python -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).parent)') \
  LIBTORCH_BYPASS_VERSION_CHECK=1 \
  CXXFLAGS="-include $PWD/torch_2_13_compat.h" \
  cargo build --release

# Tests (pyo3's extension-module cannot link test binaries):
LIBTORCH=... LIBTORCH_BYPASS_VERSION_CHECK=1 \
  CXXFLAGS="-include $PWD/torch_2_13_compat.h" \
  LD_LIBRARY_PATH=$LIBTORCH/lib \
  cargo test --no-default-features
```

Unit tests live in `src/tests/`, mirroring the source layout one file per module (wired via `#[cfg(test)] #[path = ...]`), so implementation files stay free of inline test blocks.

Supported component sets are `[Full]`, `[Full, SWA]`, `[Full, Mamba]`, and `[Full, SWA, Mamba]`.
