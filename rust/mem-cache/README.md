# mem-cache

Rust tree core for the Unified Radix Cache, covering the Full attention
component. It implements the tree side of the
`UnifiedTreeCoreInterface` split — match/insert walks, node arena, locks,
eviction walks, HiCache backup/load-back specs, and KV events — behind a
PyO3 binding, while the cache orchestration stays in Python.

## Usage

Select the backend with:

```bash
SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=rust
```

The extension builds itself with cargo on first use (see
`python/sglang/srt/mem_cache/rust_tree_core/extension.py`); libtorch and the
Python headers come from the running interpreter's torch install.

## Development

```bash
# Build (libtorch from the installed torch package):
LIBTORCH=$(python -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).parent)') \
  cargo build --release

# Tests (pyo3's extension-module cannot link test binaries):
LIBTORCH=... LD_LIBRARY_PATH=$LIBTORCH/lib cargo test --no-default-features
```

Unit tests live in `src/tests/`, mirroring the source layout one file per
module (wired via `#[cfg(test)] #[path = ...]`), so implementation files
stay free of inline test blocks.

Scope: only the `[Full]` component set is supported so far; the SWA and
Mamba components exist in the reference implementation and will follow.
