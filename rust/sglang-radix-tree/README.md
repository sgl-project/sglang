# sglang-radix-tree

Rust tree core for the Unified Radix Cache, covering Full attention, sliding window attention, and Mamba components. It implements the tree side of the `UnifiedTreeCoreInterface` split — match/insert walks, node arena, locks, eviction walks, HiCache backup/load-back specs, and KV events. The same generic core is available as a native Rust library without PyTorch; the PyO3/Tensor backend remains the production SGLang binding.

## Usage

Select the backend with:

```bash
SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=rust
```

SGLang wheels bundle the production extension. A source checkout falls back to
the shared fingerprinted Rust-extension cache; it never writes a shared object
into the Python package. LibTorch and the Python headers come from the running
interpreter's PyTorch install. PyTorch 2.11 through 2.13 are accepted explicitly,
and `torch_2_13_compat.h` covers two alignment APIs removed in PyTorch 2.13.

## Development

```bash
# Torch-free native core and simulation tests:
cargo test --manifest-path rust/sglang-radix-tree/Cargo.toml \
  --locked --no-default-features

# Build (libtorch from the installed torch package):
cd rust/sglang-radix-tree
LIBTORCH_USE_PYTORCH=1 \
  LIBTORCH_BYPASS_VERSION_CHECK=1 \
  CXXFLAGS="-include $PWD/torch_2_13_compat.h" \
  cargo build --release --locked --features python-extension

# Native tests do not enable pyo3's extension-module feature:
TORCH_ROOT=$(python3 -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)')
LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 \
  CXXFLAGS="-include $PWD/torch_2_13_compat.h" \
  LD_LIBRARY_PATH="$TORCH_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  cargo test --locked --no-default-features --features torch
```

Native consumers instantiate `UnifiedTreeCore<K, PageValue<PageId>>` with `new`. `full_kv_prefix_len` scores a key's device-resident Full-KV prefix without mutating the tree; that is the admitted prefix on FULL-only trees, while SWA and Mamba validators can admit less, so score those trees with `match_prefix`. `match_prefix` also returns the retained `NodeId` used by `insert_suffix_from_node` for suffix-only decode or chunked-prefill growth. A chunked continuation insert touches only the anchor and is O(1); a non-chunked one replays the root walk's LRU refresh, hit counts, and write-through triggers on the retained prefix, so grow a request with `chunked = true` and finish it with `chunked = false`. Component overlap hooks on the prefix are never replayed because they need the prefix KV. Pool allocation, request leases, and event publication remain downstream responsibilities.

When a simulator keys the tree by one precomputed hash per KV page, each hash is already one radix atom: configure the core with `page_size = 1`, store one `PageId` per hash, and convert page counts to token counts at the simulator boundary.

The `inspection` Cargo feature adds white-box methods for the shared Python/Rust
cache suite. Production wheels do not enable it.

Unit tests live in `src/tests/`, mirroring the source layout one file per module (wired via `#[cfg(test)] #[path = ...]`), so implementation files stay free of inline test blocks. Tensor parity tests require `--features torch`; Torch-free `PageValue` tests run with no features.

Supported component sets are `[Full]`, `[Full, SWA]`, `[Full, Mamba]`, and `[Full, SWA, Mamba]`.
