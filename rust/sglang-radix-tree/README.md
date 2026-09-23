# sglang-radix-tree

Rust tree core for the Unified Radix Cache, covering Full attention, sliding window attention, and Mamba components. It implements the tree side of the `UnifiedTreeCoreInterface` split — match/insert walks, node arena, locks, eviction walks, HiCache backup/load-back specs, and KV events — behind a PyO3 binding, while the cache orchestration stays in Python.

## Usage

Rust is the default tree core. The centralized tree-core registry falls back to
Python in these cases:

- Session-aware caching.
- C128 or other unsupported components.
- Custom component overrides.
- Non-Linux platforms.
- PyTorch versions outside 2.11 through 2.13.
- Devices other than CPU or CUDA.
- Installations containing neither the Rust extension nor its sources.

This policy also applies when Rust is explicitly selected.
Build, import, and runtime failures in supported configurations remain errors.

T-LRU supports integer and floating-point `threshold` and `next_prompt_estimate`
parameters. Integer configurations retain exact arithmetic; floating-point
configurations preserve Python's operation order, rounding, and comparison with
integer cache lengths, including NaN and infinity. Mixed configurations whose
integer-to-float conversions overflow within the native history-length range
raise `OverflowError` at initialization. Per-node path depth and branch history
preserve the tail budget across splits, host refills, and repeated eviction. The
ancestor history walk runs only when T-LRU is selected.

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
interpreter's PyTorch install. PyTorch 2.11 through 2.13 are accepted explicitly,
and `torch_2_13_compat.h` covers two alignment APIs removed in PyTorch 2.13.

## Development

```bash
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
  cargo test --locked
```

The `inspection` Cargo feature adds white-box methods for the shared Python/Rust
cache suite. Production wheels do not enable it.

Unit tests live in `src/tests/`, mirroring the source layout one file per module (wired via `#[cfg(test)] #[path = ...]`), so implementation files stay free of inline test blocks.

Supported component sets are `[Full]`, `[Full, SWA]`, `[Full, Mamba]`, and `[Full, SWA, Mamba]`.

One insertion-ordered `NodeSet` tracks device leaves, host leaves, and Full host
duplicates. Leaf eviction ranks candidates in the policy heap; duplicate
reclamation consumes insertion order directly, matching Python's dictionary.

SWA buffer-mode load-back can repair tombstoned windows in Rust. The core finds
missing SWA spans and attaches loaded slots across node boundaries, preserving
Full-KV ownership, lock accounting, and pending write-through split actions.
The shared Python pipeline handles transfers and redundant-slot cleanup.
