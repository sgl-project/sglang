# sglang-radix-tree

Rust tree core for the Unified Radix Cache, covering Full attention, sliding window attention, and Mamba components. It implements the tree side of the `UnifiedTreeCoreInterface` split — match/insert walks, node arena, locks, eviction walks, HiCache backup/load-back specs, and KV events. The same generic core is available as a native Rust library without PyTorch; the PyO3/Tensor backend remains the production SGLang binding.

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

```bash
# Torch-free native core and simulation tests:
cargo test --manifest-path rust/sglang-radix-tree/Cargo.toml \
  --locked --no-default-features

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
  cargo test --locked --no-default-features --features torch
```

Native consumers instantiate `UnifiedTreeCore<K, PageValue<PageId>>` with `new`. `full_kv_prefix_len` scores a key's device-resident Full-KV prefix without mutating the tree; that is the admitted prefix on FULL-only trees, while SWA and Mamba validators can admit less, so score those trees with `match_prefix`. `match_prefix` also returns the retained `NodeId` used by `insert_suffix_from_node` for suffix-only decode or chunked-prefill growth. A continuation insert replays the root walk's LRU refresh, priority updates, hit counts, and write-through triggers on the retained prefix. Only nodes ending past `inserted_len` count a hit; advance that watermark after each insert so a request counts each node once. Both anchor validation and prefix replay walk the ancestor path. Component overlap hooks on the prefix are never replayed because they need the prefix KV. Pool allocation, request leases, and event publication remain downstream responsibilities.

When a simulator keys the tree by one precomputed hash per KV page, each hash is already one radix atom: configure the core with `page_size = 1`, store one `PageId` per hash, and convert page counts to token counts at the simulator boundary.

The `inspection` Cargo feature adds white-box methods for the shared Python/Rust
cache suite. Production wheels do not enable it.

Unit tests live in `src/tests/`, mirroring the source layout one file per module (wired via `#[cfg(test)] #[path = ...]`), so implementation files stay free of inline test blocks. Tensor parity tests require `--features torch`; Torch-free `PageValue` tests run with no features.

Supported component sets are `[Full]`, `[Full, SWA]`, `[Full, Mamba]`, and `[Full, SWA, Mamba]`.

Full-only native device caches with cloneable values can use `snapshot_full_device` for speculative admission. The snapshot retains node handles and access order; immutable key/value backends can share buffers. Allocators and request leases remain caller-owned, and only one transaction branch may commit. Snapshots require idle insert and eviction walks. Native policies can enumerate `full_device_eviction_candidates` and use `evict_full_device_suffix` to evict an aligned suffix of an unlocked leaf, then reconsider the returned parent. This lets forecast-based simulators choose victims while the core enforces locks and updates tree accounting; normal SGLang eviction still removes whole leaves.
