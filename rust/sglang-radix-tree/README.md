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
- Source builds with a missing or unusable Rust toolchain.

This policy also applies when Rust is explicitly selected.
Build, import, and runtime failures in supported configurations remain errors.
The Rust bindings report unexpected native panics as `RuntimeError` so Python's
crash handlers can report them and coordinate shutdown. A panic during a core
operation poisons its mutex, and subsequent calls refuse to reuse that state.

T-LRU supports integer and floating-point `threshold` and `next_prompt_estimate`
parameters. Integer configurations retain exact arithmetic; floating-point
configurations preserve Python's operation order, rounding, and comparison with
integer cache lengths, including NaN and infinity. In mixed integer/float
configurations, the integer estimate must fit in i128; larger values raise
`OverflowError` at initialization. The native integer addition is checked and
panics if the actual history plus estimate overflows. Priority evaluation uses
only native integer/float arithmetic and does not allocate. Per-node path depth
and branch history preserve the tail budget across splits, host refills, and
repeated eviction. The ancestor history walk runs only when T-LRU is selected.

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
Source checkouts need working `cargo` and `rustc` to use Rust, including for
fingerprinted cache lookup. If either tool is unavailable, the registry selects
Python. Trusted bundled extensions do not require a Rust compiler.

## Development

External backends registered through `register_tree_core_backend` must implement
the new `UnifiedTreeCoreInterface.swa_tombstone_ranges` and `attach_swa_window`
methods for SWA buffer-mode repair. These methods are abstract, so existing
subclasses need to add them before they can be instantiated. The new
`finish_mamba_state_eviction` and `finish_swa_state_eviction` hooks are optional
for backends that complete these backups inline and never return a deferred
`mamba_backup_node_id` or `swa_backup_node_id`.

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

HiCache write-back preserves eligible internal SWA windows before device eviction
(#40712). Rust pauses the walk while the Python controller makes room for the
whole unbacked window and completes its host backup. The walk then resumes,
retaining reusable host state when the backup succeeds and still freeing device
slots when allocation fails. The same transfer and ACK handling serves internal
Mamba state backups (#40680).
