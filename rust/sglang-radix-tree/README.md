# sglang-radix-tree

Rust tree core for the Unified Radix Cache, covering Full attention, sliding window attention, and Mamba components. It implements the tree side of the `UnifiedTreeCoreInterface` split — match/insert walks, node arena, locks, eviction walks, HiCache backup/load-back specs, and KV events — behind a PyO3 binding, while the cache orchestration stays in Python.

## Usage

Rust is the default tree core. The centralized tree-core registry falls back to
Python in these cases:

- Session-aware caching.
- C128 or other unsupported components.
- Python-only named components or legacy class overrides.
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

## Named component overrides

`CacheInitParams.component_registry_override` maps component kinds to factory names, for example `{ComponentType.FULL: "custom_full"}`. A same-kind enum selector, such as `{ComponentType.FULL: ComponentType.FULL}`, resolves to the built-in key `"full"`. Cross-kind enum aliases are rejected because component kinds identify fixed storage slots.

Three separate registries connect backend selection and component construction:

- `register_tree_core_backend(name, factory)` selects a tree-core implementation.
- `register_python_tree_component(name, factory)` registers a callable that receives `PythonTreeComponentArgument` and returns a Python `TreeComponent`; existing component classes with `(cache, params)` constructors are also accepted.
- `register_rust_tree_component(name, factory)` registers a callable that receives `TreeComponentArgument` and returns a native `TreeComponentBinding` handle.

Python's `PythonTreeComponentArgument`, defined in `unified_cache/components/base.py`, requires `component_type`, `params`, and `cache`. The independent `TreeComponentArgument`, defined in `rust_tree_core/component_registry.py`, requires `component_type`, `params`, `native_init_params`, `native_bindings`, and `is_bigram`.

Factories run during construction; Rust retains the native component and performs tree operations without Python node callbacks. Factories should construct a fresh component for each cache. Both backends validate the returned component kind; Rust also validates its key mode. New native behavior requires a constructor compiled into the extension; a Python factory cannot turn a Python `TreeComponent` subclass into a native implementation.

For example, these factories select the existing Full implementations under a shared name:

```python
from sglang.srt.mem_cache.rust_tree_core.component_registry import (
    TreeComponentArgument,
    register_rust_tree_component,
)
from sglang.srt.mem_cache.unified_cache.components.base import PythonTreeComponentArgument
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_cache.components.registry import register_python_tree_component


def python_full(args: PythonTreeComponentArgument):
    return FullComponent(args.cache, args.params)


def native_full(args: TreeComponentArgument):
    return args.native_bindings.TreeComponentBinding.full(
        args.native_init_params, args.is_bigram
    )


register_python_tree_component("custom_full", python_full)
register_rust_tree_component("custom_full", native_full)
```

Register the Python counterpart first: a full cache always needs Python cache orchestration hooks, including with a Rust tree core. A Python-only name selects Python before the native extension is loaded. Direct `RustUnifiedTreeCore` construction does not require Python cache hooks.

Registration rejects conflicting names unless `replace=True` is supplied. Replacement affects future caches; existing caches retain their components. Replacing a Python factory invalidates its previous Rust pairing until the corresponding Rust factory is registered again. The built-in keys are `full`, `swa`, and `mamba`, and enum selectors resolve through these replaceable keys; no custom native implementation is enabled by default.

Finish registration and replacement before constructing a cache; factories must not mutate registries or register concurrently with cache construction.

Overrides are programmatic configuration, not a server CLI flag. A registered radix-cache backend can populate `ctx.params.component_registry_override` before calling `create_unified_radix_cache(ctx)`; `--radix-cache-backend` selects that backend. Registration must run in each scheduler process, for example through an installed `sglang.srt.plugins` entry point. The C128 and MLX factories supply named Python defaults while preserving explicit caller overrides. Legacy class-valued overrides remain supported and select Python.
