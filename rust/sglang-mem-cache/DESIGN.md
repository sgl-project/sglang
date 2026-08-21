# Rust Unified Radix Cache — Design

> Status: **design share / early port.** The Rust core logic and the Python
> orchestrator are in the tree for review. The build/packaging wiring, the
> registry hook, and tests are intentionally deferred (see
> [What's shared vs deferred](#whats-shared-vs-deferred)). This document
> describes the design so it can be reviewed end-to-end from the ported code.

## 1. Motivation

SGLang's prefix cache (`python/sglang/srt/mem_cache/`) is pure Python: the radix
tree, per-node lock-ref accounting, the LRU lists, and eviction all run on the
Python interpreter on the scheduler's hot path. At high request rates the tree
bookkeeping (`match_prefix` / `insert` / `evict` / `inc_lock_ref` /
`dec_lock_ref`) becomes a measurable fraction of scheduler overhead.

This component moves that bookkeeping into Rust while keeping the KV memory pools
and the allocator in Python. It mirrors the design of the existing pure-Python
`UnifiedRadixCache` (`python/sglang/srt/mem_cache/unified_radix_cache.py`) and
its component model (`unified_cache_components/`) — that Python implementation is
the behavioral specification, and the Rust source documents, method by method,
which Python method it mirrors.

The goal is a **drop-in `BasePrefixCache`** whose hot paths are Rust, selectable
without changing any scheduler code.

## 2. Layered architecture

```
┌─ Python ────────────────────────────────────────────────────────────────┐
│  RustUnifiedRadixCache(BasePrefixCache)        (the orchestrator)         │
│    python/sglang/srt/mem_cache/rust_unified_radix_cache.py                │
│    • owns req_to_token_pool + token_to_kv_pool_allocator (unchanged)      │
│    • translates the BasePrefixCache contract to Rust wrapper calls        │
│    • applies the DeferredAction list returned by Rust to the allocator    │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │  PyO3
┌─ Rust (this crate) ──────────▼───────────────────────────────────────────┐
│  PyO3 wrapper layer        radix_cache_wrapper.rs                          │
│    RustPageRadixCacheWrapper / RustBigramRadixCacheWrapper                 │
│    + result classes (RustMatchResult/RustInsertResult/RustEvictResult/…)  │
│    + torch bridge          py_interop.rs (PyTensor, vendored pyo3-tch)     │
│  ───────────────────────────────────────────────────────────────────────  │
│  Core radix cache          radix_cache.rs   RadixCache<K>                  │
│    • component model        components/{mod,full,swa,mamba}.rs             │
│    • node arena + per-component state   tree_node_pool.rs                  │
│    • per-component intrusive LRU         tree_node_lru.rs                   │
│    • deferred-action protocol            deferred_action.rs                │
│    • typed errors + ComponentType        error.rs / component_type.rs      │
└───────────────────────────────────────────────────────────────────────────┘
```

Four layers, top to bottom: **orchestrator** (Python, owns the pools),
**wrapper** (PyO3 boundary), **torch bridge** (`torch.Tensor` ↔ `tch::Tensor`),
**core** (the data structure and algorithms).

## 3. Ownership split

The single most important contract: **Rust owns the tree; Python owns the KV
memory.**

| Owned by Rust (inside `RadixCache<K>`)          | Owned by Python (orchestrator) |
|-------------------------------------------------|--------------------------------|
| tree structure (`parent`, `children`, `key`)    | `req_to_token_pool`            |
| per-component value tensors (KV indices)        | `token_to_kv_pool_allocator`   |
| per-component `lock_ref` accounting             | the host KV pool / HiCache controller |
| per-component intrusive LRU lists               | request lifecycle (`req.last_node`, prefix indices) |
| per-component evictable / protected size aggregates | metric collection |
| eviction policy (LRU) and the eviction loop     | |

The Rust cache **never frees KV slots itself** — it cannot, it doesn't own the
allocator. Instead it returns a list of [deferred actions](#6-the-deferredaction-contract)
describing what the orchestrator must free or back up. This keeps the allocator
single-owner in Python and keeps the Rust/Python boundary to one crossing per
operation.

## 4. The Rust core

### 4.1 Generic over the child-key type

`RadixCache<K: ChildKeyType>` is generic over the edge-key type, with two
production instantiations:

| Alias | `K` | Use |
|-------|-----|-----|
| `PageRadixCache`  | `Vec<i64>`        | standard path (`page_size >= 1`; `page_size = 1` uses one-element page keys) |
| `BigramRadixCache`| `Vec<(i64, i64)>` | EAGLE speculative decoding (children keyed by overlap bigram pairs `(t[i], t[i+1])`) |

### 4.2 Component model

The cache supports multiple *components* sharing one tree. A component is a slice
of per-node state plus a set of policy hooks. `ComponentType` (`component_type.rs`)
pins the discriminants:

```rust
pub enum ComponentType { Full = 0, Swa = 1, Mamba = 2 }
pub const NUM_COMPONENT_TYPES: usize = 3;
```

`RadixCache` holds a config-driven `components: Vec<Box<dyn Component<K>>>`
(built by `build_components` from `sliding_window_size` / `mamba_cache_chunk_size`),
mirroring the Python `UnifiedRadixCache._components_tuple`. The `Component<K>`
trait (`components/mod.rs`) is the per-component policy surface:

- `create_match_validator` — component-aware prefix matching
- `consume_value` / `should_skip_leaf_creation` / `commit_insert_data_on_new_leaf` — insert hooks
- `inc_lock_ref` / `dec_lock_ref` — lock-walk policy (FULL walks to root; SWA is window-bounded)
- `evict` — per-component eviction
- `bump_mru_walk` / `redistribute_on_node_split` — recency + node-split maintenance

Concrete impls: `FullComponent`, `SwaComponent`, `MambaComponent`.

### 4.3 Per-node and per-pool state

Each node carries a fixed-size array of per-component state; the pool carries the
symmetric per-component aggregates (LRU sentinels + size totals):

```rust
pub struct ComponentNodeState {
    pub value: Option<Tensor>,  // KV indices for this component; None = tombstone/never-populated
    pub lock_ref: u32,
    pub lru_data: LRUData,      // intrusive doubly-linked-list links
}
pub struct TreeNode<K> { parent, children, key, components: [ComponentNodeState; NUM_COMPONENT_TYPES] }
```

Allocating all `NUM_COMPONENT_TYPES` slots unconditionally trades a small amount
of memory (~12–15 MB at 100K nodes) for zero-cost static dispatch and a simple
node layout. The `Option<Tensor>` shape deliberately matches OSS SWA, which uses
`is_tombstone = value is None` — there is no separate "was-populated" bit.

### 4.4 Intrusive LRU

Eviction uses per-component intrusive LRU lists rather than Python's
access-time heap. `LRUSlot` (`tree_node_lru.rs`) is a marker trait — one
zero-sized struct per component (`FullLRUSlot`, `SwaLRUSlot`, `MambaLRUSlot`,
plus `HostFullLRUSlot` for the host tier) — that routes to its
`ComponentType`'s sentinels and size aggregates via `Self::COMPONENT`. The whole
eviction loop runs in Rust (`evict_full` / `evict_non_full` / `evict_host_full`);
PyO3 is crossed once per `evict` call, never mid-loop.

## 5. The PyO3 wrapper layer

`radix_cache_wrapper.rs` exposes the two production wrappers
(`RustPageRadixCacheWrapper`, `RustBigramRadixCacheWrapper`) with an **identical
Python surface**, so the orchestrator picks one at construction (by
`is_eagle`) and needs no per-method dispatch. The bigram wrapper builds the
`(t[i], t[i+1])` pairs and trims the value tensor `N → N−1` internally, so the
boundary always carries a flat 1-D int64 key.

Transport:
- **Keys** cross as a C-contiguous int64 buffer (`array.array('q')`), parsed via `PyBuffer<i64>`.
- **Values** (KV-index tensors) cross via `PyTensor` (the torch bridge).
- Hot calls (`match_prefix`, `insert`) wrap the Rust work in `py.allow_threads(...)` to release the GIL.

Result classes returned across the boundary: `RustMatchResult`,
`RustInsertResult`, `RustEvictResult`, `RustPrepareLoadBackResult`. Each carries
the deferred-action list as plain tagged tuples for the orchestrator to apply.

The torch bridge `py_interop.rs` (`PyTensor`) is vendored verbatim from
[`pyo3-tch`](https://github.com/LaurentMazare/pyo3-tch) (MIT/Apache-2.0); it is
the only piece of the boundary that touches libtorch's Python interop.

## 6. The DeferredAction contract

Because Rust cannot touch the allocator, every mutation the orchestrator must
perform is returned as a tagged action (`deferred_action.rs`). The orchestrator
pattern-matches the string tag and calls the right allocator method:

| Action | Orchestrator does |
|--------|-------------------|
| `FullDupFreed` | `allocator.free(freed_indices)` — duplicate slots a prior insert already owned |
| `SwaRecover` | `allocator.free(freed_full)` + stamp translated SWA value |
| `SwaStamp` | stamp translated SWA value on the node |
| `FullWriteThroughBackup` | back the device value up to the host tier (write-through) |
| `FullDeviceEvictOnBackedUp` | `allocator.free(device_value)` — already backed up |
| `FullHostEvict` | `host_pool.free(host_value)` |
| `FullWriteBackOnEvict` | write the value back to host on eviction (write-back) |

Eviction additionally returns `freed[ct]` lists indexed by `ComponentType`, which
the orchestrator drains per component: FULL → `allocator.free`, SWA →
`allocator.free_swa`, Mamba → `mamba_pool.free`.

## 7. The Python orchestrator

`RustUnifiedRadixCache(BasePrefixCache)` is a thin translation layer. Beyond
forwarding the `BasePrefixCache` methods to `self._rust_radix`, it:

- **Treats `req.last_node` as an opaque Rust `NodeIdx` (an int), not a Python
  `TreeNode`.** Code that previously walked `req.last_node.X` (HiCache/Mamba/LMC
  internals) does not apply. ⚠️ The handle is a raw pool index — using a stale
  index after the node is evicted is a caller bug (ABA); the lifecycle is bounded
  to a single request by the orchestrator. A generation-tagged handle is a
  listed follow-up.
- **Carries `swa_uuid_for_lock`** returned by `inc_lock_ref` back into
  `dec_lock_ref`, so SWA's release walk stops at the correct window boundary.
  FULL-only configs pass `None`.
- **Normalizes keys and page-alignment** at the boundary (`maybe_to_bigram_view`,
  `RadixKey.page_aligned`) so the atom-unit accounting matches the wrapper.

### Supported configuration and gating

The Rust core implements FULL attention, SWA (sliding window), Mamba, EAGLE
bigram, and device↔host HiCache. The orchestrator gates the combinations it has
validated; unsupported combinations fail fast with a typed
`RadixCacheInfraPyError` at construction (`_reject_unsupported` /
`_reject_unsupported_hicache`) rather than corrupting state mid-run. Notably
rejected today: non-LRU eviction, `cache_ttl_seconds`, `enable_kv_cache_events`,
insert priority, and (for HiCache) Mamba/SWA host tiers and an L3 storage backend.

## 8. Key design decisions

| Decision | Rationale |
|----------|-----------|
| Per-component state (`value`, `lock_ref`, LRU, sizes) lives in **Rust** | lock_ref was already Rust-side; Python-side state would mean cross-boundary chatter on hot paths |
| Component **policy** lives in Rust (`Component<K>` trait), not a Python ABC | the earlier plan put a `RustTreeComponent` ABC in Python; moving it into Rust avoids per-hook round-trips during insert/evict |
| `ComponentType` array slots allocated unconditionally | zero-cost static dispatch; simple node layout; memory cost is acceptable at real workloads |
| `Option<Tensor>` for the per-component value (not a 3-variant enum) | matches OSS `is_tombstone = value is None`; no unused history bit |
| One PyO3 crossing per operation; eviction loop fully Rust-side | bounded boundary cost regardless of how many leaves an evict frees |
| Allocator stays single-owner in Python via `DeferredAction` | Rust never frees KV memory; keeps ownership clean |
| `NodeIdx` opaque integer handle across the boundary | Rust owns the tree; faking Python node objects would force round-trips |

## 9. File map

| File | Responsibility |
|------|----------------|
| `src/radix_cache.rs` | `RadixCache<K>`; `match_prefix` / `insert` / `evict` / lock-ref; `PageRadixCache` / `BigramRadixCache` aliases |
| `src/components/mod.rs` | `Component<K>` trait + `MatchValidator` |
| `src/components/{full,swa,mamba}.rs` | concrete component policies |
| `src/tree_node_pool.rs` | node arena, `TreeNode<K>`, `ComponentNodeState`, per-pool component state |
| `src/tree_node_lru.rs` | `LRUSlot` trait, per-component slots, eviction routines |
| `src/deferred_action.rs` | the Rust→Python action protocol |
| `src/component_type.rs` | `ComponentType` enum (exposed to Python) |
| `src/error.rs` | typed Rust errors + the PyO3 exception classes |
| `src/radix_cache_wrapper.rs` | PyO3 wrappers + result classes (the Python-facing API) |
| `src/py_interop.rs` | `PyTensor` torch bridge (vendored `pyo3-tch`) |
| `src/lib.rs` | `#[pymodule] _mem_cache_core` registering the wrappers, results, `ComponentType`, exceptions |

## 10. What's shared vs deferred

**Shared now (in the tree for review):** the full Rust core, the PyO3 wrapper
layer + torch bridge, and the Python orchestrator.

**Deferred (follow-up):**
1. **Packaging** — a `[[tool.setuptools-rust.ext-modules]]` block in
   `python/pyproject.toml` targeting `sglang.srt.mem_cache._mem_cache_core`
   (mirrors how `rust/sglang-grpc` ships as `sglang.srt.grpc._core`). Until then
   the orchestrator's `from sglang.srt.mem_cache._mem_cache_core import …` does
   not resolve.
2. **Registration** — `install_rust_radix_cache()` exists but is not invoked, so
   the `"rust_unified"` backend is not yet selectable via `--radix-cache-backend`.
3. **Tests** — the Rust unit tests and the Python parity tests are not ported.
4. **pyo3 alignment** — this crate pins pyo3 `0.22` to match the verbatim port;
   `rust/sglang-grpc` uses `0.23`. Aligning the two is a follow-up.

## 11. Relationship to the existing OSS code

This crate is the Rust backing for the existing component design already in OSS:
`python/sglang/srt/mem_cache/unified_radix_cache.py` and
`unified_cache_components/README.md` describe the same FULL/SWA/Mamba component
model in pure Python. That code remains the behavioral reference; the Rust source
cites the specific Python method each routine mirrors.
