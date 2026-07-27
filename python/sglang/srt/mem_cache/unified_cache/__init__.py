"""Components of Unified Radix Cache.

- **TreeCore** (``unified_tree_core.UnifiedTreeCore``): the pure tree mechanism. Owns
  the tree structure, per-node values, the LRU(s), bookkeeping, and KV-cache
  events -- the only object that touches that state; no scheduling policy or IO,
  and never touches the cache. References the cache-owned component drivers to
  drive their tree-level hooks.
- **Components** (``unified_cache_components.TreeComponent`` subclasses): the
  per-component (FULL/SWA/MAMBA) drivers. Built and owned by the Controller;
  hold the cache for cache-level logic and the TreeCore for tree state.
- **Controller** (``unified_radix_cache.UnifiedRadixCache``): the scheduling brain
  and scheduler-facing facade. Owns the scheduling policy (when to evict / backup /
  load / prefetch), the IO (HiCache controller, device/host allocators, and the
  device<->host transfer), the async ack machinery, prefetch/storage, the
  distributed groups, and the component drivers; holds and drives the TreeCore.

TODO(Jialin): move all unified radix cache logic into this folder.
"""
