# Unified Radix Cache

A component-based, pluggable prefix cache framework for SGLang that unifies Full-attention, Sliding-Window-Attention (SWA), and Mamba/SSM caching into a single radix tree.

## Design Goals

1. **Unified tree structure** — One radix tree manages all KV cache types instead of separate specialized implementations (`SWARadixCache`, `MambaRadixCache`, etc.).
2. **Pluggable components** — Each attention/state type (Full, SWA, Mamba) is a `TreeComponent` that implements hook interfaces. Adding a new cache type only requires adding a new component.
3. **Per-component resource isolation** — Each component has its own LRU list, lock reference counting, evictable/protected size tracking, and eviction driver.
4. **Cascade eviction with priority** — When a component evicts a node, lower-or-equal-priority components on the same node are evicted together, maintaining cross-component consistency.
5. **Zero special-casing in the main tree** — The tree operates purely on keys (logical). All physical resource management (allocation, freeing, copy-on-write) is handled by components through hooks.

## Architecture

```
┌───────────────────────────────────────────────┐
│              UnifiedRadixCache                │
│            (unified_radix_cache.py)           │
│                                               │
│  root_node ──► UnifiedTreeNode (radix tree)   │
│  components ► {name → TreeComponent}          │
│  lru_lists ─► {name → UnifiedLRUList}         │
└──────────┬───────────┬───────────┬────────────┘
           │           │           │
           ▼           ▼           ▼
   ┌────────────┐ ┌──────────┐ ┌─────────────┐
   │    Full    │ │   SWA    │ │    Mamba    │
   │ Component  │ │Component │ │  Component  │
   └─────┬──────┘ └────┬─────┘ └──────┬──────┘
         │             │              │
         └─────────────┼──────────────┘
                       ▼
               ┌──────────────┐
               │TreeComponent │
               │    (ABC)     │
               └──────────────┘
```

### Key Data Structures

**`UnifiedTreeNode`** — Each node stores per-component data independently:

```python
node.component_data = {
    "full":  ComponentData(value=Tensor|None, lock_ref=int, metadata={}),
    "swa":   ComponentData(value=Tensor|None, lock_ref=int, metadata={}),
    "mamba": ComponentData(value=Tensor|None, lock_ref=int, metadata={}),
}
```

**`UnifiedLRUList`** — One doubly-linked list per component, threaded through the same tree nodes via `lru_prev[name]`/`lru_next[name]`. Supports O(1) insert/remove/promote and O(L) scan for eviction (L = locked nodes skipped).

**`ComponentData`** — Per-component data stored on each node:
- `value: Tensor | None` — Device indices into the component's memory pool (`TokenToKVPool` for Full, `SWAKVPool` for SWA, `MambaPool` for Mamba). `None` means tombstone (data evicted but node structure retained).
- `lock_ref: int` — Reference count of active requests using this node's component data. `lock_ref > 0` protects the node from eviction.
- `metadata: dict` — Component-specific state (e.g., SWA stores `component_uuid` for window-lock boundary tracking).

---

## File Layout

| File | Contents |
|------|----------|
| `../unified_radix_cache.py` | `UnifiedRadixCache`, `UnifiedTreeNode`, `UnifiedLRUList`, factory `create_unified_radix_cache` |
| `tree_component.py` | `TreeComponent` ABC, `ComponentType`, `ComponentData`, `get_and_increase_time_counter`, `next_component_uuid` |
| `full_component.py` | `FullComponent` — standard full-attention KV cache component |
| `swa_component.py` | `SWAComponent` — sliding-window attention component with tombstone/window tracking |
| `mamba_component.py` | `MambaComponent` — Mamba/SSM state component with copy-on-write |
| `hybrid_cache_controller.py` | `HybridCacheController` — HiCache 3-tier storage controller (L1 GPU → L2 CPU → L3 Disk) |
| `__init__.py` | Re-exports: `ComponentName`, `ComponentData`, `TreeComponent`, `FullComponent`, `SWAComponent`, `MambaComponent` |

---

## Public API Reference

All public APIs are on `UnifiedRadixCache`, which implements `BasePrefixCache`.

**Notation**: K = key length (tokens), D = matched path depth in tree (D ≤ K/P), P = page_size, C = number of components (≤ 3, treated as constant).

All tree traversal operations have two cost components: **O(K)** for data operations (key comparison, tensor clone/concat) + **O(D·C)** for component overhead (C hooks per node). Since D ≤ K/P and C is constant, overall **O(K)**.

### `match_prefix(params: MatchPrefixParams) → MatchResult`

Find the longest cached prefix for a token sequence.

| Aspect | Detail |
|--------|--------|
| **Purpose** | Walk the radix tree to find the longest prefix where **all** component validators pass |
| **Inputs** | `params.key: RadixKey` — token IDs + optional extra key for namespace isolation |
| **Output** | `MatchResult(device_indices, last_device_node, last_host_node, mamba_branching_seqlen, ...)` |
| **Mutation** | Updates `last_access_time` on matched path; promotes matched nodes to MRU in all component LRU lists; may trigger `_split_node` if match ends mid-node |
| **Complexity** | **O(K + D·C)** |

**Algorithm detail:**
1. Calls `create_match_validator()` once per component — returns a stateful closure (e.g., SWA tracks accumulated window length)
2. Walks tree edges via `key_match_fn`; at each node, calls all validator closures — the match boundary is only advanced when **all** validators return `True`
3. If match ends mid-node, calls `_split_node` → triggers `redistribute_on_node_split()` per component
4. Post-match (`_match_post_processor`):
   - Promotes matched path to MRU in each component's LRU via `node_has_component_data()` as filter
   - Updates `last_access_time` with decreasing timestamps up the path (parent < child)
   - Concatenates matched device indices via `torch.cat` (concat length ≤ K, subsumed by O(K))
   - Calls `finalize_match_result()` per component (Mamba performs copy-on-write: allocates new pool slot, copies SSM state)

---

### `insert(params: InsertParams) → InsertResult`

Insert a key-value pair into the tree.

| Aspect | Detail |
|--------|--------|
| **Purpose** | Insert token sequence + KV indices, reusing existing prefix and freeing duplicate KV slots |
| **Inputs** | `params.key: RadixKey`, `params.value: Tensor` (KV pool indices), plus component-specific fields (`mamba_value`, `swa_evicted_seqlen`, `prev_prefix_len`) |
| **Output** | `InsertResult(prefix_len, mamba_exist)` — `prefix_len` = length of reused prefix |
| **Mutation** | Creates new leaf nodes; updates component data on overlapping nodes; frees duplicate KV indices; may split nodes; updates LRU lists and evictable sizes |
| **Complexity** | **O(K + D·C)** |

**Algorithm detail** (`_insert_helper`):
1. At each existing node, calls `_touch_node` → promotes to MRU via `node_has_component_data()`
2. If key diverges mid-node, calls `_split_node` → `redistribute_on_node_split()` per component
3. For each overlapping node, calls `update_component_on_insert_overlap()` per component — returns `consumed_from` index; the tree frees `value[dup_start:consumed_from]` as duplicate pool indices
   - Full: returns `prefix_len` (no consumption, default behavior)
   - SWA: checks if the overlapping node is a tombstone (SWA value = None) within the SWA window boundary (`swa_evicted_seqlen`):
     - If entirely within window: **recovers tombstone** — frees old `full_value`, clones `value_slice`, translates to SWA indices, inserts into SWA LRU (returns `0` = all consumed)
     - If partially within window: **splits node** at boundary, recovers SWA on the window portion (returns `start_idx`)
     - If entirely outside window: returns `prefix_len` (no consumption)
   - Mamba: returns `prefix_len` (no consumption, default behavior)
4. Before creating a new leaf, checks `should_skip_leaf_creation()` per component — any veto aborts leaf creation and frees remaining value
5. Creates leaf via `_add_new_node` (clones value tensor, inserts into Full LRU)
6. Calls `commit_insert_component_data()` per component on the final target node (SWA may trigger a secondary split for window boundary; Mamba sets mamba pool indices and inserts into Mamba LRU)

---

### `evict(params: EvictParams) → EvictResult`

Free cached tokens to reclaim memory.

| Aspect | Detail |
|--------|--------|
| **Purpose** | Each component drives eviction from its own LRU list until its target is met |
| **Inputs** | `params.num_tokens` (full), `params.swa_num_tokens` (SWA), `params.mamba_num` (Mamba) |
| **Output** | `EvictResult(num_tokens_evicted, swa_num_tokens_evicted, mamba_num_evicted)` |
| **Mutation** | Frees pool indices; removes nodes from LRU lists; deletes leaf nodes from tree; cascades to lower-priority components; walks up parent chain to delete tombstone ancestors |
| **Complexity** | **O(E·H + L)** — E = nodes evicted, H = tombstone chain height, L = locked nodes skipped in LRU scan. |

**Algorithm detail:**
1. Calls `drive_eviction()` for each component:
   - Full: scans Full LRU from tail, only evicts **leaf** nodes (`get_leaf_lru_no_lock` — **O(L)**); calls `evict_component()` to free pool indices
   - SWA: scans SWA LRU from tail; **internal** nodes are tombstoned (evict SWA data, keep node), **leaf** nodes are fully deleted; both trigger cascade
   - Mamba: scans Mamba LRU from tail; **internal** nodes are tombstoned, **leaf** nodes are fully deleted; both trigger cascade
2. After each node eviction, calls `_cascade_evict`:
   - Queries `eviction_priority()` per component; evicts all with priority ≤ trigger's
   - Calls `evict_component()` + `node_has_component_data()` for cascaded components
   - For leaf: removes from parent, then `_iteratively_delete_tombstone_leaf` walks up **O(H)** ancestors

**Cascade eviction rules:**
- **Leaf nodes**: all priorities = 0 → evicting any cascades to all (node deleted)
- **Internal nodes**: Full(2) > SWA(1) > Mamba(0)
  - Evicting Mamba: no cascade
  - Evicting SWA: cascades to Mamba
  - Evicting Full: cascades to SWA + Mamba

---

### `inc_lock_ref(node: UnifiedTreeNode) → IncLockRefResult`

Lock a node to protect it (and its ancestors) from eviction.

| Aspect | Detail |
|--------|--------|
| **Purpose** | Called when a request begins using a cached prefix — prevents eviction of nodes it depends on |
| **Inputs** | `node` — the last matched node (deepest) |
| **Output** | `IncLockRefResult(swa_uuid_for_lock)` |
| **Mutation** | Increments `lock_ref` per component along the path; moves tokens from evictable to protected size counters |
| **Complexity** | **O(D)** — Full: node to root; SWA: up to window boundary O(min(D, W)); Mamba: O(1).|

**Algorithm detail:** Calls `acquire_component_lock()` for each component.

| Component | Strategy |
|-----------|----------|
| Full | **Path-lock**: walks from node to root, `lock_ref += 1` on every ancestor. On first lock (`lock_ref: 0→1`), moves tokens from `component_evictable_size_` to `component_protected_size_`. |
| SWA | **Window-lock**: walks upward, accumulating SWA value lengths until `sliding_window_size` is filled. Records a `component_uuid` at the boundary node for `dec_lock_ref` to know where to stop. |
| Mamba | **Single-node lock**: only `lock_ref += 1` on the node itself (mamba state is per-leaf, not per-path). |

---

### `dec_lock_ref(node, params?) → DecLockRefResult`

Unlock a previously locked node path.

| Aspect | Detail |
|--------|--------|
| **Purpose** | Called when a request finishes — releases eviction protection |
| **Inputs** | `node`, optional `params.swa_uuid_for_lock` for SWA boundary detection |
| **Output** | `DecLockRefResult()` |
| **Mutation** | Decrements `lock_ref` per component; moves tokens from protected back to evictable when `lock_ref` reaches 0 |
| **Complexity** | **O(D)** — symmetric to `inc_lock_ref` |

**Algorithm detail:** Calls `release_component_lock()` for each component. Full walks to root; SWA walks up until matching `component_uuid`; Mamba decrements single node.

---

### `cache_finished_req(req: Req, is_insert: bool = True)`

Cache a completed request's KV data into the tree.

| Aspect | Detail |
|--------|--------|
| **Purpose** | After a request finishes, insert its token/KV data into the tree for future reuse |
| **Inputs** | `req` — the finished request; `is_insert` — whether to insert (True) or just release locks (False) |
| **Output** | `None` |
| **Mutation** | Calls component hooks → `insert` → `dec_lock_ref` → component cleanup. Frees unaligned tail KV indices; frees non-inserted KV indices when `is_insert=False`. |
| **Complexity** | **O(K + D·C)** — insert O(K + D·C) + lock release O(D). Simplifies to **O(K)**. |

**Algorithm detail:**
1. `prepare_for_caching_req()` per component — sets component-specific insert params, returns effective cache length (SWA: sets `swa_evicted_seqlen`; Mamba: prepares `mamba_value` from ping-pong buffer, returns `mamba_last_track_seqlen` as truncation hint)
2. Truncates if `effective_cache_len < len(token_ids)`: frees excess pool indices
3. Converts token IDs (bigram if EAGLE), page-aligns keys, then calls `insert()`
4. Frees unaligned tail KV indices beyond page boundary
5. Calls `dec_lock_ref()` on the previous `req.last_node`
6. `cleanup_after_caching_req()` per component (Mamba: frees forked mamba_value based on `mamba_exist`, handles ping-pong buffer cleanup)

---

### `cache_unfinished_req(req: Req, chunked=False)`

Cache an in-progress request's partial KV data (chunked prefill).

| Aspect | Detail |
|--------|--------|
| **Purpose** | During chunked prefill, insert partial results so the next chunk can match the prefix |
| **Inputs** | `req` — the in-progress request |
| **Output** | `None` |
| **Mutation** | Inserts partial KV → re-matches prefix → updates `req.prefix_indices`, `req.cache_protected_len`, `req.last_node`; transfers lock from old node to new node |
| **Complexity** | **O(K + D·C)** — two tree traversals: insert O(K + D·C) + re-match O(K + D·C) + lock transfer O(D). Simplifies to **O(K)**. |

**Algorithm detail:**
1. `prepare_for_caching_req()` per component
2. `insert()` — first tree traversal
3. `match_prefix()` — **second** tree traversal to get updated indices
4. Writes new prefix indices into `req_to_token_pool`
5. `dec_lock_ref()` on old `req.last_node`
6. `inc_lock_ref()` on new matched node
7. Updates `req.prefix_indices`, `req.cache_protected_len`, `req.last_node`
8. `cleanup_after_caching_req()` per component

---

## TreeComponent Hook Reference

Each component implements these hooks. See `tree_component.py` for the ABC and docstrings.

### Match Phase

| Hook | Purpose | Called By | Default |
|------|---------|-----------|----------|
| `create_match_validator()` | Return a per-match stateful predicate that decides whether a node is a valid match boundary. Full: always True. SWA: tracks accumulated window length, True when contiguous window ≥ `sliding_window_size`. Mamba: True iff node has mamba data. | `_match_prefix_helper` | *abstract* |
| `finalize_match_result()` | Post-process the match result after prefix matching completes. Full/SWA: pass-through. Mamba: copy-on-write — allocates a new mamba pool slot, copies SSM state into the request pool, records `branching_seqlen`. | `_match_post_processor` | pass-through |

### Insert Phase

| Hook | Purpose | Called By | Default |
|------|---------|-----------|----------|
| `update_component_on_insert_overlap()` | Handle key overlap with an existing node during insert. Returns the index within `value_slice` from which this component consumed (took ownership of) pool slots. Full/Mamba: no consumption (`prefix_len`). SWA: may recover tombstoned nodes within the sliding window boundary. | `_insert_helper` | returns `prefix_len` |
| `should_skip_leaf_creation()` | Veto leaf creation when the entire new leaf would be a tombstone for this component. SWA: vetoes if `swa_evicted_seqlen ≥ total_prefix_len + key_len`. | `_insert_helper` | `False` |
| `commit_insert_component_data()` | Finalize component data on the target node after the insert walk completes. Full: no-op (handled by `_add_new_node`). SWA: checks window boundary, may split node — parent becomes tombstone, child gets SWA data. Mamba: sets mamba pool indices and inserts into Mamba LRU. | `_insert_helper` | no-op |

### Node Split

| Hook | Purpose | Called By | Default |
|------|---------|-----------|----------|
| `redistribute_on_node_split()` | Redistribute component data between new parent (prefix) and child (suffix) when a node is split. Full: copies `lock_ref` to parent. SWA: slices SWA value, copies `lock_ref` and `component_uuid`. Mamba: parent gets `None`/`lock_ref=0` (mamba stays on leaf). | `_split_node` | *abstract* |

### Eviction Phase

| Hook | Purpose | Called By | Default |
|------|---------|-----------|----------|
| `evict_component()` | Free this component's KV resources on a node being evicted. Internal nodes: free memory and tombstone (`value = None`). Leaf nodes: free memory, node will be deleted. Returns number of tokens freed. | `_evict_component_and_detach_lru` | *abstract* |
| `eviction_priority()` | Return cascade eviction priority (higher = evicted later). Leaf: all 0. Internal: Full(2) > SWA(1) > Mamba(0). When evicting, all components with ≤ priority on the same node are cascade-evicted. | `_cascade_evict` | `0` |
| `drive_eviction()` | Drive eviction from this component's LRU list until the target amount is freed. Full: leaf-only from Full LRU. SWA: both internal (tombstone) and leaf from SWA LRU. Mamba: both internal (tombstone) and leaf from Mamba LRU. | `evict` | *abstract* |

### Lock Phase

| Hook | Purpose | Called By | Default |
|------|---------|-----------|----------|
| `acquire_component_lock()` | Increment `lock_ref` to protect nodes from eviction; moves tokens from evictable to protected. Full: path-lock to root. SWA: window-lock with UUID boundary. Mamba: single-node lock. | `inc_lock_ref` | *abstract* |
| `release_component_lock()` | Decrement `lock_ref` to un-protect nodes; moves tokens from protected to evictable when `lock_ref` → 0. Full: path-unlock to root. SWA: walks up to UUID boundary. Mamba: single-node unlock. | `dec_lock_ref` | *abstract* |

### Caching Phase

| Hook | Purpose | Called By | Default |
|------|---------|-----------|----------|
| `prepare_for_caching_req()` | Prepare component-specific data before insert, fill fields in `InsertParams`, return effective cache length. Full: no-op. SWA: sets `swa_evicted_seqlen`. Mamba: prepares `mamba_value` from ping-pong buffer, returns `mamba_last_track_seqlen`. | `cache_finished/unfinished_req` | returns `None` |
| `cleanup_after_caching_req()` | Post-cache cleanup. Full/SWA: no-op. Mamba: frees forked `mamba_value` based on `mamba_exist`, handles ping-pong buffer `keep_idx`, resets `mamba_last_track_seqlen` on unfinished. | `cache_finished/unfinished_req` | no-op |

### Utility

| Hook | Purpose | Called By | Default |
|------|---------|-----------|----------|
| `node_has_component_data()` | Check if a node has this component's data. Used as filter for LRU operations and cascade checks. Full overrides to check `full_value` directly. | multiple | `value is not None` |

### Component Behavior Summary

| Behavior | FullComponent | SWAComponent | MambaComponent |
|----------|--------------|-------------|----------------|
| **Validator** | Always `True` | Tracks accumulated window; `True` when ≥ `sliding_window_size` | `True` iff node has mamba data |
| **Lock strategy** | Path-lock (root → node) | Window-lock (up to window boundary, UUID-tagged) | Single-node lock |
| **Internal eviction priority** | 2 (last) | 1 (middle) | 0 (first) |
| **Split behavior** | Copy `lock_ref` to parent | Slice SWA value + copy UUID | Parent gets `None` (mamba stays on leaf) |
| **Match finalize** | No-op | No-op | Copy-on-write: allocate new mamba slot, copy state |
| **Drive eviction** | Full LRU (leaf-only) → cascade all | SWA LRU → tombstone internal, cascade leaf | Mamba LRU → tombstone internal, cascade leaf |

---

## Factory Function

```python
def create_unified_radix_cache(
    params: CacheInitParams,
    component_names: Optional[tuple[ComponentName, ...]] = None,
) -> UnifiedRadixCache
```

Auto-detects component configuration from `params` if `component_names` is not specified:
- `SWATokenToKVPoolAllocator` → `(SWA,)` → `UnifiedSWARadixCache`
- `HybridReqToTokenPool` → `(MAMBA,)` → `UnifiedMambaRadixCache`
- Explicit tuple → `UnifiedRadixCache` with specified components

Enable via `--enable-unified-radix-tree` server flag.
