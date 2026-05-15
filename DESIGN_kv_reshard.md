# Design — KV Cache Resharding Across CP Ranks

Branch: `kv_reshard` (extends to `cp_multi_batch` for bs > 1). Status: design v4, no code yet.

## 1. Problem

When CP (context parallelism) is enabled, every CP rank stores the **full** KV cache today. After attention computes K, V on the rank's local token slice, `cp_allgather_and_save_kv_cache` (`python/sglang/srt/layers/utils/cp_utils.py:354`) allgathers full K, V across the CP group and writes the **entire** sequence into every rank's pool.

This is simple for SPMD scheduling and RadixCache — all ranks see identical KV state — but wastes HBM by **CP_size ×**, capping how many concurrent requests a prefill node can admit.

**Goal**: shard the physical KV bytes across CP ranks (each rank stores only its slice) while keeping a **global logical view** of the KV cache — a "global KV cache tree" — so RadixCache can still detect and reuse cached prefixes. The cluster's effective unique cache grows to **`cp_size × size_local`** at the same per-rank HBM footprint.

**Realization of the "global tree":** the radix tree is **mirrored** across all CP-rank schedulers. Every node carries an explicit per-page ownership array `cp_owner_per_page: torch.Tensor[int8]` of length `len(key) // page_size` — the same array on every rank, validated by an allgather on every insert. Only `node.value` (the physical row indices) differs per rank: real local pool rows where this rank owns a page, slot-0 sentinels everywhere else.

**Realization of attention with sharded pool:** paged FA fetches K, V by **slot index**, so every position FA reads must be indexable through `k_buffer` / `v_buffer`. v4 pre-allocates **transient pool rows** for every non-owned position at the start of a forward call, fills them from a transient staging buffer that the per-layer NCCL allgather writes into, and frees them at the end of the call. FA reads through the unchanged `req_to_token` page table — there is no separate scratch region and no mixed-source page table.

## 2. Scope of v1

- **Disaggregated prefill mode only** (`disaggregation_mode == "prefill"`). Decode handoff is already CP-aware via `filter_kv_indices_for_cp_rank` (`python/sglang/srt/disaggregation/utils.py:488`).
- **Static CP only.** Dynamic CP is not on the roadmap; no slot-affinity or group-isolation logic.
- **bs ≥ 1 CP-prefill** (`cp_multi_batch` compatible). Single-request and multi-request batches both supported; per-request ownership is locked at admission so chunked prefill stays stable.
- All three pool types: **MHA**, **MLA**, **NSA**.
- **Synchronous prefix gather** in v1 — gather is blocking inside each layer's attention. Correctness first; async overlap is a later optimization (§9).
- Gated by `--enable-cp-kv-reshard`, default off.
- **Banned in v1** (asserted at startup): hicache (host-side store is unsharded), hisparse (separate retract path), SWA hybrid pool with `cp_size > 1`, decode-mode disagg, cross-attention layers, and spec-v2 / draft-extend (those keep the legacy un-resharded path; the new write path is gated on `forward_mode == EXTEND`).

## 3. Sharding scheme

**Token-dim, contiguous global-page ranges.** For a request with global pages `[0, T)`, CP rank `r` owns a contiguous slice using the rule already implemented in `page_indices_to_cp_rank_page_indices` (`disaggregation/utils.py:442`). With `rem = total_pages % cp_size`, the first `rem` ranks each own `base + 1` pages and the rest own `base`. Per-request imbalance is at most one page; aggregate skew across many requests is bounded by `cp_size − 1` pages per rank — well within tolerance for v1. Reusing the existing partition means prefill-write ownership matches disaggregation transfer ownership without a second mapping.

> **Note**: an earlier draft of this design added a per-request `rotation_offset = req.bootstrap_room % cp_size` to spread the trailing-page extras across ranks. Profiling did not justify the added plumbing (extra field on `Req`, an extra parameter on every ownership helper, rotation-specific tests, and an extra dependency on `bootstrap_room`) for the small skew it would remove. Rotation is strictly additive and can be reintroduced as a focused follow-up if profiling later shows the skew matters.

The forward direction (page index → owner) is centralized in a new helper:

```python
def owner_for_page(global_page_idx: int, total_pages: int, cp_size: int) -> int:
    """The CP rank that owns a given page. Inverse of the existing
    page_indices_to_cp_rank_page_indices. Identical on every CP rank
    — used both to compute cp_owner_per_page and to validate it via
    allgather on insert."""
    base = total_pages // cp_size
    rem = total_pages % cp_size
    if rem == 0:
        return global_page_idx // base
    boundary = rem * (base + 1)
    if global_page_idx < boundary:
        return global_page_idx // (base + 1)
    return rem + (global_page_idx - boundary) // base
```

For **bs > 1**, per-page ownership is locked at admission as `req.cp_owner_per_page: torch.Tensor[int8]` and is stable across chunked-prefill chunks — the same logical page never changes owner mid-request. Cross-request skew within the same batch is bounded by `cp_size − 1` rows per rank; symmetric NCCL allgathers handle this by padding each rank's contribution to `max_owned` (the existing pattern in `cp_all_gather_reorganized_into_tensor_kv_cache` at `cp_utils.py:204-243`).

### How the logical-vs-physical split is implemented

The radix cache and allocator continue to operate over **logical page IDs** that conceptually span `cp_size × size_local` slots. The physical pool tensor on each rank has size `size_local + transient_reserve` rows per layer, where `transient_reserve = max_prefill_tokens × (cp_size − 1) / cp_size`. The mapping logical-position → physical-row is carried by the **per-rank page table** (`req_to_token`):

- For positions rank `r` owns: a valid persistent row from `r`'s pool (`size_local` capacity).
- For positions rank `r` does not own: a **transient row** allocated this forward from the same pool's `transient_reserve` capacity. After the forward returns, transient rows are freed back to the allocator and `req_to_token` at those positions is rewritten to slot-0 sentinel.

This is the natural extension of paged attention's existing logical/physical split — the page table is already per-request, just per-rank-projected. No new indirection structure is needed. `node.value` and `req.prefix_indices` automatically become per-rank because they are lifted from `req_to_token`.

### How "global consistent view" works

Every CP-rank scheduler holds an instance of the radix tree whose **structure is bit-identical** to peers' (keys, parent/child, lock_ref, last_access_time, cp_owner_per_page). Only `node.value` differs per rank — owned positions hold real pool rows, others hold slot-0 sentinels.

Consistency is preserved by:

- **SPMD inputs** — `recv_requests` does `broadcast_pyobj`, so every rank sees the same requests in the same order.
- **Deterministic logical clock** — `last_access_time` reads a per-tree `current_logical_clock` that ticks once per `recv_requests` iteration (§5 Fix 1), not from `time.monotonic()` which drifts independently.
- **Deterministic eviction ordering** — `TreeNode.__lt__` uses `(last_access_time, _first_token_for_order(), _extra_key_for_order())` (§5 Fix 2) so even when timestamps tie, every rank's `heapq` pops the same node. `_extra_key_for_order` returns the raw `extra_key` string, *not* `hash(extra_key)`, because Python's built-in string `hash()` is `PYTHONHASHSEED`-randomized per process.
- **Sentinel-safe `value` operations** — `evict()`, `cache_finished_req`, `cache_unfinished_req` all filter `value != 0` before `allocator.free()` (§5 Fix 3) so the per-rank `value` projection never corrupts the allocator. Only the owning rank's call to `allocator.free` releases real pool rows.
- **Explicit allgather validation** — on every successful `tree.insert(...)` for a new request, the rank-computed `req.cp_owner_per_page` is allgathered across `attn_cp_group` and asserted identical. This is the load-bearing v4 invariant: every rank stores the same `cp_owner_per_page` on every TreeNode, so per-rank "is this local?" decisions are guaranteed to agree on which physical rank holds each page's bytes.

### Why not layer-dim

Each rank computes attention on **every** layer during forward. If it doesn't own a layer's K, V it must allgather from the owner before attention and flush its new writes back afterward — net per-step traffic equals today's, but with **none** of the bandwidth savings. The imbalance worry is real but secondary: layer-dim additionally breaks the per-layer ownership baked into `LayerDoneCounter` (`managers/cache_controller.py:51-100`) and `register_layer_transfer_counter` (`mem_cache/memory_pool.py:772`).

### Why not head-dim

Heads are already split by TP. Re-splitting them across CP breaks `MHATokenToKVPool`'s `(size, head_num, head_dim)` layout (`memory_pool.py:902`) and the FA page-table indexing, and forces a custom head-rerange kernel.

### Why token-dim wins

It is the dimension CP already operates on (zigzag block split). After `cp_all_gather_rerange_kv_cache` (`cp_utils.py:286-321`) restores sequence order, the rank-owned subset is a **contiguous row slice** of `cache_loc` — no extra kernel. The owned slice maps directly to consecutive local rows allocated from this rank's pool.

## 4. Per-page ownership in the radix tree

### TreeNode field

Add a single field to `TreeNode` (`mem_cache/radix_cache.py:211-272`):

```python
self.cp_owner_per_page: Optional[torch.Tensor] = None  # int8, length = len(key) // page_size
```

- **Mirrored across all CP ranks**: every rank's TreeNode holds the same `cp_owner_per_page` array per node. Mirrored by SPMD determinism, validated by allgather on insert.
- Each entry is the integer CP rank (`0..cp_size-1`) that physically holds that page's KV bytes.
- The per-rank "is local" view is derived on demand: `is_local_per_page = (cp_owner_per_page == self_cp_rank)`. Not stored — boolean masks are cheap to recompute and storing both invites desync.
- Length **always** equals `len(key) // page_size`. Maintained as an invariant by `_split_node` (which slices both `value` and `cp_owner_per_page` at the page-aligned split) and `_insert_helper` (which carries the per-page array through node creation).

### Why per-page (not per-node)

A single multi-page TreeNode can have its pages owned by different CP ranks. With `page_size = 16` and `cp_size = 4`, a 128-token (8-page) node naturally has pages distributed across all four ranks. Forcing one-node-per-page would require restructuring `_insert_helper` to split at every page boundary, blowing up tree size by ~`seq_len / page_size`. v4 keeps the existing variable-length-key compression and uses an in-node per-page list instead.

### Ownership is fixed by the creator

`cp_owner_per_page` is set at the request that **inserts** the node, using `owner_for_page`. Subsequent requests that match this prefix do **not** rewrite it — the physical bytes already live on whichever rank the creator assigned. A new request derives its own "local-or-remote" mask from the stored array.

### `_split_node` slicing

`_split_node` (`radix_cache.py:721-741`) splits a node at a page-aligned `split_len` (since `key.match` operates with `page_size` granularity). v4 extends it to also slice `cp_owner_per_page`:

```python
def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
    # ... existing logic
    new_node.value = child.value[:split_len].clone()
    child.value = child.value[split_len:].clone()
    # NEW: slice cp_owner_per_page at the page-aligned boundary
    assert split_len % self.page_size == 0
    page_split = split_len // self.page_size
    if child.cp_owner_per_page is not None:
        new_node.cp_owner_per_page = child.cp_owner_per_page[:page_split].clone()
        child.cp_owner_per_page = child.cp_owner_per_page[page_split:].clone()
    # ...
    return new_node
```

### `_insert_helper` plumbing

`_insert_helper` (`radix_cache.py:751-803`) accepts a `cp_owner_per_page: Optional[torch.Tensor]` parameter (passed through `InsertParams`). When a new TreeNode is created for the leftover key (line 791-797), the corresponding slice of `cp_owner_per_page` is stored on it.

## 5. Eviction consistency (correctness)

This is the load-bearing piece. Eviction policy is SPMD-deterministic — no allgather on the hot path — so the tie-breaker chain must produce bit-identical decisions across ranks.

### Risk 1 — Clock skew across schedulers

`TreeNode.last_access_time` is `time.monotonic()` (`radix_cache.py:221, 222, 696, 704, 762`). Every eviction strategy in `evict_policy.py` reads it. Each scheduler process has an independent monotonic clock; with sharded KV that is a correctness bug.

### Fix 1 — Logical clock

Replace `time.monotonic()` calls with a **monotonically-increasing scheduler logical clock** that ticks exactly **once per `recv_requests` iteration**, post-`broadcast_pyobj`:

```python
# scheduler.py, after broadcast_pyobj synchronizes all CP ranks:
self.radix_logical_clock += 1
tree_cache.set_logical_clock(self.radix_logical_clock)

# radix_cache.py:
self.last_access_time = self._tree.current_logical_clock
```

`match_prefix` and `_match_prefix_helper` only **read** the clock; advancement happens at the top of the scheduler loop. This prevents within-iteration drift.

### Risk 2 — Heap-tiebreaker `id()` divergence

`TreeNode.__lt__` (`radix_cache.py:270-271`) compares only `last_access_time`. Under a discrete logical clock, many nodes share a timestamp on any iteration; Python's heap falls back to `__eq__` then `id()` — process-local and divergent across CP ranks.

### Fix 2 — Deterministic heap tiebreaker

```python
def __lt__(self, other: "TreeNode"):
    return (self.last_access_time,
            self._first_token_for_order(),
            self._extra_key_for_order()) < (
           other.last_access_time,
            other._first_token_for_order(),
            other._extra_key_for_order())

def _first_token_for_order(self) -> int:
    if self.key is None or len(self.key.token_ids) == 0:
        return -1
    return int(self.key.token_ids[0])

def _extra_key_for_order(self) -> str:
    # Raw string, NOT hash() — Python's hash() of strings is
    # PYTHONHASHSEED-randomized per process and would diverge
    # between CP rank processes, defeating the SPMD invariant.
    if self.key is None or self.key.extra_key is None:
        return ""
    return self.key.extra_key
```

The radix-tree split rule guarantees that two sibling children of the same parent diverge at their first token — `key[0]` is unique among siblings and identical across CP ranks. `extra_key` disambiguates same-token siblings with different namespaces (LoRA ids, etc.).

### Fix 3 — Sentinel filter at every `allocator.free` site

`TokenToKVPoolAllocator.clear()` (`mem_cache/allocator.py:124-167`) reserves slot 0 as the padding row. Under sharding, `node.value` legitimately contains `0` at sentinel positions. Every site that does `allocator.free(node.value)` must filter first:

```python
to_free = node.value[node.value != 0]
self.token_to_kv_pool_allocator.free(to_free)
```

Sites: `evict()` (line ~626), `cache_finished_req` (lines ~501, 522, 526, 531), `cache_unfinished_req` (line ~563).

A stronger per-rank guard is "only the owning rank frees pool rows for a given page":

```python
# In evict():
owned_mask = (node.cp_owner_per_page == self.cp_rank).repeat_interleave(self.page_size)
to_free = node.value[owned_mask & (node.value != 0)]
```

Both checks are equivalent under the invariant `value[i] == 0 iff page-of-i is not owned by self`. v4 uses Fix 3 as the primary filter and the `cp_owner_per_page` check as a debug assertion.

### Eviction sync

No allgather is needed on the eviction hot path. Given identical request streams, identical logical clocks, and identical `__lt__`, every rank pops the same node from the eviction heap. A debug-mode CRC32 allgather of `(num_evicted, hash(evicted_keys_sorted))` catches any divergence loudly.

### `evictable_size_` stays logical

`_insert_helper` does `self.evictable_size_ += len(key)`. We keep `evictable_size_` as the **logical** count for two reasons:
1. The tree is structurally mirrored; logical accounting is the global view.
2. Per-rank physical pool availability is tracked separately via `MirroredCpAvailability` (§7).

## 6. Forward path — transient pool pages

### Pool sizing

Extend `k_buffer` / `v_buffer` (`memory_pool.py:_create_buffers`) by `transient_reserve = max_prefill_tokens × (cp_size − 1) / cp_size` rows per layer. **These rows are visible to the regular `TokenToKVPoolAllocator`** — there is no separate scratch region and no hidden bookkeeping.

```
k_buffer shape per layer: [size_local + 1 + transient_reserve, head_num, head_dim]
                           └── slot 0: padding (today's sentinel)
                           └── slots [1, size_local + transient_reserve]:
                                 managed entirely by TokenToKVPoolAllocator
```

`MirroredCpAvailability.local_available[r]` accounts only the `size_local` portion; `transient_reserve` is reserved capacity invisible to admission.

### Per-forward allocation lifecycle

**Pre-forward** (in `FlashAttentionBackend.init_forward_metadata`, `flashattention_backend.py:254`):

1. For each request `s`, compute the per-position owner mask from `req.cp_owner_per_page` (extended to also cover the new-token positions of this step).
2. Sum the non-owned position counts across all requests in the batch → `total_transient`.
3. Single batched `allocator.alloc(total_transient)` → `forward_batch.cp_transient_rows`.
4. Carve into per-request slices; write each request's transient rows into `req_to_token[req_pool_idx_s, non_owned_positions]`. After this, every position in `req_to_token` for every request points to a real pool row — owned positions to permanent rows, non-owned to transient rows.

**Per layer in `forward_extend`** (`flashattention_backend.py:597`):

1. **Cache-hit remote prefix** — for each request with non-owned prefix positions, call `cp_fill_remote_prefix_pool_rows`:
   - Read the owning rank's local prefix rows from this layer's `k_buffer` / `v_buffer` (via the `is_local` mask).
   - Pad-to-max and allgather across `attn_cp_group` (reuse `cp_all_gather_reorganized_into_tensor_kv_cache` at `cp_utils.py:204-243`).
   - Unpad to canonical order — yields full `[M_s, h, d]` K and V for the prefix.
   - Scatter the non-owned positions of the canonical tensor into the **transient pool rows** pre-allocated in step (1) above. After this scatter, the layer's `k_buffer` / `v_buffer` contains complete prefix KV for every position FA will read.
2. **Current-step KV** — rewritten `cp_allgather_and_save_kv_cache` (`cp_utils.py:354`):
   - `cp_all_gather_rerange_kv_cache` (`cp_utils.py:286`) produces canonical-order full K, V for the current step's new tokens.
   - Single `token_to_kv_pool.set_kv_buffer(layer, out_cache_loc, key_full, value_full, ...)` where `out_cache_loc` covers **all positions** (both permanent owned rows and transient non-owned rows). No two-destination split; no separate scratch write.
3. **Attention** reads `k_buffer` / `v_buffer` through the standard `metadata.page_table` (which is `req_to_token`). **No mixed page table, no scratch indexing, no FA kernel change.**

**Post-forward epilogue** (in `python/sglang/srt/model_executor/model_runner.py`, between the final layer and `cache_finished_req`):

1. Batched `allocator.free(forward_batch.cp_transient_rows)`.
2. Rewrite `req_to_token[req_pool_idx_s, non_owned_positions] = 0` (sentinel) so subsequent `cache_finished_req` / `cache_unfinished_req` correctly skips non-owned positions when extracting `kv_indices`.

### Staging buffer

The allgather producing the canonical-order full K, V is a transient tensor — call it the staging buffer. It is **not** a scratch region in the pool; it is a normal device tensor allocated per layer (or reused via a pre-allocated tensor on the attention backend, sized `max_prefill_tokens × head_num × head_dim × dtype_size × 2`). It is overwritten layer-by-layer because layers execute serially. After step (1) and step (2) scatter from staging into `k_buffer`/`v_buffer`, the staging tensor's contents for this layer are no longer needed.

For MHA: K and V are two separate staging tensors. For MLA: a single fused staging tensor. For NSA: also include FP8 per-block scales.

### Per-pool variations

- **MHA**: described above. K and V are two `(rows, head_num, head_dim)` buffers; both get the owned + transient writes.
- **MLA**: K and V share one fused `(rows, 1, kv_lora_rank + qk_rope_head_dim)`. Single `set_mla_kv_buffer` call instead of two.
- **NSA**: also gather per-block FP8 scales. Add a third staging tensor for scales; their transient rows live in the scale pool.

### Why this is simpler than v3

| Aspect | v3 (scratch) | v4 (transient pool) |
|---|---|---|
| Non-owned KV destination | High-address scratch rows in `k_buffer` (invisible to allocator) | Transient pool rows (allocator-managed) |
| FA page table | `cp_per_call_page_table` mixing pool + scratch rows | Unchanged `req_to_token` |
| Per-forward host cost | Build the mixed page table (amortized over layers) | Single batched `alloc` + sentinel-rewrite epilogue |
| New entry points on pool | `set_kv_buffer_scratch` (write at high-address rows without quant/allocator) | None — reuse existing `set_kv_buffer` |
| Allocator churn | None | Per-forward batched alloc/free (cheap; monitor) |
| Memory budget | `size_local + cp_scratch_rows` per layer | `size_local + transient_reserve` per layer — same total |

## 7. Admission and scheduler integration

### `MirroredCpAvailability`

```python
class MirroredCpAvailability:
    def __init__(self, cp_size, size_local):
        self.local_available = [size_local] * cp_size  # mirrored SPMD state

    def can_admit(self, owned_counts):           # list[int] of length cp_size
        return all(self.local_available[r] >= owned_counts[r]
                   for r in range(len(owned_counts)))

    def alloc(self, owned_counts):
        for r in range(len(owned_counts)):
            self.local_available[r] -= owned_counts[r]

    def free(self, owned_counts):
        for r in range(len(owned_counts)):
            self.local_available[r] += owned_counts[r]
```

Every CP rank instantiates this with the same args and runs identical operations on identical inputs (SPMD), so `local_available` stays mirrored without any all-reduce. `evict_from_tree_cache` is called with `num_tokens = max_r(owned_count_r)` so eviction frees enough on every rank's view; the tree's `evict()` then picks the same victims deterministically (Fixes 1+2).

### Hooks

- `check_decode_mem` (`schedule_batch.py:2143`): compute per-rank owned counts using `owner_for_page`; check via `MirroredCpAvailability.can_admit`.
- `retract_decode`: same sort key (already deterministic); release calls `MirroredCpAvailability.free`.
- `cache_finished_req` / `cache_unfinished_req`: free permanent pool rows for owned positions; call `MirroredCpAvailability.free`.

### Insert-time allgather validation

Immediately after every successful `tree.insert(...)` call in `cache_finished_req` / `cache_unfinished_req`:

```python
local = req.cp_owner_per_page                          # [num_pages], int8
gathered = torch.empty((cp_size, len(local)), dtype=torch.int8, device='cuda')
attn_cp_group.all_gather_into_tensor(gathered, local.unsqueeze(0))
assert (gathered == gathered[0:1]).all(), "cp_owner divergence across CP ranks"
```

Bytes per insert ≈ `num_pages × cp_size × 1`. For a 64K-token request with `page_size = 16` and `cp_size = 8`: 4096 × 8 = 32 KiB; one NCCL allgather over NVLink ≈ 10-20 μs. Negligible vs the model forward.

In production, the assertion can be downgraded to a CRC32 check (allgather one int per rank). `SGLANG_DEBUG_KV_RESHARD=1` keeps the full-array check.

## 8. Code changes

### Files to modify

| File | Change |
|---|---|
| `python/sglang/srt/mem_cache/radix_cache.py` | (1) Add `TreeNode.cp_owner_per_page` field. (2) Replace `time.monotonic()` (lines 221, 222, 696, 704, 762) with `self._tree.current_logical_clock`. (3) Rewrite `__lt__` with `(last_access_time, _first_token_for_order(), _extra_key_for_order())` tuple — `_extra_key_for_order` returns the raw `extra_key` string, not `hash(extra_key)`, because Python's built-in `hash()` of strings is `PYTHONHASHSEED`-randomized per process and would diverge across CP ranks. (4) Update `_split_node` to also slice `cp_owner_per_page` at the page-aligned split. (5) Plumb `cp_owner_per_page` through `InsertParams` / `_insert_helper`. (6) Filter sentinel entries in `evict()`, `cache_finished_req`, `cache_unfinished_req` before `allocator.free()`. (7) Add `set_logical_clock` setter; expose `current_logical_clock`. |
| `python/sglang/srt/managers/scheduler.py` | Advance `radix_logical_clock` after `broadcast_pyobj` in `recv_requests`. Instantiate `MirroredCpAvailability`. Add allgather-validate hook around `tree.insert` calls. Add `SGLANG_DEBUG_KV_RESHARD`-gated state-hash all-gather divergence fuse. |
| `python/sglang/srt/managers/schedule_batch.py` | Add `req.cp_owner_per_page` at admission (computed once via `compute_cp_owner_per_page`). `check_decode_mem` computes per-rank owned counts; uses `MirroredCpAvailability.can_admit`. |
| `python/sglang/srt/layers/utils/cp_utils.py` | Add `owner_for_page` and `compute_cp_owner_per_page`. Rewrite `cp_allgather_and_save_kv_cache` (line 354) to write owned + transient rows through standard `set_kv_buffer`. Add `cp_fill_remote_prefix_pool_rows` for cache-hit remote prefix gather. |
| `python/sglang/srt/layers/attention/flashattention_backend.py` | (1) In `init_forward_metadata` (line 254), pre-allocate transient rows via a single batched `alloc()`; write them into `req_to_token` at non-owned positions. (2) In `forward_extend` (line 597), invoke `cp_fill_remote_prefix_pool_rows` before the rewritten `cp_allgather_and_save_kv_cache`. (3) Allocate a shared staging buffer at backend init for the prefix gather. (4) Use unchanged `metadata.page_table` (= `req_to_token`). |
| `python/sglang/srt/hardware_backend/musa/attention/flashattention_backend.py` | Mirror FA changes. |
| `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` | Apply owned-row slice; defer remote prefix gather + transient alloc to v2 on NPU. v1 NPU stays on the legacy un-resharded path. |
| `python/sglang/srt/mem_cache/memory_pool.py` | Under `--enable-cp-kv-reshard`, extend `k_buffer` / `v_buffer` by `transient_reserve` rows (exposed to allocator; no special bookkeeping). MLA + NSA variants too. |
| `python/sglang/srt/mem_cache/allocator.py` | No code change; confirm batched alloc/free is cheap; `need_sort=False` recommended for the prefill pool. |
| `python/sglang/srt/model_executor/model_runner.py` | Post-forward epilogue: batched `allocator.free(forward_batch.cp_transient_rows)` and sentinel-rewrite of `req_to_token` at non-owned positions. |
| `python/sglang/srt/server_args.py` | Add `--enable-cp-kv-reshard`; v1 guardrails (no hicache, no hisparse, no SWA + cp_size > 1, disagg-prefill only, no cross-attention models, no spec-v2). |

### Existing utilities to reuse

- `page_indices_to_cp_rank_page_indices` (`disaggregation/utils.py:442`) — reused as-is for prefill-write ownership.
- `attn_cp_all_gather_into_tensor` (`layers/dp_attention.py:599`) — sub-group allgather.
- `cp_all_gather_rerange_kv_cache` (`cp_utils.py:286-321`) — zigzag rerange.
- `cp_all_gather_reorganized_into_tensor_kv_cache` (`cp_utils.py:204-243`) — pad-to-max symmetric allgather pattern; reuse for remote prefix gather.
- `filter_kv_indices_for_cp_rank` (`disaggregation/utils.py:488`) — disagg transfer filtering.
- `poll_and_all_reduce_attn_cp_tp_group` (`disaggregation/utils.py:71-88`) — CP-consensus pattern (reuse the gloo CPU group for the debug fuse).
- `set_kv_buffer` (`memory_pool.py:1037`) and MLA/NSA variants — unchanged write path.
- Paged attention's existing `req_to_token` page-table infrastructure — the natural place for per-rank indirection.

### What is explicitly **not** in this design

- ❌ Scratch region inside `k_buffer` / `v_buffer`. v4 uses transient pool rows instead.
- ❌ Mixed-source page table (`cp_per_call_page_table`). FA reads through unchanged `req_to_token`.
- ❌ Consensus all-reduce around `check_decode_mem` → `retract_decode`. The decision stays SPMD-deterministic via `MirroredCpAvailability`.
- ❌ Consensus all-reduce around eviction victim selection. Deterministic tiebreaker (Fixes 1+2) handles it.
- ❌ A separate `global_id → local_row` indirection structure. The page table (`req_to_token`) carries it.
- ❌ Index remap that shrinks the per-rank pool tensor below `size_local`. We keep the same physical pool size and let the page table do the virtualization.

## 9. Suggested rollout — part-by-part

**Step 0** — Update this design doc (you are reading the v4 version).

**Part 1 — Radix Cache Tree changes** (start here). Land all of:
- 1a. Replace `time.monotonic()` with logical clock.
- 1b. Rewrite `TreeNode.__lt__` with 3-tuple tiebreaker.
- 1c. Filter sentinel entries at every `allocator.free` site in `radix_cache.py`.
- 1d. Add `TreeNode.cp_owner_per_page` field.
- 1e. Update `_split_node` to slice `cp_owner_per_page`.
- 1f. Plumb `cp_owner_per_page` through `InsertParams` / `_insert_helper`.
- 1g. Add `req.cp_owner_per_page` on the `Req` class.
- 1h. Filter `cache_finished_req` / `cache_unfinished_req` by owner mask.

One PR, six commits. Value-positive even without CP (1a-1c are bug-prevention fixes).

**Part 2 — Server-args, ownership helper, MirroredCpAvailability**:
- 2a. `--enable-cp-kv-reshard` flag + v1 guardrails.
- 2b. `owner_for_page` + `compute_cp_owner_per_page`.
- 2c. Rotation-aware `page_indices_to_cp_rank_page_indices` and `filter_kv_indices_for_cp_rank`.
- 2d. `MirroredCpAvailability` wired to `check_decode_mem`.
- 2e. Allgather-validate `cp_owner_per_page` at every insert.

**Part 3 — Pool sizing + transient row alloc/free**:
- 3a. Extend pools by `transient_reserve` rows (MHA, MLA, NSA).
- 3b. Confirm allocator covers the new rows.
- 3c. Per-forward batched `alloc` / `free` hooks in `init_forward_metadata` and the model-runner epilogue (verify lifecycle without yet wiring attention).

**Part 4 — Forward path rewrite** (the load-bearing part):
- 4a. Rewrite `cp_allgather_and_save_kv_cache`.
- 4b. Add `cp_fill_remote_prefix_pool_rows`.
- 4c. Build `req_to_token` mapping with transient rows in `init_forward_metadata`.
- 4d. Invoke prefix-fill + rewritten allgather in `forward_extend`.
- 4e. Post-forward epilogue (free transient + sentinel rewrite).

Delivers `cp_size×` admission for both fresh prefill and cache hit.

**Part 5 — Device + model parity**: MLA, NSA, musa, NPU (write-only path on NPU).

**Part 6 — Disagg integration**: PD-disagg regression to confirm decode outputs match baseline after sharded prefill.

**Part 7 — Future (out of scope for v1)**: async overlap of prefix gather with prior-layer MLP (mirror `HiCacheController.load_stream` pattern at `cache_controller.py:330`).

## 10. Verification

Per-part test matrix. **All Part 1 tests are unit-only — no GPU, no NCCL — run in `stage-a-test-cpu`.**

| # | Test | Validates |
|---|---|---|
| 1.1 | `test/srt/test_v4_logical_clock.py` — two simulated schedulers; assert `heapq` pops the same node under tied `last_access_time`. | 1a + 1b |
| 1.2 | `test/srt/test_v4_sentinel_filter.py` — synthesize `node.value` mixing real rows and slot-0; assert `evict`, `cache_finished_req`, `cache_unfinished_req` only free non-zero entries. | 1c |
| 1.3 | `test/srt/test_v4_split_node_cp_owner.py` — insert long key with known `cp_owner_per_page`; trigger `_split_node`; assert parent/child carry correctly sliced arrays; assert `len(value) == page_size * len(cp_owner_per_page)`. | 1d + 1e |
| 1.4 | `test/srt/test_v4_insert_cp_owner.py` — drive `insert` with synthetic `cp_owner_per_page`; assert every new TreeNode carries the expected slice; `match_prefix` returns matching slices. | 1f |
| 1.5 | `test/srt/test_v4_req_cp_owner_admission.py` — assert `compute_cp_owner_per_page` is deterministic across calls and that the owner array attached to `Req` at admission is reused across chunked-prefill chunks (no recomputation). | 1g |
| 1.6 | `test/srt/test_v4_cache_req_filter.py` — synthesize a request mixing local and remote pages; call `cache_finished_req`; assert allocator only frees locally-owned pool rows and the inserted node correctly stores sentinels elsewhere. | 1h |
| 2.1 | `test/srt/test_v4_owner_for_page.py` — `owner_for_page` matches `page_indices_to_cp_rank_page_indices`; partition imbalance is at most one page per request; every rank owns ≥1 page when `total_pages ≥ cp_size`. | 2b |
| 2.2 | `test/srt/test_v4_disagg_compat.py` — round-trip `page_indices_to_cp_rank_page_indices` ↔ `filter_kv_indices_for_cp_rank` over all `rem` combinations. | 2c |
| 2.3 | `test/srt/test_v4_mirrored_availability.py` — admit/free on 4 ranks; assert `local_available[r]` stays mirrored. | 2d |
| 2.4 | `test/srt/test_v4_insert_allgather.py` — multi-rank scheduler simulation; insert from each rank; assert allgather passes consistent, fails inconsistent. | 2e |
| 3.1 | `test/srt/test_v4_pool_sizing.py` — instantiate MHA/MLA/NSA pools with `cp_size ∈ {2, 4, 8}`; assert tensor shapes; assert `transient_reserve` excluded from admission. | 3a |
| 3.2 | `test/srt/test_v4_transient_alloc_lifecycle.py` — drive `init_forward_metadata`; assert one `alloc` + one `free` per forward; rows return cleanly across 1000 iters. | 3b + 3c |
| 4.1 | E2E `test/test_cp_single_layer.py` — bf16 logits parity, `cp_size ∈ {2, 4}`, bs ∈ {1, 2}, **no prefix hit** (fresh prefill — exercises non-owned current-step KV through transient pool rows). | 4a + 4c + 4d |
| 4.2 | E2E `test/test_cp_single_layer.py` — same with prefix hit; exercises `cp_fill_remote_prefix_pool_rows`. | 4b |
| 4.3 | `test/srt/test_v4_no_leak.py` — 1000 prefill iters; allocator free count returns to baseline each iter. | 4e |
| 4.4 | E2E `test/test_cp_qwen3_30b_local.py` — greedy decode token-by-token diff vs baseline; `cp_size ∈ {2, 4, 8}`; run under `SGLANG_DEBUG_KV_RESHARD=1`. | end-to-end |
| 4.5 | Memory: `max_total_num_tokens` admitted grows ~`cp_size×` at same `--mem-fraction-static`. | gain measurement |
| 4.6 | Pressure: 100 concurrent requests with `T % cp_size != 0`; verify aggregate per-rank skew stays bounded by `cp_size − 1` pages per rank (no rank starves before peers). | partition balance |
| 5.* | MLA + NSA + musa + NPU device parity. | Part 5 |
| 6.* | PD-disagg regression: decode outputs unchanged after sharded prefill. | Part 6 |
| Manual | Startup refuses `--enable-cp-kv-reshard` combined with hicache / hisparse / SWA+cp_size>1 / decode-disagg / cross-attn / spec-v2. | guardrails |

## 11. Risks and open questions

- **Allocator fragmentation under churn** — per-forward batched alloc/free of `~(cp_size−1)/cp_size × max_prefill_tokens` rows. `TokenToKVPoolAllocator.free` uses `torch.cat`, O(N) per call but cheap at typical sizes (~tens of μs). Monitor; set `need_sort=False` on the prefill allocator if helpful.
- **Allocator starvation** — `MirroredCpAvailability` must subtract `transient_reserve` before exposing free count to admission so the allocator never sees a starve from concurrent transient allocations.
- **`cp_owner_per_page` stability across chunked-prefill chunks** — the same logical page must keep the same owner across chunks. Contiguous partitioning is *not* prefix-stable (the owner array for `total=N` is not a prefix of the array for `total=N+k`), so the design relies on locking at admission: compute `req.cp_owner_per_page` once with the full input length and reuse it across chunks. v1's disagg-prefill-only scope makes this trivial — the full input length is known at admission and decode growth happens on a separate node.
- **Per-rank `evictable_size_` vs physical availability skew** — the tree's `evictable_size_` is logical; per-rank physical free count can lag. Eviction loop must drive `MirroredCpAvailability.can_admit` until **every** rank fits, not just one.
- **Disagg compatibility** — `filter_kv_indices_for_cp_rank` and `page_indices_to_cp_rank_page_indices` already use the same contiguous partition rule, so prefill-write ownership matches the disaggregation transfer ownership without a second mapping.
- **MLA + NSA per-pool semantics** — MLA's fused buffer halves the transient bytes; NSA also needs FP8 scale rows in the transient reserve.
- **Spec-v2 / draft-extend gating** — `extend_seq_lens_cpu` is not always populated on the draft-extend path. v1 gates the new write path on `forward_mode == EXTEND` only; spec-v2 stays on the legacy un-resharded path.
- **Cross-attention layers** — `cache_loc` is `encoder_out_cache_loc` on cross-attention; ownership semantics differ. v1 asserts against `any(layer.is_cross_attention)` when combined with the flag.
- **Non-power-of-2 CP** — handled by the `rem` branch in `owner_for_page` and `owned_slice`; verified by the partition unit test.
- **Init-time pool sizing across CP** — today `max_total_num_tokens` is synced across ranks via the world-group `all_reduce(MIN)` inside `get_available_gpu_memory` (`utils/common.py:606-611`). Add an explicit CP-group assertion at pool init under `--enable-cp-kv-reshard`. Wrap `MHATokenToKVPool._create_buffers`'s `torch.zeros` allocation in a try/`all_reduce(MIN)` barrier so init OOMs fail loudly together.
- **Staging buffer dtype mismatch with quantized pools** — when pool dtype is FP8 / INT8, the staging buffer must match (so the same scatter path works without dtype-aware loads). v1 picks staging dtype = pool dtype to keep the scatter trivial.
