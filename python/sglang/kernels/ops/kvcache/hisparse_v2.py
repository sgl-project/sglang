"""HiSparse V2 dual-source swap-in triton kernel.

For each top-k token position:
- If device_locs (req_to_token) has a valid GPU index (>= 0): return
  directly (GPU hit; covers all non-V2 requests in a mixed batch)
- If device_locs has -1 (evicted): look up host pool index from host_locs,
  DMA KV row bytes from pinned host memory to a temp GPU slot, return
  the temp slot
- If no host/temp index is available: return -1, which downstream sparse
  attention kernels treat as a masked-out (invalid) position

The host buffer is addressed through a per-layer pointer table
(host_ptrs, int64 device tensor) instead of a direct tensor argument:
CUDA graphs are captured before the HiCache host pool is attached to the
coordinator, so a direct pointer would be baked in as garbage. The
pointer table is a persistent tensor whose CONTENT is filled once the
host pool exists — graph replay loads the real pointer at run time.

All tensors are persistent (fixed data_ptr, content updated before
replay) → overlap and CUDA-graph safe. Copies are uint8 byte moves
(bit-exact): KV rows hold mixed fp8 payload + scale/rope bytes.
"""

import triton
import triton.language as tl


@triton.jit
def swap_in_plan_kernel(
    req_pool_indices,  # [padded_bs] GPU int64 — persistent
    topk_indices,  # [padded_bs, TOPK] GPU int32 — from DSA indexer
    device_locs,  # [max_reqs, max_seq] GPU int32 — req_to_token
    host_locs,  # [max_reqs, max_seq] GPU int64 — token-level host indices
    slot_pos,  # [max_reqs, LAYERS, NSLOT] int32 — resident position per
    #   device-buffer slot (-1 = empty), persistent, in-place
    pos2slot,  # [max_reqs, max_seq] int32 — transient position→slot
    #   scratch, all -1 between calls (self-cleaning)
    scratch,  # [max_reqs, 2*NSLOT + TOPK] int32 — transient scratch
    plan,  # [max_reqs, TOPK] int32 — out: hit → slot,
    #   miss → slot + NSLOT, non-buffer → -1
    num_real_reqs,  # [1] GPU int32
    layer_id,  # int (runtime scalar, fixed per captured launch)
    num_layers,  # int (runtime scalar)
    device_locs_stride: tl.constexpr,
    host_locs_stride: tl.constexpr,
    pos2slot_stride: tl.constexpr,
    TOPK: tl.constexpr,
    NSLOT: tl.constexpr,
):
    """Phase A: stable-slot assignment for the SINGLE temp device
    buffer — the plan side of V1's load_cache_to_device_buffer. NSLOT
    (device-buffer slots, >= TOPK, power of 2) is decoupled from TOPK
    (lanes) so the buffer can exceed top_k (V1's device_buffer_size
    knob):

    - a top-k position already resident in a slot HITS and RETAINS that
      slot (zero copy, zero DMA);
    - misses take free slots empty-first (then stale residents in slot
      order), so extra buffer capacity fills before anything is evicted;
    - non-retained, non-assigned slots keep their stale resident, so
      residency older than one step keeps hitting (V1's LRU-ish reuse).

    Where V1 uses a shared-memory hash table + warp ballots, this uses a
    global position→slot scratch row (scatter, barrier, gather) and
    tl.cumsum compaction — one program per request, no sorts. The
    scratch rows are shared across layers (plan calls are sequential on
    the stream) and self-cleaned before return, preserving the all -1
    invariant. Updates slot_pos in place; no host-side maintenance.
    """
    rid = tl.program_id(0)

    real_bs = tl.load(num_real_reqs)
    if rid >= real_bs:
        return

    req_idx = tl.load(req_pool_indices + rid)
    rec = slot_pos + req_idx * num_layers * NSLOT + layer_id * NSLOT
    p2s = pos2slot + req_idx * pos2slot_stride
    flag_buf = scratch + req_idx * (2 * NSLOT + TOPK)  # retained flags
    free_buf = flag_buf + NSLOT  # compacted free slots
    miss_buf = free_buf + NSLOT  # compacted miss positions

    k = tl.arange(0, TOPK)  # lanes
    s = tl.arange(0, NSLOT)  # slots

    pos = tl.load(topk_indices + rid * TOPK + k)
    safe_pos = tl.where(pos < 0, 0, pos)
    dloc = tl.load(device_locs + req_idx * device_locs_stride + safe_pos)
    hloc = tl.load(host_locs + req_idx * host_locs_stride + safe_pos)
    # Buffer-eligible: selected, evicted, host copy exists. (Evicted with
    # no host copy is masked by phase B and must NOT claim a slot.)
    needed = (pos >= 0) & (dloc < 0) & (hloc >= 0)
    npos = tl.where(needed, pos, 0)  # 0-safe index; gated by `needed`

    resident = tl.load(rec + s)
    has_res = resident >= 0
    safe_res = tl.where(has_res, resident, 0)

    # (1) publish resident → slot mapping; zero the retained flags.
    tl.store(p2s + safe_res, s, mask=has_res)
    tl.store(flag_buf + s, 0)
    tl.debug_barrier()

    # (2) hits: positions found in the mapping; mark their slots retained.
    hs = tl.load(p2s + npos, mask=needed, other=-1)
    hit = needed & (hs >= 0)
    hit_slot = tl.where(hit, hs, 0)
    tl.store(flag_buf + hit_slot, 1, mask=hit)
    tl.debug_barrier()

    # (3) compact free slots and miss positions by rank. Free slots are
    # ordered empty-first (then stale residents in slot order), so extra
    # buffer capacity fills up before any stale resident is overwritten.
    retained = tl.load(flag_buf + s) == 1
    is_free = retained == 0
    empty_free = is_free & (has_res == 0)
    stale_free = is_free & has_res
    n_empty = tl.sum(empty_free.to(tl.int32), axis=0)
    rank_e = tl.cumsum(empty_free.to(tl.int32), axis=0) - 1
    rank_s = tl.cumsum(stale_free.to(tl.int32), axis=0) - 1 + n_empty
    free_rank = tl.where(empty_free, rank_e, rank_s)
    miss = needed & (hit == 0)
    miss_rank = tl.cumsum(miss.to(tl.int32), axis=0) - 1
    num_miss = tl.sum(miss.to(tl.int32), axis=0)
    tl.store(free_buf + free_rank, s, mask=is_free)
    tl.store(miss_buf + miss_rank, pos, mask=miss)
    tl.debug_barrier()

    # (4) m-th miss ← m-th free slot; assigned slots take the m-th miss
    # position. #free >= #miss always (residents unique ⇒ #retained ==
    # #hits ⇒ #free = NSLOT - #hits >= TOPK - #hits >= #miss).
    assigned = tl.load(free_buf + tl.where(miss, miss_rank, 0), mask=miss, other=0)
    slot_assigned = is_free & (free_rank < num_miss)
    new_res = tl.load(
        miss_buf + tl.where(slot_assigned, free_rank, 0),
        mask=slot_assigned,
        other=0,
    )

    plan_val = tl.where(hit, hit_slot, tl.where(miss, assigned + NSLOT, -1))
    tl.store(plan + req_idx * TOPK + k, plan_val)
    tl.store(rec + s, tl.where(slot_assigned, new_res, resident))
    tl.debug_barrier()

    # (5) self-clean the shared mapping row for the next layer's call.
    tl.store(p2s + safe_res, -1, mask=has_res)


@triton.jit
def swap_in_serve_kernel(
    device_locs,  # [max_reqs, max_seq] GPU int32 — req_to_token
    req_pool_indices,  # [padded_bs] GPU int64
    topk_indices,  # [padded_bs, TOPK] GPU int32
    host_locs,  # [max_reqs, max_seq] GPU int64
    host_ptrs,  # [num_layers] GPU int64
    host_stride_t,  # [1] GPU int64
    device_kv,  # GPU uint8 view — per-layer KV buffer
    result,  # [padded_bs, TOPK] GPU int32 — output
    temp_slots,  # [max_reqs, NSLOT] GPU int64 — SINGLE device buffer
    plan,  # [max_reqs, TOPK] int32 — from swap_in_plan_kernel
    num_real_reqs,  # [1] GPU int32
    layer_id,  # int (runtime scalar)
    device_locs_stride: tl.constexpr,
    host_locs_stride: tl.constexpr,
    device_kv_stride: tl.constexpr,
    TOPK: tl.constexpr,
    NSLOT: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """Phase B: serve each top-k lane from the plan — device passthrough,
    zero-copy device-buffer hit (slot content untouched), or miss DMA into the
    assigned slot."""
    rid = tl.program_id(0)
    tid = tl.program_id(1)

    real_bs = tl.load(num_real_reqs)
    if rid >= real_bs:
        tl.store(result + rid * TOPK + tid, -1)
        return

    req_idx = tl.load(req_pool_indices + rid)

    topk_pos = tl.load(topk_indices + rid * TOPK + tid)
    if topk_pos < 0:
        tl.store(result + rid * TOPK + tid, -1)
        return

    gpu_idx = tl.load(device_locs + req_idx * device_locs_stride + topk_pos)
    if gpu_idx >= 0:
        tl.store(result + rid * TOPK + tid, gpu_idx)
        return

    p = tl.load(plan + req_idx * TOPK + tid)
    if p < 0:
        tl.store(result + rid * TOPK + tid, -1)
        return

    if p < NSLOT:
        # Hit: the slot already holds this position's KV — zero copy.
        t_idx = tl.load(temp_slots + req_idx * NSLOT + p)
        tl.store(result + rid * TOPK + tid, t_idx.to(tl.int32))
        return

    # Miss: DMA into the assigned slot.
    slot = p - NSLOT
    t_idx = tl.load(temp_slots + req_idx * NSLOT + slot)
    if t_idx < 0:
        tl.store(result + rid * TOPK + tid, -1)
        return

    h_idx = tl.load(host_locs + req_idx * host_locs_stride + topk_pos)
    host_base = tl.load(host_ptrs + layer_id)
    if h_idx < 0 or host_base == 0:
        tl.store(result + rid * TOPK + tid, -1)
        return
    host_kv = host_base.to(tl.pointer_type(tl.uint8))
    host_kv_stride = tl.load(host_stride_t)

    kv_offs = tl.arange(0, BLOCK_KV)
    mask = kv_offs < device_kv_stride
    src = tl.load(host_kv + h_idx * host_kv_stride + kv_offs, mask=mask, other=0)
    tl.store(device_kv + t_idx * device_kv_stride + kv_offs, src, mask=mask)

    tl.store(result + rid * TOPK + tid, t_idx.to(tl.int32))
