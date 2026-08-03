from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_paged_experts_decide_module() -> Module:
    """Compile and cache the on-device Paged Experts residency kernels (decide + gather)."""
    return load_jit(
        "paged_experts_decide",
        cuda_files=["moe/paged_experts_decide.cuh"],
        cuda_wrappers=[
            ("decide", "decide"),
            ("decide_bounded", "decide_bounded"),
            ("decide_wave", "decide_wave"),
            ("gather", "gather"),
            ("gather_multi", "gather_multi"),
            ("scatter_multi", "scatter_multi"),
            ("remap_mask", "remap_mask"),
            ("scratch_split", "scratch_split"),
            ("host_devptr", "host_devptr"),
        ],
    )


def paged_experts_scratch_split(
    l2g: torch.Tensor,
    res_src: torch.Tensor,
    res_dst: torch.Tensor,
    res_n: torch.Tensor,
    h2d_src: torch.Tensor,
    h2d_dst: torch.Tensor,
    h2d_n: torch.Tensor,
) -> None:
    """Split the streaming-prefill scratch fill by the LIVE residency map ``l2g`` ([E] int32 CUDA,
    -1 = not resident): a device-to-device plan (pool slot -> scratch expert row) for residents and a
    host-to-device plan (store row -> scratch expert row) for the rest. Counts land on-device — no host
    sync, so the plan is correct right after captured decode replays. All plan tensors are [E] int32
    CUDA; counts [1] int32 CUDA."""
    module = _jit_paged_experts_decide_module()
    module.scratch_split(l2g, res_src, res_dst, res_n, h2d_src, h2d_dst, h2d_n)


def paged_experts_decide(
    topk: torch.Tensor,
    step_ctr: torch.Tensor,
    slot_expert: torch.Tensor,
    expert_slot: torch.Tensor,
    slot_lastuse: torch.Tensor,
    freq: torch.Tensor,
    lfu: bool,
    src: torch.Tensor,
    dst: torch.Tensor,
    n_out: torch.Tensor,
    idx: torch.Tensor,
) -> None:
    """On-device keep-warm + LRU/LFU residency decision for Paged Experts (distinct active experts <= K).

    Computes the per-step paging plan entirely on the GPU — no host sync — so the decode step is
    CUDA-graph-capturable. Mutates the residency state (``step_ctr`` / ``slot_expert`` / ``expert_slot`` /
    ``slot_lastuse`` / ``freq``) in place and writes the page-in plan into the preallocated output buffers,
    which the existing ``transfer_kv_per_layer_mla`` gather then consumes (it reads the indices on-device).

    All tensors are ``int32`` and CUDA-resident. ``topk`` is ``[topk_n]`` (flattened active expert ids,
    negative = padding); ``step_ctr`` is ``[1]`` (a monotonic counter the kernel increments on-device, so a
    captured graph advances LRU recency every replay); ``slot_expert``/``slot_lastuse`` are ``[K]``;
    ``expert_slot``/``freq``/``idx`` are ``[E]``; ``src``/``dst`` are ``[>=K]`` (filled ``0..n``); ``n_out``
    is ``[1]`` (the page-in count). ``lfu`` selects LFU eviction (use-count, LRU tiebreak) over plain LRU.
    ``idx`` receives the updated logical->slot map (-1 == not resident) for the forward remap.
    """
    module = _jit_paged_experts_decide_module()
    module.decide(
        topk,
        step_ctr,
        slot_expert,
        expert_slot,
        slot_lastuse,
        freq,
        int(lfu),
        src,
        dst,
        n_out,
        idx,
    )


def paged_experts_decide_bounded(
    topk: torch.Tensor,
    step_ctr: torch.Tensor,
    slot_expert: torch.Tensor,
    expert_slot: torch.Tensor,
    slot_lastuse: torch.Tensor,
    freq: torch.Tensor,
    lfu: bool,
    log2hot: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    n_out: torch.Tensor,
    cold_log: torch.Tensor,
    cold_n: torch.Tensor,
    idx: torch.Tensor,
    needed: torch.Tensor,
    doorbell: int = 0,
) -> None:
    """On-device keep-warm + LRU/LFU decision for the pinned-WINDOW store (distinct active experts <= K).

    Like :func:`paged_experts_decide`, but splits the page-in plan by window membership so the captured
    gather only reads the pinned hot block. ``log2hot[e]`` is the hot-block index if expert ``e`` is in the
    pinned window (else -1). Window hits go to ``(src, dst, n_out)`` (on-device gather from ``host_hot``).
    Cold misses record their **logical** id in ``cold_log`` and stay unresident (no eviction — masked this
    replay) for the host to stage out-of-graph: at the post-replay refill (replay-twice) or an in-layer
    eager break (BCG). ``needed[s]`` marks slots holding an expert needed this step (the refill must not
    evict them, or the replay-twice loop never converges).

    All tensors are ``int32`` CUDA. Shapes: ``topk`` ``[topk_n]``; ``step_ctr``/``n_out``/``cold_n`` ``[1]``;
    ``slot_expert``/``slot_lastuse``/``src``/``dst``/``cold_log``/``needed`` ``[K]`` (``src``/``dst``/
    ``cold_log`` are ``[>=K]`` plan buffers); ``expert_slot``/``freq``/``idx``/``log2hot`` ``[E]``.
    ``step_ctr`` is bumped on-device so a captured graph advances recency every replay. ``doorbell``
    (optional): a MAPPED PINNED host address; the kernel writes the cold count there host-visibly
    (``__threadfence_system``) so the BCG break can spin on plain memory instead of paying a stream
    sync per layer — the host resets it to a sentinel before each replay.
    """
    module = _jit_paged_experts_decide_module()
    module.decide_bounded(
        topk,
        int(lfu),
        log2hot,
        step_ctr,
        slot_expert,
        expert_slot,
        slot_lastuse,
        freq,
        src,
        dst,
        n_out,
        cold_log,
        cold_n,
        int(doorbell),
        idx,
        needed,
    )


def paged_experts_decide_wave(
    topk: torch.Tensor,
    num_experts: int,
    num_slots: int,
    wave: int,
    src: torch.Tensor,
    dst: torch.Tensor,
    n_out: torch.Tensor,
    idx: torch.Tensor,
    slot_base: int = 0,
) -> None:
    """On-device static fixed-wave decision for Paged Experts (distinct active experts > K).

    Expert ``e`` has a static home — wave ``floor(e/K)``, slot ``e % K + slot_base``. For ``wave`` this
    emits the page-in plan for the distinct in-wave experts present in ``topk`` and writes ``idx`` so
    out-of-wave experts map to -1 (masked to weight 0). The caller runs ``ceil(num_experts/num_slots)``
    waves and sums the per-wave GEMM partials — lossless. ``slot_base`` banks the slot pool for
    double-buffered waves (page bank B while bank A computes). No eviction, no state mutation, no host
    sync (capturable).

    All tensors are ``int32`` CUDA: ``topk`` ``[topk_n]``, ``src``/``dst`` ``[>=num_slots]``, ``n_out``
    ``[1]``, ``idx`` ``[num_experts]``.
    """
    module = _jit_paged_experts_decide_module()
    module.decide_wave(
        topk,
        int(num_experts),
        int(num_slots),
        int(wave),
        int(slot_base),
        src,
        dst,
        n_out,
        idx,
    )


def paged_experts_host_devptr(pinned: torch.Tensor) -> int:
    """UVA device pointer of a pinned host tensor, resolved once at setup (not during capture). Pass the
    result to ``paged_experts_gather`` so no host CUDA call runs inside the captured region.
    """
    module = _jit_paged_experts_decide_module()
    return int(module.host_devptr(pinned))


def paged_experts_gather(
    store_devptr: int,
    slot: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    n_out: torch.Tensor,
    item_bytes: int,
) -> None:
    """Copy the ``*n_out`` experts ``src[i] -> dst[i]`` from the pinned host store (``store_devptr``, from
    ``paged_experts_host_devptr``) into the GPU slot pool. The count is read on-device, so under CUDA-graph
    capture each replay moves exactly the experts ``decide`` chose this step. ``slot`` is the device pool
    tensor; ``src``/``dst``/``n_out`` are ``int32`` CUDA; ``item_bytes`` is the per-expert block size and
    must be 16-byte aligned (float4 copy). Copy-only — marlin int4 / bf16 rows travel packed.
    """
    module = _jit_paged_experts_decide_module()
    module.gather(int(store_devptr), slot, src, dst, n_out, int(item_bytes))


def paged_experts_gather_multi(
    stores: torch.Tensor,
    slots: torch.Tensor,
    e16s: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    n_out: torch.Tensor,
) -> None:
    """Fused multi-tensor gather: page ALL of a layer's paged tensors in ONE capturable launch (a marlin
    int4 layer has 6 paged tensors -> 6 launches collapse to 1; hundreds of graph nodes per token saved).

    ``stores``/``slots``/``e16s`` are ``[ntens]`` **int64 CUDA** descriptor tensors built once at setup:
    per-tensor pinned-store UVA base pointers (:func:`paged_experts_host_devptr`), GPU slot-pool base
    pointers (``tensor.data_ptr()``), and per-expert block bytes / 16 (the caller validates 16-byte
    alignment). ``src``/``dst``/``n_out`` are the shared int32 CUDA page-in plan the decide kernel wrote —
    the count is read on-device, so under capture each replay moves exactly the chosen experts.
    """
    module = _jit_paged_experts_decide_module()
    module.gather_multi(stores, slots, e16s, src, dst, n_out)


def paged_experts_scatter_multi(
    stages: torch.Tensor,
    slots: torch.Tensor,
    e16s: torch.Tensor,
    dst: torch.Tensor,
    n: int,
) -> None:
    """Fused multi-tensor scatter — the refill's inverse of :func:`paged_experts_gather_multi`: one
    launch copies ``n`` staged rows (contiguous per tensor in a device staging area) into their victim
    slots for ALL of a layer's paged tensors, replacing per-tensor-per-expert micro-copies.

    ``stages``/``slots``/``e16s`` are ``[ntens]`` int64 CUDA descriptor tensors (device staging bases,
    slot-pool bases, per-expert bytes / 16); ``dst`` is the int32 CUDA slot-index plan; ``n`` is the
    host-known staged-row count.
    """
    module = _jit_paged_experts_decide_module()
    module.scatter_multi(stages, slots, e16s, dst, int(n))


def paged_experts_remap_mask(
    topk: torch.Tensor,
    idx: torch.Tensor,
    tw: torch.Tensor,
    safe_ids: torch.Tensor,
    masked_tw: torch.Tensor,
) -> None:
    """Fused remap + routing-weight mask: one capturable launch replacing the per-layer python chain
    ``remap = idx[topk]; safe_ids = where(remap>=0, remap, 0); masked_tw = where(remap>=0, tw, 0)``
    (5 elementwise launches). Reads the LIVE ``idx`` map, so it may run after an in-graph staging break
    (BCG) and sees just-staged experts.

    ``topk``/``safe_ids`` are ``[T]`` int32 CUDA (flattened top-k logical ids / out slot ids, masked -> 0);
    ``idx`` is ``[E]`` int32 (logical -> slot, -1 = masked); ``tw``/``masked_tw`` are ``[T]`` float32
    (routing weights in / masked weights out).
    """
    module = _jit_paged_experts_decide_module()
    module.remap_mask(topk, idx, tw, safe_ids, masked_tw)
