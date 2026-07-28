"""Step-4 LoRA-B schedule candidates (benchmark tier, plan §64.1).

The production incumbent is ``stock_grouped_lora_b`` — the upstream
fused-MoE GEMM, launched once per slice (twice at gate/up). Candidates
here challenge its launch topology and ownership:

* **one-launch sliced** (grouped): ONE launch whose N tiles are laid
  out slice-major — a tile never crosses the slice boundary; the slice
  is derived per N tile and selects both the bridge K-range (slice
  ``s`` reads bridge columns ``[s*R, (s+1)*R)``) and the destination
  column offset. What it removes vs per-slice is ONE KERNEL LAUNCH at
  gate/up — total CTA count and per-CTA plan loads are essentially
  unchanged. (Formerly misnamed ``fused_flat`` — gate-4 review: it is
  not a contiguous-flat layout; for BLOCK_N dividing the slice width
  the two layouts are the same kernel.)
* **lean per-slice** (grouped): the SAME lean body, launched once per
  slice (twice at gate/up). Exists to decompose the one-launch win into
  body leanness vs launch fusion — the gate-4 confound isolation arm.
* **indexed** (raw route): derives the fused ``(adapter, LoRA expert)``
  key per pair inline — no aligned plan at all. Its prize is LEG-level:
  with indexed A AND indexed B, the leg builds zero route kernels, the
  configuration Step 3 could not measure because stock B forced the plan.
* **deterministic rank-split** (grouped one-launch body): splits the K axis
  (K = the LoRA rank, 16-128) across programs with an FP32 workspace and
  a fixed-order reduce. B has abundant N-parallelism at prefill, so this
  arm exists for the LOW-WAVE DECODE corner only; the workspace guard
  enforces that honestly.

THE CONTRACT every family must satisfy (it is what makes destination
buffer reuse safe under one CUDA graph, pinned by the registered tests):
every destination cell targeted by ``destination_offsets`` is WRITTEN on
every call — valid pairs get the GEMM result, everything else (sentinel
routes, base tokens, non-owned experts) gets EXACT ZERO. This is stricter
than A's preserve-contract because B's output is consumed additively by
the activation join and the combine.

``intermediate_top_k`` follows ``stock_grouped_lora_b``: the bridge row
for pair ``p`` is ``p // intermediate_top_k``, so 1 consumes the
canonical pair-major bridge and ``top_k`` consumes the token-dedup
token-major bridge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import triton
import triton.language as tl

from benchmark.kernels.lora_moe.lora_b_execution import LoraBExecutionSpec
from sglang.srt.lora.sgl_lora.bf16 import stock_grouped_lora_b
from sglang.srt.lora.sgl_lora.routing import RouteView, virtual_expert_ids_inline

ONE_LAUNCH_DEFAULT_CONFIG: dict[str, int] = {
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 3,
}
INDEXED_B_DEFAULT_CONFIG: dict[str, int] = {
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 32,
    "num_warps": 4,
    "num_stages": 3,
}
RANK_SPLIT_DEFAULT_CONFIG: dict[str, int] = {
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 16,
    "GROUP_SIZE_M": 8,
    "SPLIT_K": 2,
    "num_warps": 4,
    "num_stages": 3,
}
# FP32 partials are S_k x the whole delta buffer; cap keeps the arm inside
# its justified regime (low-wave decode) instead of silently allocating
# hundreds of MB at prefill.
RANK_SPLIT_WORKSPACE_CAP_BYTES = 256 * 1024 * 1024


def _validate_b_call(
    bridge: torch.Tensor,
    weight: torch.Tensor,
    destination: torch.Tensor,
    routing: RouteView,
    *,
    destination_offsets: Sequence[int],
    intermediate_top_k: int,
) -> tuple[int, int, int]:
    """Shared contract check; returns (num_slices, slice_width, rank)."""
    num_slices = len(destination_offsets)
    if num_slices not in (1, 2):
        raise ValueError(f"1 or 2 destination offsets, got {num_slices}")
    num_groups, weight_rows, rank = weight.shape
    if weight_rows % num_slices:
        raise ValueError(
            f"weight rows {weight_rows} not divisible by {num_slices} slices"
        )
    slice_width = weight_rows // num_slices
    # Fail-closed offsets (gate-4 review finding 6): a negative offset
    # under-runs the buffer silently via int64 pointer math; overlapping
    # slices create concurrent writes to the same cells, which breaks
    # both the zero-fill contract and the determinism guarantees.
    ordered = sorted(int(o) for o in destination_offsets)
    if ordered[0] < 0:
        raise ValueError(f"destination offsets must be >= 0, got {ordered}")
    for low, high in zip(ordered, ordered[1:]):
        if high - low < slice_width:
            raise ValueError(
                f"destination offsets {ordered} overlap for slice width "
                f"{slice_width}; slices must write disjoint columns"
            )
    expected_groups = routing.max_loras * routing.lora_experts_per_adapter
    if num_groups != expected_groups:
        raise ValueError(
            f"weight groups {num_groups} != max_loras * "
            f"lora_experts_per_adapter {expected_groups}"
        )
    num_pairs = routing.topk_ids.numel()
    num_tokens, top_k = routing.topk_ids.shape
    if intermediate_top_k == 1:
        expected_rows = num_pairs
    elif intermediate_top_k == top_k:
        expected_rows = num_tokens
    else:
        raise ValueError(
            f"intermediate_top_k must be 1 or the route top_k {top_k}, got "
            f"{intermediate_top_k}"
        )
    if bridge.shape != (expected_rows, num_slices * rank):
        raise ValueError(
            f"bridge must be {(expected_rows, num_slices * rank)}, got "
            f"{tuple(bridge.shape)}"
        )
    if destination.shape[0] != num_pairs:
        raise ValueError(f"destination must have {num_pairs} rows")
    for offset in destination_offsets:
        if int(offset) + slice_width > destination.shape[1]:
            raise ValueError(
                f"destination offset {offset} + slice width {slice_width} "
                f"exceeds destination columns {destination.shape[1]}"
            )
    devices = {bridge.device, weight.device, destination.device}
    if len(devices) != 1:
        raise ValueError(f"tensors span devices {sorted(map(str, devices))}")
    return num_slices, slice_width, rank


@triton.jit
def _one_launch_sliced_lora_b_kernel(
    bridge_ptr,
    weight_ptr,
    destination_ptr,
    sorted_pair_ids_ptr,
    block_virtual_expert_ids_ptr,
    num_pairs_post_padded_ptr,
    num_pairs,
    dest_offset_0,
    dest_offset_1,
    stride_bm,
    stride_bk,
    stride_wg,
    stride_wn,
    stride_wk,
    stride_dm,
    stride_dn,
    TOP_K: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    N_PER_SLICE: tl.constexpr,
    RANK: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """All slices of one B site in ONE launch over the aligned plan.

    N tiles are laid out slice-major: tiles [0, TILES_PER_SLICE) are slice
    0, the next TILES_PER_SLICE are slice 1. The slice selects the bridge
    K-range and the destination offset; everything else is the stock
    grouped-B body, including the sentinel zero-store.
    """
    pid = tl.program_id(0)
    num_pairs_post_padded = tl.load(num_pairs_post_padded_ptr)
    tiles_per_slice: tl.constexpr = (N_PER_SLICE + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_n: tl.constexpr = NUM_SLICES * tiles_per_slice
    programs_per_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // programs_per_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(NUM_M_BLOCKS - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % programs_per_group) % group_size_m)
    pid_n = (pid % programs_per_group) // group_size_m
    if pid_m * BLOCK_SIZE_M >= num_pairs_post_padded:
        return

    slice_id = pid_n // tiles_per_slice
    n_tile = pid_n % tiles_per_slice
    dest_offset = tl.where(slice_id == 0, dest_offset_0, dest_offset_1).to(tl.int64)

    pair_slots = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    pair_ids = tl.load(sorted_pair_ids_ptr + pair_slots).to(tl.int64)
    pair_mask = pair_ids < num_pairs
    virtual_expert_id = tl.load(block_virtual_expert_ids_ptr + pid_m).to(tl.int64)

    n_offsets = n_tile * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    n_mask = n_offsets < N_PER_SLICE
    dest_ptrs = (
        destination_ptr
        + pair_ids[:, None] * stride_dm
        + (dest_offset + n_offsets)[None, :] * stride_dn
    )
    store_mask = pair_mask[:, None] & n_mask[None, :]

    if virtual_expert_id == -1:
        # The buffer-reuse contract: sentinel routes OWN their destination
        # cells and must zero them (base tokens, non-owned experts).
        zeros = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        tl.store(
            dest_ptrs,
            zeros.to(destination_ptr.dtype.element_ty),
            mask=store_mask,
        )
        return

    bridge_rows = pair_ids // TOP_K
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_begin in range(0, RANK, BLOCK_SIZE_K):
        k_offsets = k_begin + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        k_mask = k_offsets < RANK
        lhs = tl.load(
            bridge_ptr
            + bridge_rows[:, None] * stride_bm
            + (slice_id * RANK + k_offsets)[None, :] * stride_bk,
            mask=pair_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        rhs = tl.load(
            weight_ptr
            + virtual_expert_id * stride_wg
            + (slice_id * N_PER_SLICE + n_offsets)[None, :] * stride_wn
            + k_offsets[:, None] * stride_wk,
            mask=n_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        accumulator += tl.dot(lhs, rhs, out_dtype=tl.float32)

    tl.store(
        dest_ptrs,
        accumulator.to(destination_ptr.dtype.element_ty),
        mask=store_mask,
    )


def _launch_sliced_kernel(
    bridge: torch.Tensor,
    weight: torch.Tensor,
    destination: torch.Tensor,
    routing: RouteView,
    *,
    offsets: tuple[int, ...],
    slice_width: int,
    rank: int,
    config: Mapping[str, int],
    intermediate_top_k: int,
) -> None:
    """One kernel launch covering ``len(offsets)`` slices slice-major."""
    num_slices = len(offsets)
    num_pairs = routing.topk_ids.numel()
    block_size_n = int(config["BLOCK_SIZE_N"])
    num_m_blocks = triton.cdiv(routing.sorted_pair_ids.numel(), routing.block_size)
    num_pid_n = num_slices * triton.cdiv(slice_width, block_size_n)
    _one_launch_sliced_lora_b_kernel[(num_m_blocks * num_pid_n,)](
        bridge,
        weight,
        destination,
        routing.sorted_pair_ids,
        routing.block_virtual_expert_ids,
        routing.num_pairs_post_padded,
        num_pairs,
        offsets[0],
        offsets[1] if num_slices == 2 else offsets[0],
        bridge.stride(0),
        bridge.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        destination.stride(0),
        destination.stride(1),
        TOP_K=intermediate_top_k,
        NUM_SLICES=num_slices,
        N_PER_SLICE=slice_width,
        RANK=rank,
        NUM_M_BLOCKS=num_m_blocks,
        BLOCK_SIZE_M=routing.block_size,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=int(config["BLOCK_SIZE_K"]),
        GROUP_SIZE_M=int(config["GROUP_SIZE_M"]),
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )


def invoke_one_launch_sliced_lora_b(
    bridge: torch.Tensor,
    weight: torch.Tensor,
    destination: torch.Tensor,
    routing: RouteView,
    *,
    destination_offsets: Sequence[int],
    config: Mapping[str, int],
    intermediate_top_k: int = 1,
) -> None:
    num_slices, slice_width, rank = _validate_b_call(
        bridge,
        weight,
        destination,
        routing,
        destination_offsets=destination_offsets,
        intermediate_top_k=intermediate_top_k,
    )
    if routing.topk_ids.numel() == 0:
        return
    _launch_sliced_kernel(
        bridge,
        weight,
        destination,
        routing,
        offsets=tuple(int(o) for o in destination_offsets),
        slice_width=slice_width,
        rank=rank,
        config=config,
        intermediate_top_k=intermediate_top_k,
    )


def invoke_lean_per_slice_lora_b(
    bridge: torch.Tensor,
    weight: torch.Tensor,
    destination: torch.Tensor,
    routing: RouteView,
    *,
    destination_offsets: Sequence[int],
    config: Mapping[str, int],
    intermediate_top_k: int = 1,
) -> None:
    """The SAME lean body, one launch per slice (gate-4 finding 2).

    Each launch sees a single-slice view: bridge columns
    ``[s*R, (s+1)*R)``, weight rows ``[s*W, (s+1)*W)``, its own
    destination offset. Byte-identical tile work to the one-launch
    form; what differs is the LAUNCH COUNT (total CTAs and per-CTA
    plan loads are essentially unchanged) — the effect this arm
    isolates. Note the sweep tunes it independently, so decided-cell
    ratios vs one-launch mix fusion with a config interaction; the
    ``lean_matched`` decided arm (one-launch's promoted config on this
    body) isolates pure fusion.
    """
    num_slices, slice_width, rank = _validate_b_call(
        bridge,
        weight,
        destination,
        routing,
        destination_offsets=destination_offsets,
        intermediate_top_k=intermediate_top_k,
    )
    if routing.topk_ids.numel() == 0:
        return
    offsets = tuple(int(o) for o in destination_offsets)
    for slice_id, offset in enumerate(offsets):
        _launch_sliced_kernel(
            bridge[:, slice_id * rank : (slice_id + 1) * rank],
            weight[:, slice_id * slice_width : (slice_id + 1) * slice_width, :],
            destination,
            routing,
            offsets=(offset,),
            slice_width=slice_width,
            rank=rank,
            config=config,
            intermediate_top_k=intermediate_top_k,
        )


@triton.jit
def _indexed_lora_b_kernel(
    bridge_ptr,
    weight_ptr,
    destination_ptr,
    topk_ids_ptr,
    token_slots_ptr,
    lora_expert_map_ptr,
    num_pairs,
    routed_expert_id_bound,
    dest_offset_0,
    dest_offset_1,
    stride_bm,
    stride_bk,
    stride_wg,
    stride_wn,
    stride_wk,
    stride_dm,
    stride_dn,
    TOP_K: tl.constexpr,
    INTERMEDIATE_TOP_K: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    N_PER_SLICE: tl.constexpr,
    RANK: tl.constexpr,
    LORA_EXPERTS_PER_ADAPTER: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    USE_LORA_EXPERT_MAP: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """No-plan B: one program per (pair, N tile), key derived inline.

    Mirrors the indexed-A structure (vector reduction, serial K,
    deterministic) with B's OWN sentinel contract: an invalid pair STORES
    ZEROS to its destination tile — A preserves, B must own the cell
    because the activation join and combine read it unconditionally.
    """
    pair = tl.program_id(0)
    pid_n = tl.program_id(1)
    tiles_per_slice = (N_PER_SLICE + BLOCK_N - 1) // BLOCK_N
    slice_id = pid_n // tiles_per_slice
    n_tile = pid_n % tiles_per_slice
    dest_offset = tl.where(slice_id == 0, dest_offset_0, dest_offset_1).to(tl.int64)

    key = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_slots_ptr,
        lora_expert_map_ptr,
        pair,
        pair < num_pairs,
        routed_expert_id_bound,
        LORA_EXPERTS_PER_ADAPTER=LORA_EXPERTS_PER_ADAPTER,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        USE_LORA_EXPERT_MAP=USE_LORA_EXPERT_MAP,
        SHARED_OUTER=SHARED_OUTER,
    )
    valid = key != -1
    group = tl.maximum(key, 0).to(tl.int64)
    pair64 = pair.to(tl.int64)
    bridge_row = pair64 // INTERMEDIATE_TOP_K

    offs_n = n_tile.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    n_mask = offs_n < N_PER_SLICE
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(RANK, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K).to(tl.int64)
        k_mask = offs_k < RANK
        x = tl.load(
            bridge_ptr
            + bridge_row * stride_bm
            + (slice_id * RANK + offs_k) * stride_bk,
            mask=valid & k_mask,
            other=0.0,
        )
        w = tl.load(
            weight_ptr
            + group * stride_wg
            + (slice_id * N_PER_SLICE + offs_n)[:, None] * stride_wn
            + offs_k[None, :] * stride_wk,
            mask=valid & n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        accumulator += tl.sum(w.to(tl.float32) * x[None, :].to(tl.float32), axis=1)

    # Invalid pairs fall through with a zero accumulator and STORE it —
    # exactly the sentinel zero-fill the grouped families perform.
    tl.store(
        destination_ptr + pair64 * stride_dm + (dest_offset + offs_n) * stride_dn,
        accumulator.to(destination_ptr.dtype.element_ty),
        mask=n_mask,
    )


def invoke_indexed_lora_b(
    bridge: torch.Tensor,
    weight: torch.Tensor,
    destination: torch.Tensor,
    routing: RouteView,
    *,
    destination_offsets: Sequence[int],
    config: Mapping[str, int],
    intermediate_top_k: int = 1,
) -> None:
    """Launch off a RouteView's SOURCES; ROUTE_RAW is the honest request."""
    num_slices, slice_width, rank = _validate_b_call(
        bridge,
        weight,
        destination,
        routing,
        destination_offsets=destination_offsets,
        intermediate_top_k=intermediate_top_k,
    )
    num_pairs = routing.topk_ids.numel()
    if num_pairs == 0:
        return
    offsets = tuple(int(o) for o in destination_offsets)
    block_n = int(config["BLOCK_SIZE_N"])
    use_map = routing.lora_expert_map is not None
    shared = routing.shared_outer_local_expert_count is not None
    bound = (
        routing.shared_outer_local_expert_count
        if shared
        else (routing.lora_expert_map.numel() if use_map else 0)
    )
    map_arg = routing.lora_expert_map if use_map else routing.topk_ids
    _indexed_lora_b_kernel[(num_pairs, num_slices * triton.cdiv(slice_width, block_n))](
        bridge,
        weight,
        destination,
        routing.topk_ids,
        routing.token_slots,
        map_arg,
        num_pairs,
        bound,
        offsets[0],
        offsets[1] if num_slices == 2 else offsets[0],
        bridge.stride(0),
        bridge.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        destination.stride(0),
        destination.stride(1),
        TOP_K=routing.topk_ids.shape[1],
        INTERMEDIATE_TOP_K=intermediate_top_k,
        NUM_SLICES=num_slices,
        N_PER_SLICE=slice_width,
        RANK=rank,
        LORA_EXPERTS_PER_ADAPTER=routing.lora_experts_per_adapter,
        MAX_LORAS=routing.max_loras,
        USE_LORA_EXPERT_MAP=use_map,
        SHARED_OUTER=shared,
        BLOCK_N=block_n,
        BLOCK_K=int(config["BLOCK_SIZE_K"]),
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )


@triton.jit
def _rank_split_lora_b_partial_kernel(
    bridge_ptr,
    weight_ptr,
    workspace_ptr,
    sorted_pair_ids_ptr,
    block_virtual_expert_ids_ptr,
    num_pairs_post_padded_ptr,
    num_pairs,
    dest_offset_0,
    dest_offset_1,
    stride_bm,
    stride_bk,
    stride_wg,
    stride_wn,
    stride_wk,
    stride_ws,
    stride_wm,
    stride_wcol,
    TOP_K: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    N_PER_SLICE: tl.constexpr,
    RANK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """One K chunk of the one-launch sliced body, FP32 partial workspace.

    Sentinel blocks store ZERO partials (every slot), so the fixed-order
    reduce yields exact zeros for them without reading the plan twice.
    """
    pid = tl.program_id(0)
    split_id = tl.program_id(1)
    num_pairs_post_padded = tl.load(num_pairs_post_padded_ptr)
    tiles_per_slice: tl.constexpr = (N_PER_SLICE + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_n: tl.constexpr = NUM_SLICES * tiles_per_slice
    programs_per_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // programs_per_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(NUM_M_BLOCKS - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % programs_per_group) % group_size_m)
    pid_n = (pid % programs_per_group) // group_size_m
    if pid_m * BLOCK_SIZE_M >= num_pairs_post_padded:
        return

    slice_id = pid_n // tiles_per_slice
    n_tile = pid_n % tiles_per_slice
    dest_offset = tl.where(slice_id == 0, dest_offset_0, dest_offset_1).to(tl.int64)

    pair_slots = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    pair_ids = tl.load(sorted_pair_ids_ptr + pair_slots).to(tl.int64)
    pair_mask = pair_ids < num_pairs
    virtual_expert_id = tl.load(block_virtual_expert_ids_ptr + pid_m).to(tl.int64)

    n_offsets = n_tile * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    n_mask = n_offsets < N_PER_SLICE
    ws_ptrs = (
        workspace_ptr
        + split_id.to(tl.int64) * stride_ws
        + pair_ids[:, None] * stride_wm
        + (dest_offset + n_offsets)[None, :] * stride_wcol
    )
    store_mask = pair_mask[:, None] & n_mask[None, :]

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if virtual_expert_id != -1:
        bridge_rows = pair_ids // TOP_K
        chunk = (RANK + SPLIT_K - 1) // SPLIT_K
        k_start = split_id * chunk
        k_stop = tl.minimum(k_start + chunk, RANK)
        for k_begin in range(0, chunk, BLOCK_SIZE_K):
            k_offsets = k_start + k_begin + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            k_mask = k_offsets < k_stop
            lhs = tl.load(
                bridge_ptr
                + bridge_rows[:, None] * stride_bm
                + (slice_id * RANK + k_offsets)[None, :] * stride_bk,
                mask=pair_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            rhs = tl.load(
                weight_ptr
                + virtual_expert_id * stride_wg
                + (slice_id * N_PER_SLICE + n_offsets)[None, :] * stride_wn
                + k_offsets[:, None] * stride_wk,
                mask=n_mask[None, :] & k_mask[:, None],
                other=0.0,
            )
            accumulator += tl.dot(lhs, rhs, out_dtype=tl.float32)
    tl.store(ws_ptrs, accumulator, mask=store_mask)


@triton.jit
def _rank_split_lora_b_reduce_kernel(
    workspace_ptr,
    destination_ptr,
    sorted_pair_ids_ptr,
    num_pairs_post_padded_ptr,
    num_pairs,
    dest_offset_0,
    dest_offset_1,
    stride_ws,
    stride_wm,
    stride_wcol,
    stride_dm,
    stride_dn,
    NUM_SLICES: tl.constexpr,
    N_PER_SLICE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fixed-order slot sum over the SAME plan traversal as the partials."""
    pid = tl.program_id(0)
    num_pairs_post_padded = tl.load(num_pairs_post_padded_ptr)
    tiles_per_slice: tl.constexpr = (N_PER_SLICE + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_n: tl.constexpr = NUM_SLICES * tiles_per_slice
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_m * BLOCK_SIZE_M >= num_pairs_post_padded:
        return
    slice_id = pid_n // tiles_per_slice
    n_tile = pid_n % tiles_per_slice
    dest_offset = tl.where(slice_id == 0, dest_offset_0, dest_offset_1).to(tl.int64)

    pair_slots = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    pair_ids = tl.load(sorted_pair_ids_ptr + pair_slots).to(tl.int64)
    pair_mask = pair_ids < num_pairs
    n_offsets = n_tile * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    n_mask = n_offsets < N_PER_SLICE
    load_mask = pair_mask[:, None] & n_mask[None, :]

    total = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for split_id in range(SPLIT_K):
        total += tl.load(
            workspace_ptr
            + split_id * stride_ws
            + pair_ids[:, None] * stride_wm
            + (dest_offset + n_offsets)[None, :] * stride_wcol,
            mask=load_mask,
            other=0.0,
        )
    tl.store(
        destination_ptr
        + pair_ids[:, None] * stride_dm
        + (dest_offset + n_offsets)[None, :] * stride_dn,
        total.to(destination_ptr.dtype.element_ty),
        mask=load_mask,
    )


def rank_split_workspace_fits(destination: torch.Tensor, *, split_k: int) -> bool:
    """The ONE workspace-cap formula (squash review: the sweep had its own
    copy of this arithmetic, which could drift from the allocator's)."""
    return split_k * destination.numel() * 4 <= RANK_SPLIT_WORKSPACE_CAP_BYTES


def rank_split_b_workspace(destination: torch.Tensor, *, split_k: int) -> torch.Tensor:
    bytes_needed = split_k * destination.numel() * 4
    if bytes_needed > RANK_SPLIT_WORKSPACE_CAP_BYTES:
        raise ValueError(
            f"rank-split workspace would take {bytes_needed} bytes "
            f"(cap {RANK_SPLIT_WORKSPACE_CAP_BYTES}); the arm exists for "
            "the low-wave decode corner, not this shape"
        )
    return torch.empty(
        (split_k, destination.shape[0], destination.shape[1]),
        dtype=torch.float32,
        device=destination.device,
    )


def invoke_rank_split_lora_b(
    bridge: torch.Tensor,
    weight: torch.Tensor,
    destination: torch.Tensor,
    routing: RouteView,
    *,
    destination_offsets: Sequence[int],
    config: Mapping[str, int],
    intermediate_top_k: int = 1,
    workspace: torch.Tensor | None = None,
) -> None:
    num_slices, slice_width, rank = _validate_b_call(
        bridge,
        weight,
        destination,
        routing,
        destination_offsets=destination_offsets,
        intermediate_top_k=intermediate_top_k,
    )
    num_pairs = routing.topk_ids.numel()
    if num_pairs == 0:
        return
    split_k = int(config["SPLIT_K"])
    if split_k < 2:
        raise ValueError("SPLIT_K must be >= 2; whole-rank is the other arm")
    if workspace is None:
        workspace = rank_split_b_workspace(destination, split_k=split_k)
    else:
        # Fail-closed (gate-4 review finding 6): a non-FP32 workspace
        # silently degrades the deterministic fixed-order reduction's
        # precision contract; a wrong-device workspace faults or races.
        if workspace.shape != (split_k, destination.shape[0], destination.shape[1]):
            raise ValueError("workspace shape does not match (SPLIT_K, *destination)")
        if workspace.dtype != torch.float32:
            raise ValueError(
                f"rank-split workspace must be float32 (the deterministic "
                f"partial-plane contract), got {workspace.dtype}"
            )
        if workspace.device != destination.device:
            raise ValueError(
                f"workspace device {workspace.device} != destination "
                f"device {destination.device}"
            )
    offsets = tuple(int(o) for o in destination_offsets)
    dest_offset_0 = offsets[0]
    dest_offset_1 = offsets[1] if num_slices == 2 else offsets[0]
    block_size_n = int(config["BLOCK_SIZE_N"])
    num_m_blocks = triton.cdiv(routing.sorted_pair_ids.numel(), routing.block_size)
    num_pid_n = num_slices * triton.cdiv(slice_width, block_size_n)
    _rank_split_lora_b_partial_kernel[(num_m_blocks * num_pid_n, split_k)](
        bridge,
        weight,
        workspace,
        routing.sorted_pair_ids,
        routing.block_virtual_expert_ids,
        routing.num_pairs_post_padded,
        num_pairs,
        dest_offset_0,
        dest_offset_1,
        bridge.stride(0),
        bridge.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        workspace.stride(0),
        workspace.stride(1),
        workspace.stride(2),
        TOP_K=intermediate_top_k,
        NUM_SLICES=num_slices,
        N_PER_SLICE=slice_width,
        RANK=rank,
        SPLIT_K=split_k,
        NUM_M_BLOCKS=num_m_blocks,
        BLOCK_SIZE_M=routing.block_size,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=int(config["BLOCK_SIZE_K"]),
        GROUP_SIZE_M=int(config["GROUP_SIZE_M"]),
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )
    _rank_split_lora_b_reduce_kernel[(num_m_blocks * num_pid_n,)](
        workspace,
        destination,
        routing.sorted_pair_ids,
        routing.num_pairs_post_padded,
        num_pairs,
        dest_offset_0,
        dest_offset_1,
        workspace.stride(0),
        workspace.stride(1),
        workspace.stride(2),
        destination.stride(0),
        destination.stride(1),
        NUM_SLICES=num_slices,
        N_PER_SLICE=slice_width,
        SPLIT_K=split_k,
        NUM_M_BLOCKS=num_m_blocks,
        BLOCK_SIZE_M=routing.block_size,
        BLOCK_SIZE_N=block_size_n,
        num_warps=int(config["num_warps"]),
        num_stages=2,
    )


def run_lora_b(
    spec: LoraBExecutionSpec,
    *,
    bridge: torch.Tensor,
    weight: torch.Tensor,
    destination: torch.Tensor,
    routing: RouteView,
    destination_offsets: Sequence[int],
    config: Mapping[str, int],
    intermediate_top_k: int = 1,
    workspace: torch.Tensor | None = None,
) -> None:
    """Execute one B candidate FROM its spec — the spec IS the dispatch."""
    # Validate HERE so every family — including the stock path, which
    # otherwise runs the upstream kernel's own older checks — passes the
    # same fail-closed offset/shape contract (second-review finding 6).
    _validate_b_call(
        bridge,
        weight,
        destination,
        routing,
        destination_offsets=destination_offsets,
        intermediate_top_k=intermediate_top_k,
    )
    if spec.ownership == "grouped" and spec.slicing == "per_slice":
        stock_grouped_lora_b(
            bridge,
            weight,
            destination,
            routing,
            destination_offsets=destination_offsets,
            config=config,
            intermediate_top_k=intermediate_top_k,
        )
    elif spec.ownership == "grouped" and spec.reduction == "deterministic_rank_split":
        invoke_rank_split_lora_b(
            bridge,
            weight,
            destination,
            routing,
            destination_offsets=destination_offsets,
            config=config,
            intermediate_top_k=intermediate_top_k,
            workspace=workspace,
        )
    elif spec.ownership == "grouped" and spec.slicing == "one_launch_sliced":
        invoke_one_launch_sliced_lora_b(
            bridge,
            weight,
            destination,
            routing,
            destination_offsets=destination_offsets,
            config=config,
            intermediate_top_k=intermediate_top_k,
        )
    elif spec.ownership == "grouped" and spec.slicing == "lean_per_slice":
        invoke_lean_per_slice_lora_b(
            bridge,
            weight,
            destination,
            routing,
            destination_offsets=destination_offsets,
            config=config,
            intermediate_top_k=intermediate_top_k,
        )
    elif spec.ownership == "indexed":
        invoke_indexed_lora_b(
            bridge,
            weight,
            destination,
            routing,
            destination_offsets=destination_offsets,
            config=config,
            intermediate_top_k=intermediate_top_k,
        )
    else:
        raise NotImplementedError(f"no executor for {spec.key()!r}")
