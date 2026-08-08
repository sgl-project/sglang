"""Step-6 down-B + finalizer candidates (benchmark tier, plan §65.2).

The production incumbent is the 2-op materialized path: the down-B GEMM
writes a pair delta ``[P, H]`` and a combine op folds it into the token
output. Candidates here challenge the pair-delta materialization:

* **materialized** (FM baseline building block): the combine op alone —
  ``token_out[t] += sum_k w[t, k] * pair_delta[t*K + k]``. Its
  invalid-pair guarantee is INHERITED from B's zero-fill contract (an
  invalid pair's delta row is exact zero), so the kernel reads no route
  at all; a bench pairs it with any down-B family to form the 2-op arm.
* **token_owned** (FTOK): one program per (token, H tile), a SERIAL k
  loop deriving the fused ``(adapter, LoRA expert)`` key per pair inline
  (the ``_indexed_lora_b_kernel`` shape) — ``acc += w[t, k] *
  (bridge[pair] @ b_down[veid][h_tile])`` in FP32, then ONE
  read-modify-write into the token output. No pair delta is ever
  materialized; each destination cell has exactly one writing program,
  so the accumulate is deterministic without atomics. The grid's token
  axis is single-token by construction: a multi-token block would need
  a per-row weight matrix inside one dot (a ``[BT, BH, BK]`` gather),
  which is exactly what the measured indexed-B vector body avoids.
* **shared_rank_reduce** (FSHARED, shared-B only): the algebraic
  ``top_k``-fold FLOP cut. Every pair of a token multiplies the SAME
  shared B, so the combine weights fold in RANK space first — kernel A
  reduces ``bridge [P, R] -> tok_bridge [T, R]`` with ``tok_bridge[t] =
  sum_k w[t, k] * bridge[t*K + k]`` over valid pairs (FP32 accumulate),
  then kernel B runs ONE grouped-by-adapter GEMM ``[T, R] x [R, H]``
  through the upstream unchunked SGMV body, whose base-output add
  semantics land the ``+=`` into ``token_out`` for free. The segment
  metadata is PREPARED-tier (host-built, `build_shared_finalize_info`);
  charging a device-side builder is the §64.12 adoption gate.

THE FINALIZE CONTRACT every family must satisfy (pinned by the
registered tests): ``token_out`` is a NONZERO base destination — BF16 or
caller-selected FP32 — and every valid pair adds ``w[t, k] *
(bridge[pair] @ b_down[veid]^T)`` EXACTLY ONCE in a fixed order; invalid
pairs and zero weights contribute EXACT ZERO, so a token with no valid
pairs keeps its base row bitwise intact; replays are bitwise stable
(FP32 accumulation, serial reductions, one owner per cell — atomics
stay diagnostic-only per the Step-6 charter).
"""

from __future__ import annotations

from collections.abc import Mapping

import msgspec
import torch
import triton
import triton.language as tl

from benchmark.kernels.lora_moe.bench_sgmv_real import synthesize_unchunked_batch_info
from sglang.kernels.ops.gemm.sgemm_lora_b import _sgemm_lora_b_kernel
from sglang.srt.lora.sgl_lora.routing import RouteView, virtual_expert_ids_inline
from sglang.srt.lora.utils import LoRABatchInfo

TOKEN_FINALIZE_DEFAULT_CONFIG: dict[str, int] = {
    "BLOCK_SIZE_H": 128,
    "BLOCK_SIZE_K": 32,
    "num_warps": 4,
    "num_stages": 3,
}
MATERIALIZED_FINALIZE_DEFAULT_CONFIG: dict[str, int] = {
    "BLOCK_SIZE_T": 16,
    "BLOCK_SIZE_H": 128,
    "num_warps": 4,
    "num_stages": 2,
}
# 15th S5/6 review: FSHARED's two kernels are SEQUENTIAL and share no
# launch parameter semantics, yet one flat mapping forced a single
# num_warps/num_stages pair onto both — testing only diagonal
# combinations of their independent optima. The config is now SECTIONED:
# "reduce" drives kernel A (the [P,R]->[T,R] weighted reduce) and "gemm"
# drives kernel B (the SGMV GEMM, upstream ``sgemm_lora_b_fwd`` axes).
SHARED_RANK_REDUCE_DEFAULT_CONFIG: dict[str, dict[str, int]] = {
    "reduce": {"BLOCK_SIZE_T": 32, "num_warps": 4, "num_stages": 2},
    "gemm": {
        "BLOCK_S": 16,
        "BLOCK_N": 256,
        "BLOCK_K": 16,
        "num_warps": 4,
        "num_stages": 3,
    },
}

FINALIZE_FAMILIES = ("materialized", "token_owned", "shared_rank_reduce")
FINALIZE_OWNERSHIPS = ("per_expert", "shared")
_DESTINATION_DTYPES = (torch.bfloat16, torch.float32)


class FinalizeExecutionSpec(msgspec.Struct, frozen=True, kw_only=True):
    """One point in the finalizer candidate space (mirrors LoraBExecutionSpec).

    ``ownership`` names how ``b_down`` is keyed: ``per_expert`` carries one
    copy per (adapter, LoRA expert) — ``[G, H, R]`` with ``G = max_loras *
    lora_experts_per_adapter`` — while ``shared`` carries one copy per
    adapter, ``[L, H, R]``, under the section 60.5 shared-outer route form.
    """

    family: str
    ownership: str

    def __post_init__(self):
        for field_name, (value, vocabulary) in {
            "family": (self.family, FINALIZE_FAMILIES),
            "ownership": (self.ownership, FINALIZE_OWNERSHIPS),
        }.items():
            if value not in vocabulary:
                raise ValueError(f"{field_name}={value!r} is not one of {vocabulary}")
        if self.family == "shared_rank_reduce" and self.ownership != "shared":
            raise ValueError(
                "the weighted rank reduction folds a token's K pairs in rank "
                "space BEFORE the GEMM; that algebra only holds when every "
                "pair of the token multiplies the SAME B factor, so "
                "ownership='shared' is its precondition"
            )

    def key(self) -> str:
        parts = ["finalize", self.family]
        if self.ownership == "shared" and self.family != "shared_rank_reduce":
            parts.append("shared")
        return "_".join(parts)


def _validate_finalize_common(
    routing: RouteView,
    combine_weights: torch.Tensor,
    token_out: torch.Tensor,
    *,
    ownership: str,
) -> tuple[int, int, int]:
    """Shared contract check; returns (num_tokens, top_k, hidden)."""
    num_tokens, top_k = routing.topk_ids.shape
    if combine_weights.shape != (num_tokens, top_k):
        raise ValueError(
            f"combine_weights must be {(num_tokens, top_k)}, got "
            f"{tuple(combine_weights.shape)}"
        )
    # Fail-closed dtype: routed weights are FP32 on the production path; a
    # BF16 copy silently halves the mantissa of the exactly-once application.
    if combine_weights.dtype != torch.float32:
        raise ValueError(
            f"combine_weights must be float32, got {combine_weights.dtype}"
        )
    if token_out.ndim != 2 or token_out.shape[0] != num_tokens:
        raise ValueError(
            f"token_out must have {num_tokens} rows, got {tuple(token_out.shape)}"
        )
    if token_out.dtype not in _DESTINATION_DTYPES:
        raise ValueError(
            f"token_out dtype must be one of {_DESTINATION_DTYPES} (the "
            f"caller-selected destination contract), got {token_out.dtype}"
        )
    if ownership == "per_expert":
        if routing.shared_outer_local_expert_count is not None:
            raise ValueError(
                "ownership='per_expert' over a shared-outer route is "
                "contradictory; build the route without "
                "shared_outer_local_expert_count"
            )
    else:  # "shared" — the spec vocabulary admits nothing else
        if routing.shared_outer_local_expert_count is None:
            raise ValueError(
                "ownership='shared' requires the section 60.5 shared-outer "
                "route form (shared_outer_local_expert_count set); a "
                "per-expert route would key one B copy per expert"
            )
    devices = {combine_weights.device, token_out.device, routing.topk_ids.device}
    if len(devices) != 1:
        raise ValueError(f"tensors span devices {sorted(map(str, devices))}")
    return num_tokens, top_k, token_out.shape[1]


def _validate_weight_gemm_inputs(
    bridge: torch.Tensor,
    b_down: torch.Tensor,
    routing: RouteView,
    *,
    ownership: str,
    hidden: int,
) -> int:
    """B-weight/bridge check for the GEMM families; returns the rank."""
    if b_down.ndim != 3 or b_down.shape[1] != hidden:
        raise ValueError(
            f"b_down must be [groups, {hidden}, rank], got {tuple(b_down.shape)}"
        )
    expected_groups = (
        routing.max_loras
        if ownership == "shared"
        else routing.max_loras * routing.lora_experts_per_adapter
    )
    if b_down.shape[0] != expected_groups:
        raise ValueError(
            f"b_down carries {b_down.shape[0]} groups; ownership="
            f"{ownership!r} over this route needs {expected_groups}"
        )
    rank = b_down.shape[2]
    num_pairs = routing.topk_ids.numel()
    if bridge.shape != (num_pairs, rank):
        raise ValueError(
            f"bridge must be {(num_pairs, rank)}, got {tuple(bridge.shape)}"
        )
    devices = {bridge.device, b_down.device, routing.topk_ids.device}
    if len(devices) != 1:
        raise ValueError(f"tensors span devices {sorted(map(str, devices))}")
    return rank


def _route_key_args(routing: RouteView) -> tuple[torch.Tensor, int, bool, bool]:
    """(map_arg, bound, use_map, shared) for ``virtual_expert_ids_inline``."""
    use_map = routing.lora_expert_map is not None
    shared = routing.shared_outer_local_expert_count is not None
    bound = (
        routing.shared_outer_local_expert_count
        if shared
        else (routing.lora_expert_map.numel() if use_map else 0)
    )
    map_arg = routing.lora_expert_map if use_map else routing.topk_ids
    return map_arg, bound, use_map, shared


@triton.jit
def _token_finalize_kernel(
    bridge_ptr,
    weight_ptr,
    token_out_ptr,
    combine_weights_ptr,
    topk_ids_ptr,
    token_slots_ptr,
    lora_expert_map_ptr,
    num_pairs,
    routed_expert_id_bound,
    stride_bm,
    stride_bk,
    stride_wg,
    stride_wh,
    stride_wk,
    stride_om,
    stride_oh,
    stride_cm,
    stride_ck,
    HIDDEN: tl.constexpr,
    RANK: tl.constexpr,
    TOP_K: tl.constexpr,
    LORA_EXPERTS_PER_ADAPTER: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    USE_LORA_EXPERT_MAP: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Token-owned fused finalizer: one program per (token, H tile).

    The serial constexpr k loop visits the token's pairs in a FIXED order;
    each pair's key comes from the one canonical inline derivation, its
    bridge row and weight tile reduce exactly like ``_indexed_lora_b_kernel``
    (vector reduction, serial rank chunks, FP32), and the routed weight
    scales the pair's contribution EXACTLY ONCE. Invalid pairs mask every
    load, so they add exact zero — and a masked weight load returns zero,
    so a garbage combine-weight value at an invalid pair cannot poison the
    accumulator via ``0 * inf``. One read-modify-write lands the block on
    the nonzero base; this program is that cell's ONLY writer, which is
    what makes the accumulate deterministic without atomics.
    """
    token = tl.program_id(0)
    pid_h = tl.program_id(1)
    token64 = token.to(tl.int64)
    offs_h = pid_h.to(tl.int64) * BLOCK_H + tl.arange(0, BLOCK_H).to(tl.int64)
    h_mask = offs_h < HIDDEN

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for k in range(TOP_K):
        pair = token * TOP_K + k
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
        w = tl.load(
            combine_weights_ptr + token64 * stride_cm + k * stride_ck,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        part = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for k_block in range(0, tl.cdiv(RANK, BLOCK_K)):
            offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K).to(tl.int64)
            k_mask = offs_k < RANK
            x = tl.load(
                bridge_ptr + pair64 * stride_bm + offs_k * stride_bk,
                mask=valid & k_mask,
                other=0.0,
            )
            wt = tl.load(
                weight_ptr
                + group * stride_wg
                + offs_h[:, None] * stride_wh
                + offs_k[None, :] * stride_wk,
                mask=valid & h_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            part += tl.sum(wt.to(tl.float32) * x[None, :].to(tl.float32), axis=1)
        acc += w * part

    out_ptrs = token_out_ptr + token64 * stride_om + offs_h * stride_oh
    base = tl.load(out_ptrs, mask=h_mask, other=0.0).to(tl.float32)
    tl.store(
        out_ptrs,
        (base + acc).to(token_out_ptr.dtype.element_ty),
        mask=h_mask,
    )


def invoke_token_finalize(
    bridge: torch.Tensor,
    b_down: torch.Tensor,
    token_out: torch.Tensor,
    combine_weights: torch.Tensor,
    routing: RouteView,
    *,
    ownership: str,
    config: Mapping[str, int],
) -> None:
    """FTOK: launch off a RouteView's SOURCES; ROUTE_RAW is the honest request."""
    num_tokens, _, hidden = _validate_finalize_common(
        routing, combine_weights, token_out, ownership=ownership
    )
    rank = _validate_weight_gemm_inputs(
        bridge, b_down, routing, ownership=ownership, hidden=hidden
    )
    if routing.topk_ids.numel() == 0:
        return
    map_arg, bound, use_map, shared = _route_key_args(routing)
    block_h = int(config["BLOCK_SIZE_H"])
    _token_finalize_kernel[(num_tokens, triton.cdiv(hidden, block_h))](
        bridge,
        b_down,
        token_out,
        combine_weights,
        routing.topk_ids,
        routing.token_slots,
        map_arg,
        routing.topk_ids.numel(),
        bound,
        bridge.stride(0),
        bridge.stride(1),
        b_down.stride(0),
        b_down.stride(1),
        b_down.stride(2),
        token_out.stride(0),
        token_out.stride(1),
        combine_weights.stride(0),
        combine_weights.stride(1),
        HIDDEN=hidden,
        RANK=rank,
        TOP_K=routing.topk_ids.shape[1],
        LORA_EXPERTS_PER_ADAPTER=routing.lora_experts_per_adapter,
        MAX_LORAS=routing.max_loras,
        USE_LORA_EXPERT_MAP=use_map,
        SHARED_OUTER=shared,
        BLOCK_H=block_h,
        BLOCK_K=int(config["BLOCK_SIZE_K"]),
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )


@triton.jit
def _shared_rank_reduce_kernel(
    bridge_ptr,
    tok_bridge_ptr,
    combine_weights_ptr,
    topk_ids_ptr,
    token_slots_ptr,
    lora_expert_map_ptr,
    num_tokens,
    routed_expert_id_bound,
    stride_bm,
    stride_bk,
    stride_tm,
    stride_tk,
    stride_cm,
    stride_ck,
    RANK: tl.constexpr,
    TOP_K: tl.constexpr,
    LORA_EXPERTS_PER_ADAPTER: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    USE_LORA_EXPERT_MAP: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """FSHARED kernel A: weighted segmented reduce ``[P, R] -> [T, R]``.

    Combine weights fold in RANK space (the top_k-fold FLOP cut): FP32
    accumulate over a FIXED constexpr k order, invalid pairs masked to
    exact zero. EVERY token row is stored — a token with no valid pairs
    owns an exact-zero row, so the downstream grouped GEMM adds zero for
    it rather than reading garbage.
    """
    pid = tl.program_id(0)
    tokens = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = tokens < num_tokens
    tokens64 = tokens.to(tl.int64)
    offs_r = tl.arange(0, BLOCK_R).to(tl.int64)
    r_mask = offs_r < RANK

    acc = tl.zeros((BLOCK_T, BLOCK_R), dtype=tl.float32)
    for k in range(TOP_K):
        pairs = tokens * TOP_K + k
        keys = virtual_expert_ids_inline(
            topk_ids_ptr,
            token_slots_ptr,
            lora_expert_map_ptr,
            pairs,
            t_mask,
            routed_expert_id_bound,
            LORA_EXPERTS_PER_ADAPTER=LORA_EXPERTS_PER_ADAPTER,
            MAX_LORAS=MAX_LORAS,
            TOP_K=TOP_K,
            USE_LORA_EXPERT_MAP=USE_LORA_EXPERT_MAP,
            SHARED_OUTER=SHARED_OUTER,
        )
        valid = keys != -1
        w = tl.load(
            combine_weights_ptr + tokens64 * stride_cm + k * stride_ck,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        x = tl.load(
            bridge_ptr
            + pairs.to(tl.int64)[:, None] * stride_bm
            + offs_r[None, :] * stride_bk,
            mask=valid[:, None] & r_mask[None, :],
            other=0.0,
        )
        acc += w[:, None] * x.to(tl.float32)

    tl.store(
        tok_bridge_ptr + tokens64[:, None] * stride_tm + offs_r[None, :] * stride_tk,
        acc.to(tok_bridge_ptr.dtype.element_ty),
        mask=t_mask[:, None] & r_mask[None, :],
    )


def invoke_shared_rank_reduce(
    bridge: torch.Tensor,
    tok_bridge: torch.Tensor,
    combine_weights: torch.Tensor,
    routing: RouteView,
    *,
    config: Mapping[str, int],
) -> None:
    """FSHARED kernel A over the route SOURCES (shared-outer form required)."""
    num_tokens, top_k, _ = _validate_finalize_common(
        routing, combine_weights, tok_bridge, ownership="shared"
    )
    rank = tok_bridge.shape[1]
    if bridge.shape != (num_tokens * top_k, rank):
        raise ValueError(
            f"bridge must be {(num_tokens * top_k, rank)}, got "
            f"{tuple(bridge.shape)}"
        )
    if bridge.device != tok_bridge.device:
        raise ValueError(
            f"bridge device {bridge.device} != tok_bridge device "
            f"{tok_bridge.device}"
        )
    if routing.topk_ids.numel() == 0:
        # S5/6 verification: an empty route means the weighted reduction is
        # exactly zero. Returning WITHOUT writing left tok_bridge holding
        # whatever garbage the caller allocated, and kernel B then applied
        # B @ garbage to every token.
        tok_bridge.zero_()
        return
    map_arg, bound, use_map, shared = _route_key_args(routing)
    block_t = int(config["BLOCK_SIZE_T"])
    _shared_rank_reduce_kernel[(triton.cdiv(num_tokens, block_t),)](
        bridge,
        tok_bridge,
        combine_weights,
        routing.topk_ids,
        routing.token_slots,
        map_arg,
        num_tokens,
        bound,
        bridge.stride(0),
        bridge.stride(1),
        tok_bridge.stride(0),
        tok_bridge.stride(1),
        combine_weights.stride(0),
        combine_weights.stride(1),
        RANK=rank,
        TOP_K=top_k,
        LORA_EXPERTS_PER_ADAPTER=routing.lora_experts_per_adapter,
        MAX_LORAS=routing.max_loras,
        USE_LORA_EXPERT_MAP=use_map,
        SHARED_OUTER=shared,
        BLOCK_T=block_t,
        BLOCK_R=triton.next_power_of_2(rank),
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )


def build_shared_finalize_info(
    token_slots: torch.Tensor,
    *,
    max_loras: int,
    rank: int,
) -> LoRABatchInfo:
    """PREPARED-tier segment metadata for FSHARED's grouped GEMM.

    One unchunked segment per adapter run over the ADAPTER-SORTED valid
    tokens (``synthesize_unchunked_batch_info``): ``weight_indices`` carry
    the adapter slot indexing the shared ``[L, H, R]`` weights, and the
    permutation maps segment positions back to original token rows, so
    base tokens are never touched. Host-built and host-synced — build it
    OUTSIDE any timed region or graph capture; a charged device-side
    builder is the §64.12 adoption gate for this arm.
    """
    if token_slots.numel() and int(token_slots.max()) >= max_loras:
        raise ValueError(
            f"token slot {int(token_slots.max())} >= max_loras {max_loras}; "
            "an out-of-capacity slot would index the shared B weights out "
            "of bounds"
        )
    return synthesize_unchunked_batch_info(
        token_slots.to(torch.int32),
        max_loras=max_loras,
        physical_rank=rank,
        device=token_slots.device,
    )


def invoke_shared_finalize(
    tok_bridge: torch.Tensor,
    b_down: torch.Tensor,
    token_out: torch.Tensor,
    finalize_info: LoRABatchInfo,
    *,
    config: Mapping[str, int],
) -> None:
    """FSHARED kernel B: grouped-by-adapter GEMM ADDED into ``token_out``.

    Reuses the upstream unchunked ``_sgemm_lora_b_kernel`` over TOKEN
    segments sorted by adapter — its base-output read-modify-write gives
    the ``+=`` for free, each (segment row, N tile) has exactly one
    writing program, and the serial K loop keeps it deterministic.
    """
    if tok_bridge.dtype != b_down.dtype:
        raise ValueError(
            f"tok_bridge dtype {tok_bridge.dtype} != b_down dtype "
            f"{b_down.dtype}; the SGMV dot requires matching operands"
        )
    if finalize_info.permutation is None or finalize_info.seg_lens is None:
        raise ValueError(
            "finalize_info must carry adapter-sorted unchunked segments "
            "(permutation + seg_lens); build it with build_shared_finalize_info"
        )
    if finalize_info.bs == 0:
        return  # no valid tokens: the base destination IS the answer
    hidden, rank = b_down.shape[1], b_down.shape[2]
    grid = (
        triton.cdiv(finalize_info.max_len, int(config["BLOCK_S"]))
        * triton.cdiv(hidden, int(config["BLOCK_N"])),
        finalize_info.bs,
    )
    _sgemm_lora_b_kernel[grid](
        tok_bridge,
        b_down,
        token_out,
        hidden,
        rank,
        tok_bridge.stride(0),
        tok_bridge.stride(1),
        b_down.stride(0),
        b_down.stride(1),
        b_down.stride(2),
        token_out.stride(0),
        token_out.stride(1),
        finalize_info.seg_lens,
        finalize_info.seg_indptr,
        finalize_info.weight_indices,
        finalize_info.lora_ranks,
        finalize_info.permutation,
        True,  # sorted by adapter
        int(config["BLOCK_S"]),
        int(config["BLOCK_N"]),
        int(config["BLOCK_K"]),
        finalize_info.scalings,
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )


@triton.jit
def _materialized_combine_kernel(
    pair_delta_ptr,
    combine_weights_ptr,
    token_out_ptr,
    num_tokens,
    stride_pm,
    stride_ph,
    stride_cm,
    stride_ck,
    stride_om,
    stride_oh,
    HIDDEN: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """FM combine: ``token_out[t] += sum_k w[t, k] * pair_delta[t*K + k]``.

    The 2-op path's second op. Reads NO route: its invalid-pair guarantee
    is B's zero-fill contract (invalid delta rows are exact zero), which
    also requires finite combine weights on every pair — the production
    router invariant. FP32 accumulate, fixed k order, one writing program
    per (token block, H tile) cell.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    tokens = (pid_t * BLOCK_T + tl.arange(0, BLOCK_T)).to(tl.int64)
    t_mask = tokens < num_tokens
    offs_h = (pid_h * BLOCK_H + tl.arange(0, BLOCK_H)).to(tl.int64)
    h_mask = offs_h < HIDDEN
    cell_mask = t_mask[:, None] & h_mask[None, :]

    acc = tl.zeros((BLOCK_T, BLOCK_H), dtype=tl.float32)
    for k in range(TOP_K):
        w = tl.load(
            combine_weights_ptr + tokens * stride_cm + k * stride_ck,
            mask=t_mask,
            other=0.0,
        ).to(tl.float32)
        delta = tl.load(
            pair_delta_ptr
            + (tokens * TOP_K + k)[:, None] * stride_pm
            + offs_h[None, :] * stride_ph,
            mask=cell_mask,
            other=0.0,
        )
        acc += w[:, None] * delta.to(tl.float32)

    out_ptrs = token_out_ptr + tokens[:, None] * stride_om + offs_h[None, :] * stride_oh
    base = tl.load(out_ptrs, mask=cell_mask, other=0.0).to(tl.float32)
    tl.store(
        out_ptrs,
        (base + acc).to(token_out_ptr.dtype.element_ty),
        mask=cell_mask,
    )


def invoke_materialized_finalize(
    pair_delta: torch.Tensor,
    token_out: torch.Tensor,
    combine_weights: torch.Tensor,
    routing: RouteView,
    *,
    ownership: str,
    config: Mapping[str, int],
) -> None:
    num_tokens, top_k, hidden = _validate_finalize_common(
        routing, combine_weights, token_out, ownership=ownership
    )
    if pair_delta.shape != (num_tokens * top_k, hidden):
        raise ValueError(
            f"pair_delta must be {(num_tokens * top_k, hidden)}, got "
            f"{tuple(pair_delta.shape)}"
        )
    if pair_delta.device != token_out.device:
        raise ValueError(
            f"pair_delta device {pair_delta.device} != token_out device "
            f"{token_out.device}"
        )
    if routing.topk_ids.numel() == 0:
        return
    block_t = int(config["BLOCK_SIZE_T"])
    block_h = int(config["BLOCK_SIZE_H"])
    _materialized_combine_kernel[
        (triton.cdiv(num_tokens, block_t), triton.cdiv(hidden, block_h))
    ](
        pair_delta,
        combine_weights,
        token_out,
        num_tokens,
        pair_delta.stride(0),
        pair_delta.stride(1),
        combine_weights.stride(0),
        combine_weights.stride(1),
        token_out.stride(0),
        token_out.stride(1),
        HIDDEN=hidden,
        TOP_K=top_k,
        BLOCK_T=block_t,
        BLOCK_H=block_h,
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )


def _require_family_args(
    family: str,
    provided: Mapping[str, object],
    *,
    required: tuple[str, ...],
    optional: tuple[str, ...] = (),
) -> None:
    """Fail-closed argument coupling: nothing missing, nothing ignored."""
    for name in required:
        if provided[name] is None:
            raise ValueError(f"family {family!r} requires {name}")
    for name, value in provided.items():
        if name not in required and name not in optional and value is not None:
            raise ValueError(
                f"family {family!r} does not take {name}; a silently "
                "ignored input hides a mis-specified arm"
            )


def run_finalize(
    spec: FinalizeExecutionSpec,
    *,
    routing: RouteView,
    combine_weights: torch.Tensor,
    token_out: torch.Tensor,
    config: Mapping[str, int],
    bridge: torch.Tensor | None = None,
    b_down: torch.Tensor | None = None,
    pair_delta: torch.Tensor | None = None,
    tok_bridge: torch.Tensor | None = None,
    finalize_info: LoRABatchInfo | None = None,
) -> None:
    """Execute one finalizer candidate FROM its spec — the spec IS the dispatch."""
    provided = {
        "bridge": bridge,
        "b_down": b_down,
        "pair_delta": pair_delta,
        "tok_bridge": tok_bridge,
        "finalize_info": finalize_info,
    }
    if spec.family == "materialized":
        _require_family_args(spec.family, provided, required=("pair_delta",))
        invoke_materialized_finalize(
            pair_delta,
            token_out,
            combine_weights,
            routing,
            ownership=spec.ownership,
            config=config,
        )
    elif spec.family == "token_owned":
        _require_family_args(spec.family, provided, required=("bridge", "b_down"))
        invoke_token_finalize(
            bridge,
            b_down,
            token_out,
            combine_weights,
            routing,
            ownership=spec.ownership,
            config=config,
        )
    elif spec.family == "shared_rank_reduce":
        _require_family_args(
            spec.family,
            provided,
            required=("bridge", "b_down", "finalize_info"),
            optional=("tok_bridge",),
        )
        # The GEMM writes token_out THROUGH the permutation, so the
        # destination contract must hold before either launch (the reduce
        # only sees tok_bridge).
        _validate_finalize_common(
            routing, combine_weights, token_out, ownership=spec.ownership
        )
        _validate_weight_gemm_inputs(
            bridge,
            b_down,
            routing,
            ownership=spec.ownership,
            hidden=token_out.shape[1],
        )
        if tok_bridge is None:
            tok_bridge = torch.empty(
                (routing.topk_ids.shape[0], b_down.shape[2]),
                dtype=b_down.dtype,
                device=token_out.device,
            )
        invoke_shared_rank_reduce(
            bridge,
            tok_bridge,
            combine_weights,
            routing,
            config=config["reduce"],
        )
        invoke_shared_finalize(
            tok_bridge,
            b_down,
            token_out,
            finalize_info,
            config=config["gemm"],
        )
    else:
        raise NotImplementedError(f"no executor for {spec.key()!r}")
