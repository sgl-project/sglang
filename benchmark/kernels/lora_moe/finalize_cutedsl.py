"""CuTeDSL Step-6 finalize arms (plan §65.2, the mandatory CUTE obligation).

Two arms, both compatible with the Triton finalize inputs — ``bridge_down``
[P, R] pair-major down-A output, ``b_down`` grouped B factors, the route
sources (``topk_ids`` [T, K] / ``token_slots`` [T]), ``combine_weights``
[T, K] fp32, and a NONZERO-base ``token_out`` [T, H] destination:

* **cutedsl_shared_finalize** (FSHARED-CUTE): the algebraic top_k-fold FLOP
  cut for SHARED B.  A small Triton kernel folds the combine weights in
  RANK space — ``reduced[t] = sum_k w[t,k] * bridge[t*K+k]`` over valid
  pairs ([P, R] -> [T, R], memory-trivial) — then ONE masked grouped
  CuTeDSL GEMM grouped BY ADAPTER over token rows ([T, R] x [R, H]) and a
  scatter-ADD into ``token_out``.  The S2 masked GEMM writes C and cannot
  accumulate into it, so the delta is added FROM the staged C — the
  pair-domain [P, H] delta never exists.
* **cutedsl_token_finalize** (FTOK-CUTE): per-expert B.  Pairs are staged
  by VIRTUAL expert through the P5 dispatch machinery
  (``fused_moe_dispatch_index`` over the fused ids), the masked grouped
  GEMM runs [P_staged, R] x [R, H] per group, and ONE weighted scatter-add
  applies ``combine_weights`` EXACTLY ONCE while adding into the nonzero
  base — a serial-k loop in a single program per (token, H-tile), so the
  reduction order is fixed per token.

Exactly-once weighting: the shared arm folds the weights in the rank
reduce (BEFORE its unweighted GEMM); the token arm applies them in the
scatter (AFTER its unweighted GEMM).  Neither path touches them twice, and
the GEMMs themselves are weight-free.

Prepared-metadata boundary: the same split as ``lora_a_cutedsl`` — the
dispatch, the inverse maps, and the packed tile schedules are built at plan
construction (``build_metadata`` rebuilds them for the SAME route, verified
fail-closed), ``m_max`` is host-read at build only, and a timed thunk
re-executes exactly ``plan.run(...)``: reduce/stage + GEMM + scatter.

STATUS (S5/6 review): these are CONTROLS, not yet optimized candidates.
The staging step is ``index_select`` FOLLOWED BY ``index_copy_`` — two
operations with a temporary between them, so the real launch count is one
higher than "reduce + GEMM + scatter" suggests. Fusing the gather and the
dispatched-row store into one kernel is required before a CuTeDSL arm may
be fairly REJECTED; until then a CuTeDSL loss is not evidence against the
CuTeDSL approach.

Determinism: the dispatch atomics order rows WITHIN a group
nondeterministically, but each C row is an independent K-reduction whose
value does not depend on its m-position, and both scatters invert the maps
exactly — so ``token_out`` values are placement-invariant, and every
element has exactly ONE writer program with a fixed serial-k order.  At
the prepared boundary (frozen metadata) replays are bitwise, including
under CUDA-graph capture.

Destination contract: ``token_out`` may be BF16 or caller-selected FP32;
the read-modify-write accumulates in FP32 either way.  Tokens without a
single valid pair receive an exact ``+= 0`` (token arm) or are skipped
entirely (shared arm), so their base rows survive bitwise.

Compile discipline is shared with ``lora_a_cutedsl``: one compile per
(device, config, groups, N, K) through the provider's process-global
``_COMPILE_CACHE``, zero-tile warmup at compile time.
"""

from __future__ import annotations

from typing import Any

import msgspec
import torch
import triton
import triton.language as tl

from benchmark.kernels.lora_moe.lora_a_cutedsl import (
    CutedslAConfig,
    _compiled_masked_gemm,
)
from sglang.kernels.ops.moe.ep_moe_kernels import fused_moe_dispatch_index
from sglang.srt.lora.sgl_lora.routing import RouteView

# The reduce and the scatters are memory-trivial next to the GEMM; fixed
# launch shapes suffice (the sweepable geometry is the GEMM's, through
# CutedslAConfig — deliberately reused: it is masked-GEMM geometry, not
# A-site identity).
REDUCE_NUM_WARPS = 2
SCATTER_BLOCK_H = 256
SCATTER_NUM_WARPS = 4
TOKEN_OUT_DTYPES = (torch.bfloat16, torch.float32)


@triton.jit
def _weighted_rank_reduce_kernel(
    bridge_ptr,
    combine_weights_ptr,
    virtual_topk_ids_ptr,
    reduced_ptr,
    stride_bm,
    stride_bk,
    stride_wt,
    stride_wk,
    stride_rm,
    stride_rk,
    TOP_K: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """``reduced[t] = sum_k w[t,k] * bridge[t*K+k]`` over VALID pairs.

    One program per token, serial k — deterministic by construction.
    Invalid pairs (fused id ``-1``) mask BOTH the weight and the bridge
    LOAD: sentinel bridge rows are contractually undefined (the stock-B
    convention), so multiplying them by a zero weight is not enough — a
    NaN-poisoned row times zero is still NaN.
    """
    token = tl.program_id(0).to(tl.int64)
    offs_r = tl.arange(0, BLOCK_R).to(tl.int64)
    r_mask = offs_r < RANK
    accumulator = tl.zeros((BLOCK_R,), dtype=tl.float32)
    for k in range(TOP_K):
        pair = token * TOP_K + k
        virtual_id = tl.load(virtual_topk_ids_ptr + pair)
        valid = virtual_id >= 0
        weight = tl.load(combine_weights_ptr + token * stride_wt + k * stride_wk)
        weight = tl.where(valid, weight.to(tl.float32), 0.0)
        row = tl.load(
            bridge_ptr + pair * stride_bm + offs_r * stride_bk,
            mask=r_mask & valid,
            other=0.0,
        )
        accumulator += weight * row.to(tl.float32)
    tl.store(
        reduced_ptr + token * stride_rm + offs_r * stride_rk,
        accumulator.to(reduced_ptr.dtype.element_ty),
        mask=r_mask,
    )


@triton.jit
def _shared_scatter_add_kernel(
    c_ptr,
    token_out_ptr,
    valid_token_ids_ptr,
    source_rows_ptr,
    stride_cm,
    stride_cn,
    stride_om,
    stride_on,
    hidden,
    BLOCK_H: tl.constexpr,
):
    """``token_out[token] += C_flat[source_row]`` for each dispatched token.

    One program per (valid token, H-tile); every destination cell has
    exactly one writer, and the weights were already folded in rank space,
    so this is a pure FP32 read-modify-write into the nonzero base.
    """
    index = tl.program_id(0)
    tile = tl.program_id(1).to(tl.int64)
    token = tl.load(valid_token_ids_ptr + index).to(tl.int64)
    source_row = tl.load(source_rows_ptr + index).to(tl.int64)
    offs_h = tile * BLOCK_H + tl.arange(0, BLOCK_H).to(tl.int64)
    h_mask = offs_h < hidden
    delta = tl.load(
        c_ptr + source_row * stride_cm + offs_h * stride_cn,
        mask=h_mask,
        other=0.0,
    ).to(tl.float32)
    out_ptrs = token_out_ptr + token * stride_om + offs_h * stride_on
    base = tl.load(out_ptrs, mask=h_mask, other=0.0).to(tl.float32)
    tl.store(
        out_ptrs,
        (base + delta).to(token_out_ptr.dtype.element_ty),
        mask=h_mask,
    )


@triton.jit
def _token_weighted_scatter_add_kernel(
    c_ptr,
    combine_weights_ptr,
    src2dst_safe_ptr,
    token_out_ptr,
    stride_cm,
    stride_cn,
    stride_wt,
    stride_wk,
    stride_om,
    stride_on,
    hidden,
    TOP_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """``token_out[t] += sum_k w[t,k] * C_flat[src2dst_safe[t*K+k]]``.

    One program per (token, H-tile) with a SERIAL fixed-order k loop: the
    routed weight is applied exactly once per pair, the destination cell
    has exactly one writer, and ``src2dst_safe`` carries ``-1`` for
    invalid pairs (their staged row was never written), masking both the
    weight and the C load.  Tokens with no valid pair fall through with a
    zero accumulator: an exact ``+= 0`` that preserves the base bitwise.
    """
    token = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    offs_h = tile * BLOCK_H + tl.arange(0, BLOCK_H).to(tl.int64)
    h_mask = offs_h < hidden
    accumulator = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for k in range(TOP_K):
        source_row = tl.load(src2dst_safe_ptr + token * TOP_K + k)
        valid = source_row >= 0
        weight = tl.load(combine_weights_ptr + token * stride_wt + k * stride_wk)
        weight = tl.where(valid, weight.to(tl.float32), 0.0)
        row = tl.load(
            c_ptr
            + tl.maximum(source_row, 0).to(tl.int64) * stride_cm
            + offs_h * stride_cn,
            mask=h_mask & valid,
            other=0.0,
        )
        accumulator += weight * row.to(tl.float32)
    out_ptrs = token_out_ptr + token * stride_om + offs_h * stride_on
    base = tl.load(out_ptrs, mask=h_mask, other=0.0).to(tl.float32)
    tl.store(
        out_ptrs,
        (base + accumulator).to(token_out_ptr.dtype.element_ty),
        mask=h_mask,
    )


def _validate_finalize_build(
    *,
    route: RouteView,
    weight: torch.Tensor,
    adapter_grouped: bool,
) -> tuple[int, int, int, int, int]:
    """Fail-closed geometry contract shared by both builders.

    Returns ``(num_groups, hidden, rank, num_tokens, top_k)``.
    """
    from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_masked.schedule_builder import (  # noqa: E501
        MAX_EXPERTS,
    )

    num_groups = route.num_virtual_experts
    if adapter_grouped and route.lora_experts_per_adapter != 1:
        raise ValueError(
            "the shared finalize groups token rows BY ADAPTER, which is only "
            "meaningful for the one-factor (shared-outer) route form; got "
            f"lora_experts_per_adapter={route.lora_experts_per_adapter}"
        )
    if weight.ndim != 3 or weight.shape[0] != num_groups:
        raise ValueError(
            f"b_down must be [num_virtual_experts={num_groups}, H, R]; got "
            f"{tuple(weight.shape)}"
        )
    if weight.dtype != torch.bfloat16:
        raise ValueError(f"b_down must be bf16, got {weight.dtype}")
    if not weight.is_contiguous():
        raise ValueError(
            "b_down must be contiguous — a silent .contiguous() copy here "
            "would wrap a temporary whose Torch owner dies"
        )
    hidden, rank = weight.shape[1], weight.shape[2]
    if hidden % 8 or rank % 8:
        raise ValueError(
            f"H={hidden} and R={rank} must be multiples of 8 (the masked "
            "GEMM's 16-byte TMA alignment)"
        )
    if num_groups > MAX_EXPERTS:
        raise ValueError(
            f"{num_groups} groups exceed the direct schedule's "
            f"{MAX_EXPERTS}-group packing"
        )
    num_tokens, top_k = route.topk_ids.shape
    if num_tokens == 0:
        raise ValueError("an empty batch has no finalize work; do not plan it")
    return num_groups, hidden, rank, num_tokens, top_k


def _check_run_io(
    plan,
    bridge: torch.Tensor,
    combine_weights: torch.Tensor,
    token_out: torch.Tensor,
) -> None:
    """Per-call IO contract (host-cheap: tuple compares only)."""
    num_pairs = plan.num_tokens * plan.top_k
    if bridge.shape != (num_pairs, plan.rank) or bridge.dtype != torch.bfloat16:
        raise ValueError(
            f"bridge_down must be bf16 [{num_pairs}, {plan.rank}]; got "
            f"{tuple(bridge.shape)} {bridge.dtype}"
        )
    if combine_weights.shape != (plan.num_tokens, plan.top_k):
        raise ValueError(
            f"combine_weights must be [{plan.num_tokens}, {plan.top_k}]; got "
            f"{tuple(combine_weights.shape)}"
        )
    if combine_weights.dtype != torch.float32:
        raise ValueError(
            "combine_weights must be float32 (the production topk_weights "
            f"dtype); a {combine_weights.dtype} copy silently degrades the "
            "exactly-once routed weighting"
        )
    if token_out.shape != (plan.num_tokens, plan.hidden):
        raise ValueError(
            f"token_out must be [{plan.num_tokens}, {plan.hidden}]; got "
            f"{tuple(token_out.shape)}"
        )
    if token_out.dtype not in TOKEN_OUT_DTYPES:
        raise ValueError(
            "token_out must be bf16 or the caller-selected fp32; got "
            f"{token_out.dtype}"
        )
    devices = {bridge.device, combine_weights.device, token_out.device}
    if len(devices) != 1 or bridge.device != plan.masked_m.device:
        raise ValueError("run tensors must all live on the plan's device")


def _require_plan_binding(plan, weight: torch.Tensor, routing) -> None:
    """A declared invocation must be THIS plan's fixture (pointer-level).

    Same scope as the A plan's binding (tenth S3 review): identity is
    addresses + shapes + strides; in-place content mutation is caught by
    ``build_metadata(verify=True)``, not here.
    """
    if (
        plan.weight_owner is None
        or plan.weight_owner.data_ptr() != weight.data_ptr()
        or plan.weight_owner.shape != weight.shape
        or plan.weight_owner.stride() != weight.stride()
    ):
        raise ValueError(
            "finalize plan's b_down is not the tensor this invocation "
            "declares (address, shape, and strides must all match) — plans "
            "bind per fixture"
        )
    if routing is None:
        raise ValueError(
            "a declared cutedsl finalize invocation must supply its routing "
            "view — binding cannot be skipped"
        )
    if (
        routing.topk_ids.data_ptr() != plan.route_topk_ptr
        or routing.token_slots.data_ptr() != plan.route_slots_ptr
        or routing.topk_ids.shape != plan.virtual_topk_ids.shape
    ):
        raise ValueError(
            "finalize plan was dispatched from a different route than the "
            "declared routing view (source tensor mismatch)"
        )
    if routing.num_virtual_experts != plan.num_groups:
        raise ValueError(
            f"the declared routing view's group domain "
            f"({routing.num_virtual_experts}) is not this plan's dispatch "
            f"domain ({plan.num_groups}) — same source tensors, different "
            "factor semantics"
        )


def _bind_weight_and_route(plan, *, weight: torch.Tensor, route: RouteView) -> None:
    from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_masked.api import (
        as_dynamic_cute_tensor,
    )

    plan.b_arg = as_dynamic_cute_tensor(weight, leading_dim=2)
    plan.weight_owner = weight
    plan.route_topk_ptr = route.topk_ids.data_ptr()
    plan.route_slots_ptr = route.token_slots.data_ptr()


def _single_stage_schedule(
    *,
    masked_m: torch.Tensor,
    m_max: int,
    config: CutedslAConfig,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One packed schedule off ``masked_m``.

    The builder is dual by design (the provider's two chained stages); the
    finalize arms have a single GEMM, so stage 2 is a minimal one-cluster
    placeholder whose buffers are discarded — a few hundred int32 of wasted
    writes, outside every timed thunk.
    """
    from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_masked.schedule_builder import (  # noqa: E501
        build_dual_stage_schedules,
    )

    schedule, tiles, _, _ = build_dual_stage_schedules(
        masked_m,
        m_max=m_max,
        token_width=config.token_width,
        n_gemm1=n,
        n_gemm2=config.output_width,
        output_width=config.output_width,
    )
    return schedule, tiles


def _launch_masked_gemm(plan) -> None:
    import cuda.bindings.driver as cuda_driver

    from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_masked.api import (
        as_dynamic_cute_tensor as dyn,
    )

    stream = cuda_driver.CUstream(
        torch.cuda.current_stream(plan.staged.device).cuda_stream
    )
    plan.compiled(
        dyn(plan.staged, leading_dim=2),
        plan.b_arg,
        dyn(plan.c, leading_dim=2),
        dyn(plan.masked_m, leading_dim=0),
        dyn(plan.schedule, leading_dim=0),
        dyn(plan.tiles, leading_dim=0),
        stream,
    )


def _sized_m_max(widest: int, token_width: int) -> int:
    return max(
        (widest + token_width - 1) // token_width * token_width,
        token_width,
    )


def _require_same_route(
    *,
    fresh_masked_m: torch.Tensor,
    plan_masked_m: torch.Tensor,
    fresh_valid: torch.Tensor,
    plan_valid: torch.Tensor,
    m_max: int,
) -> None:
    """The verify half of ``build_metadata`` — identical wording to the A
    plan so drivers treat both plan families uniformly."""
    if int(fresh_masked_m.max().item()) > m_max:
        raise ValueError(
            f"route contents changed: widest group "
            f"{int(fresh_masked_m.max().item())} exceeds the sized m_max="
            f"{m_max}; rebuild the plan for the new route"
        )
    if not torch.equal(fresh_masked_m, plan_masked_m) or not torch.equal(
        fresh_valid, plan_valid
    ):
        raise ValueError(
            "route contents changed since plan build (group counts or the "
            "valid set differ) — this plan is fixture-only; rebuild it for "
            "the new route"
        )


class CutedslSharedFinalizePlan(msgspec.Struct, kw_only=True):
    """FSHARED-CUTE prepared state: rank reduce -> by-adapter GEMM -> add.

    Token rows (not pairs) are the dispatch domain: ``token_group_ids[t]``
    is the token's adapter slot when the token has at least one valid pair
    (``amax`` over its fused ids — every valid key of a token equals its
    slot under the one-factor route) and ``-1`` otherwise, so undispatched
    tokens are skipped entirely and their base rows survive bitwise.
    """

    config: CutedslAConfig
    num_groups: int  # adapter capacity (max_loras)
    m_max: int
    top_k: int
    rank: int
    hidden: int
    num_tokens: int
    virtual_topk_ids: torch.Tensor  # [T, K] int32, -1 sentinel (stored copy)
    token_group_ids: torch.Tensor  # [T] int32, -1 = token undispatched
    valid_token_ids: torch.Tensor  # [T_valid] int64
    masked_m: torch.Tensor  # [L] int32
    src2dst: torch.Tensor  # [T] int32 (valid positions only)
    src2dst_valid: torch.Tensor  # [T_valid] int64 staging/scatter rows
    reduced: torch.Tensor  # [T, R] bf16 rank-space weighted reduce
    staged: torch.Tensor  # [L, m_max, R] bf16
    c: torch.Tensor  # [L, m_max, H] bf16
    schedule: torch.Tensor
    tiles: torch.Tensor
    compiled: Any
    b_arg: Any = None
    weight_owner: torch.Tensor | None = None
    route_topk_ptr: int = 0
    route_slots_ptr: int = 0

    def require_binding(self, weight: torch.Tensor, routing) -> None:
        _require_plan_binding(self, weight, routing)

    def build_metadata(self, verify: bool = True) -> None:
        """Re-run dispatch + schedule for the SAME route contents.

        The token grouping is re-derived from the stored fused ids, so any
        content change that alters TOKEN-level grouping (a token gaining or
        losing its last valid pair, or moving adapter) is refused here.
        Scope: a pair-level sentinel flip that leaves every token's group
        unchanged is legitimately tolerated — the rank reduce reads the
        live fused ids each run, and the dispatch never saw pairs.
        ``verify=False`` is only legal after a verified call on the
        identical route (the dispatch counts are deterministic, one
        verification covers every replay).
        """
        fresh_groups = self.virtual_topk_ids.amax(dim=1).contiguous()
        masked_m, src2dst = fused_moe_dispatch_index(
            fresh_groups, self.num_groups, self.m_max
        )
        if verify:
            _require_same_route(
                fresh_masked_m=masked_m,
                plan_masked_m=self.masked_m,
                fresh_valid=(fresh_groups >= 0).nonzero(as_tuple=False).view(-1),
                plan_valid=self.valid_token_ids,
                m_max=self.m_max,
            )
        self.token_group_ids.copy_(fresh_groups)
        self.masked_m.copy_(masked_m)
        self.src2dst.copy_(src2dst)
        # The dispatch atomics assign a FRESH within-group order every
        # rebuild; the inverse map must be refreshed with it.
        self.src2dst_valid.copy_(
            src2dst.index_select(0, self.valid_token_ids).to(torch.int64)
        )
        schedule, tiles = _single_stage_schedule(
            masked_m=self.masked_m,
            m_max=self.m_max,
            config=self.config,
            n=self.hidden,
        )
        self.schedule.copy_(schedule)
        self.tiles.copy_(tiles)

    def run(
        self,
        *,
        bridge: torch.Tensor,
        combine_weights: torch.Tensor,
        token_out: torch.Tensor,
    ) -> None:
        """Prepared-boundary thunk: reduce + stage + GEMM + scatter-add."""
        _check_run_io(self, bridge, combine_weights, token_out)
        _weighted_rank_reduce_kernel[(self.num_tokens,)](
            bridge,
            combine_weights,
            self.virtual_topk_ids,
            self.reduced,
            bridge.stride(0),
            bridge.stride(1),
            combine_weights.stride(0),
            combine_weights.stride(1),
            self.reduced.stride(0),
            self.reduced.stride(1),
            TOP_K=self.top_k,
            RANK=self.rank,
            BLOCK_R=triton.next_power_of_2(self.rank),
            num_warps=REDUCE_NUM_WARPS,
        )
        self.staged.view(-1, self.rank).index_copy_(
            0,
            self.src2dst_valid,
            self.reduced.index_select(0, self.valid_token_ids),
        )
        _launch_masked_gemm(self)
        if self.valid_token_ids.numel() == 0:
            return  # zero-tile GEMM already ran; nothing owns any base row
        _shared_scatter_add_kernel[
            (
                self.valid_token_ids.numel(),
                triton.cdiv(self.hidden, SCATTER_BLOCK_H),
            )
        ](
            self.c,
            token_out,
            self.valid_token_ids,
            self.src2dst_valid,
            self.c.stride(1),
            self.c.stride(2),
            token_out.stride(0),
            token_out.stride(1),
            self.hidden,
            BLOCK_H=SCATTER_BLOCK_H,
            num_warps=SCATTER_NUM_WARPS,
        )

    def gemm_only(self) -> None:
        """Diagnostic ideal bound: the GEMM alone over prepared staging."""
        _launch_masked_gemm(self)


class CutedslTokenFinalizePlan(msgspec.Struct, kw_only=True):
    """FTOK-CUTE prepared state: by-virtual-expert GEMM -> weighted add.

    Pairs are the dispatch domain (the P5 machinery verbatim);
    ``src2dst_safe`` re-encodes the dispatch map with a ``-1`` sentinel at
    invalid pairs so the weighted scatter never reads an unwritten C row.
    """

    config: CutedslAConfig
    num_groups: int  # V = max_loras * lora_experts_per_adapter
    m_max: int
    top_k: int
    rank: int
    hidden: int
    num_tokens: int
    virtual_topk_ids: torch.Tensor  # [T, K] int32, -1 sentinel (stored copy)
    valid_pair_ids: torch.Tensor  # [P_valid] int64
    masked_m: torch.Tensor  # [V] int32
    src2dst: torch.Tensor  # [T*K] int32 (valid positions only)
    src2dst_valid: torch.Tensor  # [P_valid] int64 staging rows
    src2dst_safe: torch.Tensor  # [T*K] int32, -1 at invalid pairs
    staged: torch.Tensor  # [V, m_max, R] bf16
    c: torch.Tensor  # [V, m_max, H] bf16
    schedule: torch.Tensor
    tiles: torch.Tensor
    compiled: Any
    b_arg: Any = None
    weight_owner: torch.Tensor | None = None
    route_topk_ptr: int = 0
    route_slots_ptr: int = 0

    def require_binding(self, weight: torch.Tensor, routing) -> None:
        _require_plan_binding(self, weight, routing)

    def build_metadata(self, verify: bool = True) -> None:
        """Re-run dispatch + schedule for the SAME route contents."""
        masked_m, src2dst = fused_moe_dispatch_index(
            self.virtual_topk_ids, self.num_groups, self.m_max
        )
        if verify:
            _require_same_route(
                fresh_masked_m=masked_m,
                plan_masked_m=self.masked_m,
                fresh_valid=(
                    (self.virtual_topk_ids.view(-1) >= 0)
                    .nonzero(as_tuple=False)
                    .view(-1)
                ),
                plan_valid=self.valid_pair_ids,
                m_max=self.m_max,
            )
        self.masked_m.copy_(masked_m)
        self.src2dst.copy_(src2dst)
        valid_rows = src2dst.index_select(0, self.valid_pair_ids)
        self.src2dst_valid.copy_(valid_rows.to(torch.int64))
        self.src2dst_safe.fill_(-1)
        self.src2dst_safe.index_copy_(0, self.valid_pair_ids, valid_rows)
        schedule, tiles = _single_stage_schedule(
            masked_m=self.masked_m,
            m_max=self.m_max,
            config=self.config,
            n=self.hidden,
        )
        self.schedule.copy_(schedule)
        self.tiles.copy_(tiles)

    def run(
        self,
        *,
        bridge: torch.Tensor,
        combine_weights: torch.Tensor,
        token_out: torch.Tensor,
    ) -> None:
        """Prepared-boundary thunk: stage + GEMM + weighted scatter-add."""
        _check_run_io(self, bridge, combine_weights, token_out)
        self.staged.view(-1, self.rank).index_copy_(
            0,
            self.src2dst_valid,
            bridge.index_select(0, self.valid_pair_ids),
        )
        _launch_masked_gemm(self)
        _token_weighted_scatter_add_kernel[
            (self.num_tokens, triton.cdiv(self.hidden, SCATTER_BLOCK_H))
        ](
            self.c,
            combine_weights,
            self.src2dst_safe,
            token_out,
            self.c.stride(1),
            self.c.stride(2),
            combine_weights.stride(0),
            combine_weights.stride(1),
            token_out.stride(0),
            token_out.stride(1),
            self.hidden,
            TOP_K=self.top_k,
            BLOCK_H=SCATTER_BLOCK_H,
            num_warps=SCATTER_NUM_WARPS,
        )

    def gemm_only(self) -> None:
        """Diagnostic ideal bound: the GEMM alone over prepared staging."""
        _launch_masked_gemm(self)


def build_cutedsl_shared_finalize_plan(
    *,
    shared_route: RouteView,
    down_weight: torch.Tensor,
    config: CutedslAConfig,
) -> CutedslSharedFinalizePlan:
    """Size, compile, and pre-build the metadata for one shared-B fixture.

    ``shared_route`` must be the ROUTE_FUSED_IDS view of the ONE-FACTOR
    (shared-outer) route form: ``lora_experts_per_adapter=1`` with the
    local expert count carried as the validity bound, so its fused ids are
    exactly ``{adapter slot, -1}`` and ``num_virtual_experts`` is the
    adapter capacity — the GEMM grouping this arm exists for.
    ``down_weight`` is the per-adapter shared B, ``[max_loras, H, R]``.
    """
    num_groups, hidden, rank, num_tokens, top_k = _validate_finalize_build(
        route=shared_route, weight=down_weight, adapter_grouped=True
    )
    virtual_topk_ids = shared_route.virtual_topk_ids.to(torch.int32).contiguous()
    device = virtual_topk_ids.device
    token_group_ids = virtual_topk_ids.amax(dim=1).contiguous()
    masked_probe, _ = fused_moe_dispatch_index(token_group_ids, num_groups, 1)
    m_max = _sized_m_max(int(masked_probe.max().item()), config.token_width)
    valid_token_ids = (token_group_ids >= 0).nonzero(as_tuple=False).view(-1)
    plan = CutedslSharedFinalizePlan(
        config=config,
        num_groups=num_groups,
        m_max=m_max,
        top_k=top_k,
        rank=rank,
        hidden=hidden,
        num_tokens=num_tokens,
        virtual_topk_ids=virtual_topk_ids,
        token_group_ids=token_group_ids,
        valid_token_ids=valid_token_ids,
        masked_m=torch.empty(num_groups, dtype=torch.int32, device=device),
        src2dst=torch.empty(num_tokens, dtype=torch.int32, device=device),
        src2dst_valid=torch.empty(
            valid_token_ids.numel(), dtype=torch.int64, device=device
        ),
        reduced=torch.empty((num_tokens, rank), dtype=torch.bfloat16, device=device),
        staged=torch.zeros(
            (num_groups, m_max, rank), dtype=torch.bfloat16, device=device
        ),
        c=torch.empty((num_groups, m_max, hidden), dtype=torch.bfloat16, device=device),
        schedule=torch.empty(0, dtype=torch.int32, device=device),
        tiles=torch.empty(1, dtype=torch.int32, device=device),
        compiled=_compiled_masked_gemm(
            device=device, config=config, num_groups=num_groups, n=hidden, k=rank
        ),
    )
    _bind_weight_and_route(plan, weight=down_weight, route=shared_route)
    masked_m, src2dst = fused_moe_dispatch_index(token_group_ids, num_groups, m_max)
    plan.masked_m.copy_(masked_m)
    plan.src2dst.copy_(src2dst)
    plan.src2dst_valid.copy_(src2dst.index_select(0, valid_token_ids).to(torch.int64))
    plan.schedule, plan.tiles = _single_stage_schedule(
        masked_m=plan.masked_m, m_max=m_max, config=config, n=hidden
    )
    return plan


def build_cutedsl_token_finalize_plan(
    *,
    fused_route: RouteView,
    down_weight: torch.Tensor,
    config: CutedslAConfig,
) -> CutedslTokenFinalizePlan:
    """Size, compile, and pre-build the metadata for one per-expert fixture.

    ``fused_route`` must be the ROUTE_FUSED_IDS view of the SAME case route
    the Triton finalize arms consume — one id domain, one grouping.
    ``down_weight`` is the flattened per-virtual-expert B, ``[V, H, R]``.
    """
    num_groups, hidden, rank, num_tokens, top_k = _validate_finalize_build(
        route=fused_route, weight=down_weight, adapter_grouped=False
    )
    virtual_topk_ids = fused_route.virtual_topk_ids.to(torch.int32).contiguous()
    device = virtual_topk_ids.device
    masked_probe, _ = fused_moe_dispatch_index(virtual_topk_ids, num_groups, 1)
    m_max = _sized_m_max(int(masked_probe.max().item()), config.token_width)
    flat = virtual_topk_ids.view(-1)
    valid_pair_ids = (flat >= 0).nonzero(as_tuple=False).view(-1)
    plan = CutedslTokenFinalizePlan(
        config=config,
        num_groups=num_groups,
        m_max=m_max,
        top_k=top_k,
        rank=rank,
        hidden=hidden,
        num_tokens=num_tokens,
        virtual_topk_ids=virtual_topk_ids,
        valid_pair_ids=valid_pair_ids,
        masked_m=torch.empty(num_groups, dtype=torch.int32, device=device),
        src2dst=torch.empty(flat.numel(), dtype=torch.int32, device=device),
        src2dst_valid=torch.empty(
            valid_pair_ids.numel(), dtype=torch.int64, device=device
        ),
        src2dst_safe=torch.empty(flat.numel(), dtype=torch.int32, device=device),
        staged=torch.zeros(
            (num_groups, m_max, rank), dtype=torch.bfloat16, device=device
        ),
        c=torch.empty((num_groups, m_max, hidden), dtype=torch.bfloat16, device=device),
        schedule=torch.empty(0, dtype=torch.int32, device=device),
        tiles=torch.empty(1, dtype=torch.int32, device=device),
        compiled=_compiled_masked_gemm(
            device=device, config=config, num_groups=num_groups, n=hidden, k=rank
        ),
    )
    _bind_weight_and_route(plan, weight=down_weight, route=fused_route)
    masked_m, src2dst = fused_moe_dispatch_index(virtual_topk_ids, num_groups, m_max)
    plan.masked_m.copy_(masked_m)
    plan.src2dst.copy_(src2dst)
    valid_rows = src2dst.index_select(0, valid_pair_ids)
    plan.src2dst_valid.copy_(valid_rows.to(torch.int64))
    plan.src2dst_safe.fill_(-1)
    plan.src2dst_safe.index_copy_(0, valid_pair_ids, valid_rows)
    plan.schedule, plan.tiles = _single_stage_schedule(
        masked_m=plan.masked_m, m_max=m_max, config=config, n=hidden
    )
    return plan


def cutedsl_shared_finalize(
    *,
    plan: CutedslSharedFinalizePlan,
    bridge: torch.Tensor,
    combine_weights: torch.Tensor,
    token_out: torch.Tensor,
    weight: torch.Tensor,
    routing: RouteView,
) -> None:
    """The declared FSHARED-CUTE invocation: bind, then run the thunk body."""
    plan.require_binding(weight, routing)
    plan.run(bridge=bridge, combine_weights=combine_weights, token_out=token_out)


def cutedsl_token_finalize(
    *,
    plan: CutedslTokenFinalizePlan,
    bridge: torch.Tensor,
    combine_weights: torch.Tensor,
    token_out: torch.Tensor,
    weight: torch.Tensor,
    routing: RouteView,
) -> None:
    """The declared FTOK-CUTE invocation: bind, then run the thunk body."""
    plan.require_binding(weight, routing)
    plan.run(bridge=bridge, combine_weights=combine_weights, token_out=token_out)
