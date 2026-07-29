"""CuTeDSL fused-middle arm (Step 5 CUTE, plan §65.1).

STATUS (S5/6 review): a staged composite CONTROL, not yet an optimized
candidate — the custom single-kernel fusion is explicitly out of scope
below. A loss recorded against it rejects THIS composite, never the
CuTeDSL approach; the arm may not be eliminated on these numbers.

The fused middle is gate/up-B -> activation join -> down-A. This arm runs
both GEMMs on the S2 masked grouped tensor-core kernel (the P5 composite
family, ``lora_a_cutedsl.py``), keeping every intermediate in the STAGED
row domain — the pair-domain ``gate_up_delta`` ``[P, 2W]`` buffer of the
materialized baseline is never allocated and never scattered:

1. **stage**: one Triton gather places the bridge ``[P, 2R]`` into
   ``[S*V, m_max, R]`` — SLICE-INTERLEAVED groups ``(2v, 2v+1)`` so the
   gate and up B-slices become ONE masked grouped GEMM over ``S*V``
   groups whose weight is the pure contiguous view
   ``b_gate_up.view(S*V, W, R)`` (a ``[V, 2W, R]`` tensor reshapes to
   slice-interleaved ``[2V, W, R]`` with zero data movement — the only
   arrangement that keeps the provider's contiguous ``[E, N, K]`` ABI);
2. **GEMM1**: the compiled masked grouped kernel writes the gate/up
   delta ``[S*V, m_max, W]`` in the staged domain;
3. **join**: one row-local Triton kernel adds the pair-domain base
   ``[P, S*W]``, applies the activation (SwiGLU or non-gated ReLU^2,
   compile-time ``NUM_SLICES``), and dual-stores the activated value:
   ``staged_act[dst]`` (GEMM2's input, valid pairs) AND the pair-domain
   ``act[P, W]`` (the UNIVERSAL output — sentinel pairs get
   ``activation(base)`` with a zero delta, the §65.1 contract) — the
   pair store costs nothing extra because the value is in registers, and
   it removes the staged-act re-read a separate scatter would pay;
4. **GEMM2**: masked grouped down-A over the same dispatch
   (``staged_act [V, m_max, W] @ a_down [V, R, W] -> c_down``);
5. **scatter**: ONE kernel writes ``down_rank_out[P, R]`` — GEMM rows
   for valid pairs, EXACT ZERO for sentinels.

**Why the activation is not inside the GEMM epilogue.** The compiled
kernel DOES expose an epilogue hook (``epilogue_op`` in
``cutedsl_masked/kernel.py`` — a Constexpr elementwise lambda applied to
each output element before the TMA store), and it was evaluated first
per the Step-5 charter. It is structurally insufficient here: (a) SwiGLU
is not elementwise in the output domain — it pairs gate column ``j``
with up column ``j`` from DIFFERENT output tiles (different GROUPS in
the interleaved slice layout), and the lambda sees exactly one
accumulator value with no column/tile identity; (b) both guardrail
activations apply to ``base + delta`` and the base addend lives in a
second, pair-indexed tensor the epilogue ABI cannot read. The earliest
legal fusion point is therefore the row-local join above. A custom
CuTeDSL kernel that owns gate and up in one tile and joins in its own
epilogue is the identified next rung, out of this arm's scope.

**Materialization ledger** (per §65.1): the S2 kernel cannot chain
in-register, so the gate/up delta IS materialized — but only in the
staged domain, read back exactly once by the join; ``act`` stays
materialized at the common output boundary (it is also the base W2
input); the extra cost vs the Triton FULL arm is the staged round-trip
(``staged_bridge`` write + ``c_slices`` round-trip + ``staged_act``
write), the price of tensor-core GEMM throughput — the prefill win P5
demonstrated for r>=64 gate/up (1.13-2.4x).

**Plan-vs-execute split** (mirrors ``CutedslLoraAPlan``):
``build_cutedsl_fused_middle_plan`` sizes ``m_max`` (host sync, build
only), compiles through the provider's process-global cache with
zero-tile warmup, and allocates every buffer; ``build_metadata``
re-runs dispatch + stage-row derivation + schedules for the SAME route
(chargeable in a route-inclusive thunk, ``verify=False`` only after a
verified call); ``run_middle`` is the prepared-boundary thunk. Two
schedule-builder calls are needed because the slice GEMM and the down
GEMM run in DIFFERENT group domains (``S*V`` vs ``V``); each discarded
second stage is sized to one output cluster. The §45 dual-ownership
rule holds per GEMM: each schedule is built from the same ``masked_m``
tensor its GEMM reads, both derived from one dispatch.

Bitwise stability: dispatch atomics permute rows WITHIN a group per
rebuild, but every staged row is an independent fixed-order K-reduction
at both GEMMs, the join is row-local, and the scatter inverts
``src2dst`` exactly — pair-domain outputs are placement-invariant.
"""

from __future__ import annotations

from typing import Any

import msgspec
import torch
import triton
import triton.language as tl

from benchmark.kernels.lora_moe.lora_a_cutedsl import (  # noqa: F401  (re-exported for bench/tests)
    CutedslAConfig,
    _compiled_masked_gemm,
    supported_token_widths,
)
from sglang.kernels.ops.moe.ep_moe_kernels import fused_moe_dispatch_index
from sglang.srt.lora.sgl_lora.routing import RouteView

# Guardrail activations (§65.1): the slice count is a property of the
# activation, not a free parameter — SwiGLU pairs a gate and an up slice,
# non-gated ReLU^2 consumes a single slice.
ACTIVATION_SLICES: dict[str, int] = {"silu_mul": 2, "relu2": 1}


@triton.jit
def _stage_bridge_slices_kernel(
    bridge_ptr,  # [rows, S*R] bf16
    staged_ptr,  # [S*V*m_max, R] bf16 flat, slice-interleaved groups
    stage_rows_ptr,  # [P_valid] int64: first-slice staged row
    bridge_rows_ptr,  # [P_valid] int64: bridge source row
    m_max,
    stride_bm,
    stride_bk,
    NUM_SLICES: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Gather one valid pair's bridge slices into the interleaved staging.

    Slice ``s`` of the pair staged at ``(v, i)`` lands at flat row
    ``(v * NUM_SLICES + s) * m_max + i = first + s * m_max``.
    """
    idx = tl.program_id(0).to(tl.int64)
    source_row = tl.load(bridge_rows_ptr + idx)
    first = tl.load(stage_rows_ptr + idx)
    offs = tl.arange(0, BLOCK_R).to(tl.int64)
    mask = offs < RANK
    for s in tl.static_range(NUM_SLICES):
        vals = tl.load(
            bridge_ptr + source_row * stride_bm + (s * RANK + offs) * stride_bk,
            mask=mask,
            other=0.0,
        )
        tl.store(staged_ptr + (first + s * m_max) * RANK + offs, vals, mask=mask)


@triton.jit
def _join_staged_rows_kernel(
    c_slices_ptr,  # [S*V*m_max, W] bf16 flat: staged gate/up delta
    base_ptr,  # [P, S*W] bf16: pair-domain base W13 output
    act_pair_ptr,  # [P, W] bf16 out: UNIVERSAL activation
    staged_act_ptr,  # [V*m_max, W] bf16 out: GEMM2 input (valid rows)
    virtual_ids_ptr,  # [P] int32 fused ids, -1 sentinel
    src2dst_ptr,  # [P] int32 (defined at valid positions only)
    m_max,
    inter,
    stride_base_m,
    stride_act_m,
    NUM_SLICES: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Base add + activation at the earliest ABI-legal point (see module
    docstring), dual-storing staged and pair-domain activations.

    Sentinel pairs take masked (zero) delta loads, so their pair-domain
    act is exactly ``activation(base)`` — the §65.1 universal-act
    contract — and they never touch the staged buffers.
    """
    pair = tl.program_id(0).to(tl.int64)
    key = tl.load(virtual_ids_ptr + pair)
    valid = key >= 0
    dst = tl.load(src2dst_ptr + pair, mask=valid, other=0).to(tl.int64)
    group = dst // m_max
    first = group * (NUM_SLICES * m_max) + (dst - group * m_max)
    vec = tl.arange(0, BLOCK_W).to(tl.int64)
    for start in tl.range(0, inter, BLOCK_W):
        offs = start + vec
        w_mask = offs < inter
        gate = tl.load(
            base_ptr + pair * stride_base_m + offs, mask=w_mask, other=0.0
        ).to(tl.float32)
        gate += tl.load(
            c_slices_ptr + first * inter + offs, mask=w_mask & valid, other=0.0
        ).to(tl.float32)
        if NUM_SLICES == 2:
            up = tl.load(
                base_ptr + pair * stride_base_m + inter + offs,
                mask=w_mask,
                other=0.0,
            ).to(tl.float32)
            up += tl.load(
                c_slices_ptr + (first + m_max) * inter + offs,
                mask=w_mask & valid,
                other=0.0,
            ).to(tl.float32)
            act = gate * tl.sigmoid(gate) * up
        else:
            rectified = tl.maximum(gate, 0.0)
            act = rectified * rectified
        act_store = act.to(act_pair_ptr.dtype.element_ty)
        tl.store(act_pair_ptr + pair * stride_act_m + offs, act_store, mask=w_mask)
        tl.store(staged_act_ptr + dst * inter + offs, act_store, mask=w_mask & valid)


@triton.jit
def _scatter_down_rank_kernel(
    c_down_ptr,  # [V*m_max, R] bf16 flat
    down_rank_ptr,  # [P, R] bf16 out
    virtual_ids_ptr,  # [P] int32
    src2dst_ptr,  # [P] int32
    stride_dm,
    stride_dk,
    RANK: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """The single pair-domain scatter: GEMM rows for valid pairs, EXACT
    ZERO for sentinels (down_rank_out is consumed additively downstream —
    B's ownership contract, not A's preserve contract)."""
    pair = tl.program_id(0).to(tl.int64)
    key = tl.load(virtual_ids_ptr + pair)
    valid = key >= 0
    dst = tl.load(src2dst_ptr + pair, mask=valid, other=0).to(tl.int64)
    offs = tl.arange(0, BLOCK_R).to(tl.int64)
    mask = offs < RANK
    vals = tl.load(c_down_ptr + dst * RANK + offs, mask=mask & valid, other=0.0)
    tl.store(
        down_rank_ptr + pair * stride_dm + offs * stride_dk,
        vals.to(down_rank_ptr.dtype.element_ty),
        mask=mask,
    )


def _derived_stage_rows(
    src2dst: torch.Tensor,
    valid_pair_ids: torch.Tensor,
    *,
    m_max: int,
    num_slices: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(src2dst_valid, first-slice staged rows) for the CURRENT dispatch.

    The atomics assign a fresh within-group order every dispatch, so both
    tensors must be refreshed together with ``src2dst`` (the A-plan's
    seventh-review lesson: a stale inverse map silently permutes).
    """
    src2dst_valid = src2dst.index_select(0, valid_pair_ids).to(torch.int64)
    group = src2dst_valid // m_max
    row = src2dst_valid - group * m_max
    return src2dst_valid, group * (num_slices * m_max) + row


def _build_middle_schedules(
    *,
    masked_m: torch.Tensor,
    masked_m_slices: torch.Tensor,
    m_max: int,
    config: CutedslAConfig,
    intermediate: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Schedules for both GEMMs. Two builder calls because the GEMMs run
    in different group domains (``S*V`` vs ``V``); each call's unused
    second stage is sized to a single output cluster."""
    from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_masked.schedule_builder import (  # noqa: E501
        build_dual_stage_schedules,
    )

    schedule_slices, tiles_slices, _, _ = build_dual_stage_schedules(
        masked_m_slices,
        m_max=m_max,
        token_width=config.token_width,
        n_gemm1=intermediate,
        n_gemm2=config.output_width,
        output_width=config.output_width,
    )
    schedule_down, tiles_down, _, _ = build_dual_stage_schedules(
        masked_m,
        m_max=m_max,
        token_width=config.token_width,
        n_gemm1=rank,
        n_gemm2=config.output_width,
        output_width=config.output_width,
    )
    return schedule_slices, tiles_slices, schedule_down, tiles_down


class CutedslFusedMiddlePlan(msgspec.Struct, kw_only=True):
    """Per-fixture prepared state for one fused-middle leg.

    Everything here is derivable per batch/layer; a timed thunk declares
    what it re-executes through ``build_metadata`` / ``run_middle``.
    """

    config: CutedslAConfig
    activation: str
    num_slices: int
    intermediate_top_k: int
    num_groups: int  # V
    m_max: int
    top_k: int
    rank: int
    intermediate: int  # W
    virtual_topk_ids: torch.Tensor  # [T, K] int32, -1 sentinel (derived copy)
    valid_pair_ids: torch.Tensor  # [P_valid] int64
    bridge_rows_valid: torch.Tensor  # [P_valid] int64 (valid_pair_ids // itk)
    masked_m: torch.Tensor  # [V] int32
    masked_m_slices: torch.Tensor  # [S*V] int32 (interleaved repeat)
    src2dst: torch.Tensor  # [T*K] int32
    src2dst_valid: torch.Tensor  # [P_valid] int64
    stage_rows_first: torch.Tensor  # [P_valid] int64
    staged_bridge: torch.Tensor  # [S*V, m_max, R] bf16
    c_slices: torch.Tensor  # [S*V, m_max, W] bf16
    staged_act: torch.Tensor  # [V, m_max, W] bf16
    c_down: torch.Tensor  # [V, m_max, R] bf16
    schedule_slices: torch.Tensor
    tiles_slices: torch.Tensor
    schedule_down: torch.Tensor
    tiles_down: torch.Tensor
    compiled_slices: Any
    compiled_down: Any
    b_arg_slices: Any
    b_arg_down: Any
    # Torch owners of the wrapped weights: the DLPack wrappers hold no
    # reference, and binding needs the identity the plan compiled against.
    weight_owner_gate_up: torch.Tensor | None = None
    weight_owner_down: torch.Tensor | None = None
    # Source-route identity: virtual_topk_ids is a derived copy, so the
    # SOURCE tensors' addresses are what binding must remember.
    route_topk_ptr: int = 0
    route_slots_ptr: int = 0

    def require_binding(
        self,
        *,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        routing,
    ) -> None:
        """A declared invocation must be THIS plan's fixture and weights.

        Pointer-level identity (addresses, shapes, strides, domains), the
        same scope as ``CutedslLoraAPlan.require_binding``: in-place
        mutation of the source route keeps addresses and is caught only by
        ``build_metadata(verify=True)`` through group counts and the
        valid-pair set.
        """
        for label, owner, weight in (
            ("gate/up B", self.weight_owner_gate_up, gate_up_weight),
            ("down A", self.weight_owner_down, down_weight),
        ):
            if (
                owner is None
                or owner.data_ptr() != weight.data_ptr()
                or owner.shape != weight.shape
                or owner.stride() != weight.stride()
            ):
                raise ValueError(
                    f"cutedsl fused-middle plan's {label} weight is not the "
                    "tensor this invocation declares (address, shape, and "
                    "strides must all match) — plans bind per fixture and "
                    "per site"
                )
        if routing is None:
            raise ValueError(
                "a declared cutedsl invocation must supply its routing view "
                "— binding cannot be skipped"
            )
        if (
            routing.topk_ids.data_ptr() != self.route_topk_ptr
            or routing.token_slots.data_ptr() != self.route_slots_ptr
            or routing.topk_ids.shape != self.virtual_topk_ids.shape
        ):
            raise ValueError(
                "cutedsl fused-middle plan was dispatched from a different "
                "route than the declared routing view (source tensor "
                "mismatch)"
            )
        if routing.num_virtual_experts != self.num_groups:
            raise ValueError(
                "the declared routing view's virtual-expert domain "
                f"({routing.num_virtual_experts}) is not this plan's "
                f"dispatch domain ({self.num_groups})"
            )

    # ---- metadata ----

    def build_metadata(self, verify: bool = True) -> None:
        """Re-run dispatch + derived rows + schedules for the SAME route.

        Strictly same-route-rebuild: ``m_max`` was sized at plan build and
        the dispatch kernel has no overflow guard. ``verify=True`` (the
        default) re-checks group counts and the valid-pair set against the
        build; a timed thunk may pass ``verify=False`` ONLY after a
        verified call on the identical route.
        """
        masked_m, src2dst = fused_moe_dispatch_index(
            self.virtual_topk_ids, self.num_groups, self.m_max
        )
        if verify and int(masked_m.max().item()) > self.m_max:
            raise ValueError(
                f"route contents changed: widest group "
                f"{int(masked_m.max().item())} exceeds the sized m_max="
                f"{self.m_max}; rebuild the plan for the new route"
            )
        if verify:
            fresh_valid = (
                (self.virtual_topk_ids.view(-1) >= 0).nonzero(as_tuple=False).view(-1)
            )
            if not torch.equal(masked_m, self.masked_m) or not torch.equal(
                fresh_valid, self.valid_pair_ids
            ):
                raise ValueError(
                    "route contents changed since plan build (group counts "
                    "or valid-pair set differ) — this plan is fixture-only; "
                    "rebuild it for the new route"
                )
        self.masked_m.copy_(masked_m)
        self.src2dst.copy_(src2dst)
        src2dst_valid, stage_rows = _derived_stage_rows(
            src2dst,
            self.valid_pair_ids,
            m_max=self.m_max,
            num_slices=self.num_slices,
        )
        self.src2dst_valid.copy_(src2dst_valid)
        self.stage_rows_first.copy_(stage_rows)
        self.masked_m_slices.view(-1, self.num_slices).copy_(
            masked_m.unsqueeze(1).expand(-1, self.num_slices)
        )
        schedule_slices, tiles_slices, schedule_down, tiles_down = (
            _build_middle_schedules(
                masked_m=self.masked_m,
                masked_m_slices=self.masked_m_slices,
                m_max=self.m_max,
                config=self.config,
                intermediate=self.intermediate,
                rank=self.rank,
            )
        )
        self.schedule_slices.copy_(schedule_slices)
        self.tiles_slices.copy_(tiles_slices)
        self.schedule_down.copy_(schedule_down)
        self.tiles_down.copy_(tiles_down)

    # ---- pipeline stages ----

    def _stage(self, bridge_gu: torch.Tensor) -> None:
        num_valid = self.valid_pair_ids.numel()
        if num_valid == 0:
            return
        _stage_bridge_slices_kernel[(num_valid,)](
            bridge_gu,
            self.staged_bridge,
            self.stage_rows_first,
            self.bridge_rows_valid,
            self.m_max,
            bridge_gu.stride(0),
            bridge_gu.stride(1),
            NUM_SLICES=self.num_slices,
            RANK=self.rank,
            BLOCK_R=triton.next_power_of_2(self.rank),
        )

    def _launch(self, stage: str) -> None:
        import cuda.bindings.driver as cuda_driver

        from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_masked.api import (
            as_dynamic_cute_tensor as dyn,
        )

        if stage == "slices":
            compiled, a, c = self.compiled_slices, self.staged_bridge, self.c_slices
            b_arg, masked_m = self.b_arg_slices, self.masked_m_slices
            schedule, tiles = self.schedule_slices, self.tiles_slices
        elif stage == "down":
            compiled, a, c = self.compiled_down, self.staged_act, self.c_down
            b_arg, masked_m = self.b_arg_down, self.masked_m
            schedule, tiles = self.schedule_down, self.tiles_down
        else:
            raise ValueError(f"no GEMM stage named {stage!r}")
        stream = cuda_driver.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
        compiled(
            dyn(a, leading_dim=2),
            b_arg,
            dyn(c, leading_dim=2),
            dyn(masked_m, leading_dim=0),
            dyn(schedule, leading_dim=0),
            dyn(tiles, leading_dim=0),
            stream,
        )

    def _join(self, *, base_gu: torch.Tensor, act_out: torch.Tensor) -> None:
        num_pairs = self.virtual_topk_ids.numel()
        _join_staged_rows_kernel[(num_pairs,)](
            self.c_slices,
            base_gu,
            act_out,
            self.staged_act,
            self.virtual_topk_ids,
            self.src2dst,
            self.m_max,
            self.intermediate,
            base_gu.stride(0),
            act_out.stride(0),
            NUM_SLICES=self.num_slices,
            BLOCK_W=512,
        )

    def _scatter(self, *, down_rank_out: torch.Tensor) -> None:
        num_pairs = self.virtual_topk_ids.numel()
        _scatter_down_rank_kernel[(num_pairs,)](
            self.c_down,
            down_rank_out,
            self.virtual_topk_ids,
            self.src2dst,
            down_rank_out.stride(0),
            down_rank_out.stride(1),
            RANK=self.rank,
            BLOCK_R=triton.next_power_of_2(self.rank),
        )

    # ---- arm composition ----

    def run_middle(
        self,
        *,
        bridge_gu: torch.Tensor,
        base_gu: torch.Tensor,
        act_out: torch.Tensor,
        down_rank_out: torch.Tensor,
    ) -> None:
        """Prepared-boundary thunk: stage + GEMM1 + join + GEMM2 + scatter
        (metadata prebuilt; a route-inclusive frame calls
        ``build_metadata`` first)."""
        if self.virtual_topk_ids.numel() == 0:
            return
        self._stage(bridge_gu)
        self._launch("slices")
        self._join(base_gu=base_gu, act_out=act_out)
        self._launch("down")
        self._scatter(down_rank_out=down_rank_out)

    def gemm_only(self, stage: str) -> None:
        """Diagnostic ideal bound per GEMM (``"slices"`` or ``"down"``):
        what a fused gather-GEMM could not beat."""
        self._launch(stage)


def _validate_middle_build(
    *,
    b_gate_up: torch.Tensor,
    a_down: torch.Tensor,
    num_groups: int,
    activation: str,
    intermediate_top_k: int,
    top_k: int,
) -> tuple[int, int, int]:
    """Fail-closed geometry checks BEFORE any compile or allocation.

    Returns ``(num_slices, rank, intermediate)``.
    """
    from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_masked.schedule_builder import (  # noqa: E501
        MAX_EXPERTS,
    )

    if activation not in ACTIVATION_SLICES:
        raise ValueError(
            f"activation must be one of {tuple(ACTIVATION_SLICES)}, got "
            f"{activation!r}"
        )
    num_slices = ACTIVATION_SLICES[activation]
    if b_gate_up.ndim != 3 or a_down.ndim != 3:
        raise ValueError(
            "expected b_gate_up [V, S*W, R] and a_down [V, R, W]; got "
            f"{tuple(b_gate_up.shape)} / {tuple(a_down.shape)}"
        )
    if not b_gate_up.is_contiguous() or not a_down.is_contiguous():
        raise ValueError(
            "factor weights must be contiguous — a silent .contiguous() "
            "copy here would wrap a temporary whose Torch owner dies, and "
            "the slice-interleaved view exists only on the contiguous layout"
        )
    if b_gate_up.dtype != torch.bfloat16 or a_down.dtype != torch.bfloat16:
        raise ValueError("the masked grouped GEMM ABI is BF16-only")
    if b_gate_up.shape[0] != num_groups or a_down.shape[0] != num_groups:
        raise ValueError(
            f"factor weights must have {num_groups} virtual groups; got "
            f"{b_gate_up.shape[0]} / {a_down.shape[0]}"
        )
    rank = b_gate_up.shape[2]
    intermediate = a_down.shape[2]
    if a_down.shape[1] != rank:
        raise ValueError(f"a_down rank {a_down.shape[1]} != b_gate_up rank {rank}")
    if b_gate_up.shape[1] != num_slices * intermediate:
        raise ValueError(
            f"activation {activation!r} needs b_gate_up rows == "
            f"{num_slices} * W = {num_slices * intermediate}, got "
            f"{b_gate_up.shape[1]}"
        )
    if rank % 8 or intermediate % 8:
        raise ValueError(
            f"R={rank} and W={intermediate} must be multiples of 8 for "
            "16-byte TMA alignment"
        )
    if num_slices * num_groups > MAX_EXPERTS:
        raise ValueError(
            f"{num_slices} slices x {num_groups} virtual groups exceed the "
            f"direct schedule's {MAX_EXPERTS}-group packing (the slice-"
            "interleaved GEMM1 doubles the group domain)"
        )
    if intermediate_top_k not in (1, top_k):
        raise ValueError(
            f"intermediate_top_k must be 1 or the route top_k {top_k}, got "
            f"{intermediate_top_k}"
        )
    return num_slices, rank, intermediate


def _allocate_middle_plan(
    *,
    config: CutedslAConfig,
    activation: str,
    num_slices: int,
    intermediate_top_k: int,
    num_groups: int,
    m_max: int,
    top_k: int,
    rank: int,
    intermediate: int,
    virtual_topk_ids: torch.Tensor,
    valid_pair_ids: torch.Tensor,
) -> CutedslFusedMiddlePlan:
    """Compile (via the shared cache) and allocate every plan buffer."""
    device = virtual_topk_ids.device
    flat = virtual_topk_ids.view(-1)
    return CutedslFusedMiddlePlan(
        config=config,
        activation=activation,
        num_slices=num_slices,
        intermediate_top_k=intermediate_top_k,
        num_groups=num_groups,
        m_max=m_max,
        top_k=top_k,
        rank=rank,
        intermediate=intermediate,
        virtual_topk_ids=virtual_topk_ids,
        valid_pair_ids=valid_pair_ids,
        bridge_rows_valid=valid_pair_ids // intermediate_top_k,
        masked_m=torch.empty(num_groups, dtype=torch.int32, device=device),
        masked_m_slices=torch.empty(
            num_slices * num_groups, dtype=torch.int32, device=device
        ),
        src2dst=torch.empty(flat.numel(), dtype=torch.int32, device=device),
        src2dst_valid=torch.empty(
            valid_pair_ids.numel(), dtype=torch.int64, device=device
        ),
        stage_rows_first=torch.empty(
            valid_pair_ids.numel(), dtype=torch.int64, device=device
        ),
        staged_bridge=torch.zeros(
            (num_slices * num_groups, m_max, rank),
            dtype=torch.bfloat16,
            device=device,
        ),
        c_slices=torch.empty(
            (num_slices * num_groups, m_max, intermediate),
            dtype=torch.bfloat16,
            device=device,
        ),
        staged_act=torch.zeros(
            (num_groups, m_max, intermediate), dtype=torch.bfloat16, device=device
        ),
        c_down=torch.empty(
            (num_groups, m_max, rank), dtype=torch.bfloat16, device=device
        ),
        schedule_slices=torch.empty(0, dtype=torch.int32, device=device),
        tiles_slices=torch.empty(1, dtype=torch.int32, device=device),
        schedule_down=torch.empty(0, dtype=torch.int32, device=device),
        tiles_down=torch.empty(1, dtype=torch.int32, device=device),
        compiled_slices=_compiled_masked_gemm(
            device=device,
            config=config,
            num_groups=num_slices * num_groups,
            n=intermediate,
            k=rank,
        ),
        compiled_down=_compiled_masked_gemm(
            device=device,
            config=config,
            num_groups=num_groups,
            n=rank,
            k=intermediate,
        ),
        b_arg_slices=None,
        b_arg_down=None,
    )


def build_cutedsl_fused_middle_plan(
    *,
    fused_route: RouteView,
    b_gate_up: torch.Tensor,
    a_down: torch.Tensor,
    config: CutedslAConfig,
    activation: str = "silu_mul",
    intermediate_top_k: int = 1,
) -> CutedslFusedMiddlePlan:
    """Size, compile, and pre-run the metadata for one fixture.

    ``fused_route`` must be the ROUTE_FUSED_IDS view of the SAME case
    route the Triton arms run on — one id domain, one grouping. The
    ``m_max`` host sync happens HERE, outside any timed thunk (the same
    deliberate boundary as the A composite).
    """
    virtual_topk_ids = fused_route.virtual_topk_ids.to(torch.int32).contiguous()
    num_groups = fused_route.num_virtual_experts
    top_k = virtual_topk_ids.shape[1]
    num_slices, rank, intermediate = _validate_middle_build(
        b_gate_up=b_gate_up,
        a_down=a_down,
        num_groups=num_groups,
        activation=activation,
        intermediate_top_k=intermediate_top_k,
        top_k=top_k,
    )

    # Sizing pass: masked_m only, then right-size m_max token-width aligned.
    masked_m_probe, _ = fused_moe_dispatch_index(virtual_topk_ids, num_groups, 1)
    widest = int(masked_m_probe.max().item())
    m_max = max(
        (widest + config.token_width - 1) // config.token_width * config.token_width,
        config.token_width,
    )
    flat = virtual_topk_ids.view(-1)
    valid_pair_ids = (flat >= 0).nonzero(as_tuple=False).view(-1)
    plan = _allocate_middle_plan(
        config=config,
        activation=activation,
        num_slices=num_slices,
        intermediate_top_k=intermediate_top_k,
        num_groups=num_groups,
        m_max=m_max,
        top_k=top_k,
        rank=rank,
        intermediate=intermediate,
        virtual_topk_ids=virtual_topk_ids,
        valid_pair_ids=valid_pair_ids,
    )
    from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_masked.api import (
        as_dynamic_cute_tensor,
    )

    # The slice-interleaved weight is a PURE VIEW of the contiguous
    # [V, 2W, R] tensor: group 2v is expert v's gate slice, 2v+1 its up
    # slice. Owner identity stays the original tensor.
    plan.b_arg_slices = as_dynamic_cute_tensor(
        b_gate_up.view(num_slices * num_groups, intermediate, rank), leading_dim=2
    )
    plan.b_arg_down = as_dynamic_cute_tensor(a_down, leading_dim=2)
    plan.weight_owner_gate_up = b_gate_up
    plan.weight_owner_down = a_down
    plan.route_topk_ptr = fused_route.topk_ids.data_ptr()
    plan.route_slots_ptr = fused_route.token_slots.data_ptr()

    # Real dispatch + schedules once at build; schedule buffers keep the
    # builder's own capacity sizing so build_metadata can copy_ in place.
    masked_m, src2dst = fused_moe_dispatch_index(virtual_topk_ids, num_groups, m_max)
    plan.masked_m.copy_(masked_m)
    plan.src2dst.copy_(src2dst)
    src2dst_valid, stage_rows = _derived_stage_rows(
        src2dst, valid_pair_ids, m_max=m_max, num_slices=num_slices
    )
    plan.src2dst_valid.copy_(src2dst_valid)
    plan.stage_rows_first.copy_(stage_rows)
    plan.masked_m_slices.view(-1, num_slices).copy_(
        masked_m.unsqueeze(1).expand(-1, num_slices)
    )
    (
        plan.schedule_slices,
        plan.tiles_slices,
        plan.schedule_down,
        plan.tiles_down,
    ) = _build_middle_schedules(
        masked_m=plan.masked_m,
        masked_m_slices=plan.masked_m_slices,
        m_max=m_max,
        config=config,
        intermediate=intermediate,
        rank=rank,
    )
    return plan


def _validate_middle_call(
    *,
    plan: CutedslFusedMiddlePlan,
    bridge_gu: torch.Tensor,
    base_gu: torch.Tensor,
    act_out: torch.Tensor,
    down_rank_out: torch.Tensor,
) -> None:
    """Fail-closed per-call shape/dtype/device contract."""
    num_pairs = plan.virtual_topk_ids.numel()
    num_tokens = plan.virtual_topk_ids.shape[0]
    bridge_rows = num_pairs if plan.intermediate_top_k == 1 else num_tokens
    expected = {
        "bridge_gu": (bridge_gu, (bridge_rows, plan.num_slices * plan.rank)),
        "base_gu": (base_gu, (num_pairs, plan.num_slices * plan.intermediate)),
        "act_out": (act_out, (num_pairs, plan.intermediate)),
        "down_rank_out": (down_rank_out, (num_pairs, plan.rank)),
    }
    for name, (tensor, shape) in expected.items():
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must be {shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != torch.bfloat16:
            raise ValueError(
                f"{name} must be bfloat16 (the staged GEMM ABI), got " f"{tensor.dtype}"
            )
    for name, tensor in (("base_gu", base_gu), ("act_out", act_out)):
        # The join kernel indexes rows with unit element stride; a
        # column-sliced view would silently read/write wrong cells.
        if tensor.stride(1) != 1:
            raise ValueError(
                f"{name} rows must be contiguous (element stride 1), got "
                f"stride {tuple(tensor.stride())}"
            )
    devices = {
        bridge_gu.device,
        base_gu.device,
        act_out.device,
        down_rank_out.device,
        plan.virtual_topk_ids.device,
    }
    if len(devices) != 1:
        raise ValueError(f"tensors span devices {sorted(map(str, devices))}")


def invoke_cutedsl_fused_middle(
    *,
    bridge_gu: torch.Tensor,
    b_gate_up: torch.Tensor,
    base_gu: torch.Tensor,
    a_down: torch.Tensor,
    routing: RouteView,
    act: torch.Tensor,
    down_rank_out: torch.Tensor,
    plan: CutedslFusedMiddlePlan,
) -> None:
    """Execute the fused middle at the prepared-metadata boundary.

    Input/output contract matches the Triton ``run_fused_middle`` arms
    (same tensor vocabulary): ``bridge_gu`` [P or T, S*R], ``b_gate_up``
    [V, S*W, R], ``base_gu`` [P, S*W], ``a_down`` [V, R, W]; outputs
    ``act`` [P, W] (universal — sentinel pairs get ``activation(base)``)
    and ``down_rank_out`` [P, R] (exact zero at sentinel pairs).
    ``routing`` and the weights are validated against the plan's binding;
    per-forward metadata is NOT rebuilt here — a route-inclusive bench
    frame calls ``plan.build_metadata`` inside its thunk, a prepared
    frame outside.
    """
    plan.require_binding(gate_up_weight=b_gate_up, down_weight=a_down, routing=routing)
    _validate_middle_call(
        plan=plan,
        bridge_gu=bridge_gu,
        base_gu=base_gu,
        act_out=act,
        down_rank_out=down_rank_out,
    )
    plan.run_middle(
        bridge_gu=bridge_gu,
        base_gu=base_gu,
        act_out=act,
        down_rank_out=down_rank_out,
    )


def reference_fused_middle(
    *,
    bridge_gu: torch.Tensor,
    b_gate_up: torch.Tensor,
    base_gu: torch.Tensor,
    a_down: torch.Tensor,
    virtual_topk_ids: torch.Tensor,
    activation: str,
    intermediate_top_k: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch FP32 admission reference: ``(act [P, W], down [P, R])``.

    Group-loop like ``reference.py`` (memory O(P*W), never O(P*W*R)), pure
    FP32 end to end (K1 ruling: the oracle never rounds along with the
    candidate). Sentinel pairs: ``act = activation(base)``, ``down = 0``.
    Host-syncing (``unique``) — admission only, never inside a timed
    region.
    """
    if activation not in ACTIVATION_SLICES:
        raise ValueError(
            f"activation must be one of {tuple(ACTIVATION_SLICES)}, got "
            f"{activation!r}"
        )
    num_slices = ACTIVATION_SLICES[activation]
    rank = b_gate_up.shape[2]
    inter = a_down.shape[2]
    flat = virtual_topk_ids.reshape(-1).to(torch.int64)
    num_pairs = flat.numel()
    valid = flat >= 0

    pre = base_gu.detach().to(torch.float32).clone()
    down = torch.zeros(num_pairs, rank, dtype=torch.float32, device=base_gu.device)
    bridge = bridge_gu.detach().to(torch.float32)
    pair_index = torch.arange(num_pairs, device=base_gu.device)
    for group in torch.unique(flat[valid]).tolist():
        rows = pair_index[flat == group]
        x = bridge[rows // intermediate_top_k]  # [n, S*R]
        weight = b_gate_up[group].to(torch.float32)  # [S*W, R]
        for s in range(num_slices):
            pre[rows, s * inter : (s + 1) * inter] += (
                x[:, s * rank : (s + 1) * rank]
                @ weight[s * inter : (s + 1) * inter, :].T
            )
    if activation == "silu_mul":
        act = torch.nn.functional.silu(pre[:, :inter]) * pre[:, inter:]
    else:
        act = torch.relu(pre) ** 2
    for group in torch.unique(flat[valid]).tolist():
        rows = pair_index[flat == group]
        down[rows] = act[rows] @ a_down[group].to(torch.float32).T
    return act, down
