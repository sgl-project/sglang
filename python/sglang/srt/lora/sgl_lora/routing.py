"""Canonical virtual-expert routing for SGL LoRA MoE kernels.

A schedule requests only the route representation it consumes (execution plan
sections 7.1 and 29 R1).  Three views exist:

``raw``
    Nothing is materialized.  An indexed schedule derives ``(adapter, factor)``
    per pair inline from ``topk_ids`` and ``token_slots``; building a plan it
    never reads is pure cost.
``fused_ids``
    ``virtual_topk_ids`` only.  For a consumer that wants the fused key without
    the sort-and-pad plan.
``aligned``
    The block plan, for grouped GEMMs. NOTE: not a superset of ``fused_ids`` —
    when the policy selects the fused kernel, the key array is never
    materialized and ``virtual_topk_ids`` raises; consumers of the plan read
    only the ``[T, K]`` shape off ``topk_ids``.

Which view wins is a property of the CONSUMING kernel, not of routing, so this
module offers the menu and never picks from it.  Fields belonging to a view
that was not built are ``None``; read them through the accessors, which name
the view you would have had to request.
"""

from __future__ import annotations

import msgspec
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.virtual_experts import (
    _align_block_size_jit,
    _align_block_size_torch,
)

# The JIT align kernel covers 1024 * EPT bucket indices with its largest
# instantiated EPT of 32, and `num_experts` there is V + 1 (the +1 sentinel
# convention), so V tops out one short of 32768.
#
# On CUDA this module never selects the JIT kernel's EPT 16/32 rungs — the
# dispatch below goes fused at V >= 8192. Those rungs are NOT dead code: the
# triton fused_moe_lora path (`virtual_experts._align_block_size_large`) uses
# the same kernel for virtual expert counts in [8192, 32767] (e.g. 384
# experts x 32 slots), where they replaced the old torch fallback.
#
# There is no separate AOT branch: below 1024 buckets the JIT path selects a
# one-thread-per-expert kernel structurally identical to the AOT one, and the
# two measured 0-3.7% apart across 8 paired cells -- noise. One path is simpler
# (execution plan section 36).
_JIT_ALIGN_MAX_VIRTUAL_EXPERTS = 32767

# ALIGNED-view kernel policy, sited by the section 40 redo (15 rung-edge-aware
# V values x P in {8..16384} x 3 seeds x 2 interleaved repeats x graph AND
# eager modes x iid+hotset routes; winner = unanimous sign + >=1.05 geo margin;
# GB300, evidence align_boundary_v1.json / align_p_edge_v1.json).
#
# No scalar V boundary exists: the fused kernel NEVER LOSES the P=16384
# column (decided wins in 41 of 42 cells; the one exception, eager V=4096, is
# a fused-leaning tie at 1.0455), while JIT wins small-P below the EPT 8->16
# rung. The policy is therefore two predicates, OR-ed:
#
# * V >= 8192 -- the JIT kernel's EPT 8->16 rung edge, where its serial
#   per-thread leg doubles and the fused kernel passes it (graph-mode wins
#   from 1.07x at the edge to 2.25x at V=32736; graph P=2048 at the edge
#   leans jit but is tied under the gate). Its only decided losses at the
#   edge are eager small-P cells (<=1.09x), a regime production does not run:
#   decode executes under graphs.
# * P >= 16384 -- see above; never lost in either mode. P=8192 measured
#   jit-or-tied at small V except one decided fused win at eager V=1024
#   (1.07x, the policy's one accepted give-away in that direction), so per
#   section 13 rule 3 the unsampled (8192, 16384) interior stays JIT.
_FUSED_ALIGN_MIN_VIRTUAL = 8192
_FUSED_ALIGN_MIN_PAIRS = 16384

ROUTE_RAW = "raw"
ROUTE_FUSED_IDS = "fused_ids"
ROUTE_ALIGNED = "aligned"
ROUTE_VIEWS = (ROUTE_RAW, ROUTE_FUSED_IDS, ROUTE_ALIGNED)


class RouteView(msgspec.Struct, frozen=True, kw_only=True):
    """One route representation over canonical ``(adapter, factor expert)`` IDs.

    Carries the source tensors unconditionally so a ``raw`` consumer has what
    it needs to fuse the key computation into its own kernel.
    """

    view: str
    num_virtual_experts: int
    block_size: int
    topk_ids: torch.Tensor
    token_slots: torch.Tensor
    factor_expert_count: int
    max_loras: int
    routed_expert_to_factor_id: torch.Tensor | None = None
    # Present for `fused_ids`; for `aligned` only when the JIT path built it
    # (the fused kernel derives keys inline and never materializes this).
    maybe_virtual_topk_ids: torch.Tensor | None = None
    # Present only for `aligned`.
    maybe_sorted_pair_ids: torch.Tensor | None = None
    maybe_block_virtual_expert_ids: torch.Tensor | None = None
    maybe_num_pairs_post_padded: torch.Tensor | None = None

    def _require(self, value, field: str, needed: str):
        if value is None:
            raise ValueError(
                f"route view {self.view!r} did not build {field}; the consumer "
                f"must request view {needed!r} or derive it inline"
            )
        return value

    @property
    def virtual_topk_ids(self) -> torch.Tensor:
        return self._require(
            self.maybe_virtual_topk_ids, "virtual_topk_ids", ROUTE_FUSED_IDS
        )

    @property
    def sorted_pair_ids(self) -> torch.Tensor:
        return self._require(
            self.maybe_sorted_pair_ids, "sorted_pair_ids", ROUTE_ALIGNED
        )

    @property
    def block_virtual_expert_ids(self) -> torch.Tensor:
        return self._require(
            self.maybe_block_virtual_expert_ids,
            "block_virtual_expert_ids",
            ROUTE_ALIGNED,
        )

    @property
    def num_pairs_post_padded(self) -> torch.Tensor:
        return self._require(
            self.maybe_num_pairs_post_padded, "num_pairs_post_padded", ROUTE_ALIGNED
        )


@triton.jit
def virtual_expert_ids_inline(
    topk_ids_ptr,
    token_slots_ptr,
    factor_map_ptr,
    pair_ids,
    pair_mask,
    factor_map_size,
    FACTOR_EXPERT_COUNT: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    USE_FACTOR_MAP: tl.constexpr,
):
    """Canonical fused ``(adapter, factor expert)`` key for a block of pairs.

    A Triton device function, not a kernel: this is the ONE definition of the
    key and of what makes a pair valid.  A ``raw``-view consumer (an indexed
    LoRA GEMM) calls it from inside its own mainloop and materializes nothing;
    the ``fused_ids`` builder below calls it and stores the result.  Sharing the
    body is the point — a second hand-inlined copy is how sentinel and
    validity semantics silently diverge between the plan-free and plan-based
    paths, and such a divergence produces wrong LoRA output rather than an
    error (execution plan section 29 R2).

    Returns the fused key per lane, or ``-1`` where the pair must contribute
    nothing.
    """
    token_ids = pair_ids // TOP_K
    # int32 throughout: every operand is bounded by V + 1 < 2**31 (enforced by
    # the fused-align host; the JIT path's own ceiling is far lower), and int64
    # multiplies/divides in this per-pair hot loop cost real issue slots.
    adapter_ids = tl.load(
        token_slots_ptr + token_ids,
        mask=pair_mask,
        other=-1,
    ).to(tl.int32)
    routed_expert_ids = tl.load(
        topk_ids_ptr + pair_ids,
        mask=pair_mask,
        other=-1,
    ).to(tl.int32)
    if USE_FACTOR_MAP:
        in_map = (routed_expert_ids >= 0) & (routed_expert_ids < factor_map_size)
        factor_ids = tl.load(
            factor_map_ptr + routed_expert_ids,
            mask=pair_mask & in_map,
            other=-1,
        ).to(tl.int32)
    else:
        factor_ids = routed_expert_ids

    valid = (
        (adapter_ids >= 0)
        & (adapter_ids < MAX_LORAS)
        & (factor_ids >= 0)
        & (factor_ids < FACTOR_EXPERT_COUNT)
    )
    return tl.where(valid, adapter_ids * FACTOR_EXPERT_COUNT + factor_ids, -1)


@triton.jit
def _build_virtual_topk_ids_kernel(
    topk_ids_ptr,
    token_lora_mapping_ptr,
    factor_map_ptr,
    virtual_topk_ids_ptr,
    num_pairs,
    factor_map_size,
    FACTOR_EXPERT_COUNT: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    USE_FACTOR_MAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pair_ids = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    pair_mask = pair_ids < num_pairs
    virtual_ids = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_lora_mapping_ptr,
        factor_map_ptr,
        pair_ids,
        pair_mask,
        factor_map_size,
        FACTOR_EXPERT_COUNT=FACTOR_EXPERT_COUNT,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        USE_FACTOR_MAP=USE_FACTOR_MAP,
    )
    tl.store(virtual_topk_ids_ptr + pair_ids, virtual_ids, mask=pair_mask)


def _build_virtual_topk_ids(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    factor_expert_count: int,
    max_loras: int,
    routed_expert_to_factor_id: torch.Tensor | None,
) -> torch.Tensor:
    virtual_topk_ids = torch.empty_like(topk_ids)
    if topk_ids.numel() == 0:
        return virtual_topk_ids

    if not topk_ids.is_cuda:
        adapter_valid = (token_lora_mapping >= 0) & (token_lora_mapping < max_loras)
        if routed_expert_to_factor_id is None:
            factor_ids = topk_ids
        elif routed_expert_to_factor_id.numel() == 0:
            factor_ids = torch.full_like(topk_ids, -1)
        else:
            in_map = (topk_ids >= 0) & (topk_ids < routed_expert_to_factor_id.numel())
            safe_ids = topk_ids.clamp(min=0, max=routed_expert_to_factor_id.numel() - 1)
            factor_ids = torch.where(
                in_map,
                routed_expert_to_factor_id[safe_ids],
                -1,
            )
        factor_valid = (factor_ids >= 0) & (factor_ids < factor_expert_count)
        virtual_ids = token_lora_mapping[:, None] * factor_expert_count + factor_ids
        return torch.where(adapter_valid[:, None] & factor_valid, virtual_ids, -1)

    block_size = 1024
    factor_map = (
        topk_ids if routed_expert_to_factor_id is None else routed_expert_to_factor_id
    )
    _build_virtual_topk_ids_kernel[(triton.cdiv(topk_ids.numel(), block_size),)](
        topk_ids,
        token_lora_mapping,
        factor_map,
        virtual_topk_ids,
        topk_ids.numel(),
        0 if routed_expert_to_factor_id is None else factor_map.numel(),
        FACTOR_EXPERT_COUNT=factor_expert_count,
        MAX_LORAS=max_loras,
        TOP_K=topk_ids.shape[1],
        USE_FACTOR_MAP=routed_expert_to_factor_id is not None,
        BLOCK_SIZE=block_size,
    )
    return virtual_topk_ids


def _routing_capacity(
    num_pairs: int,
    block_size: int,
    num_virtual_experts: int,
) -> int:
    if num_pairs == 0:
        return 0
    max_nonempty_buckets = min(num_pairs, num_virtual_experts + 1)
    upper_bound = num_pairs + max_nonempty_buckets * (block_size - 1)
    return triton.cdiv(triton.cdiv(upper_bound, block_size) * block_size, 4) * 4


def _align_virtual_topk_ids(
    virtual_topk_ids: torch.Tensor,
    block_size: int,
    num_virtual_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not virtual_topk_ids.is_cuda:
        return _align_block_size_torch(
            virtual_topk_ids, block_size, num_virtual_experts
        )
    if num_virtual_experts <= _JIT_ALIGN_MAX_VIRTUAL_EXPERTS:
        return _align_block_size_jit(virtual_topk_ids, block_size, num_virtual_experts)
    # Past the JIT kernel's capability ceiling. Our own fused kernel is the only
    # CUDA-speed option here -- measured ~60 us against the torch path's
    # ~4700 us at V = 32768 -- so route to it rather than to torch. It needs the
    # SOURCE tensors, which this signature does not carry, so the caller with
    # them does the dispatch; see `build_virtual_expert_routing`.
    raise NotImplementedError(
        f"align over {num_virtual_experts} virtual experts exceeds the JIT "
        f"kernel's {_JIT_ALIGN_MAX_VIRTUAL_EXPERTS}; callers holding the source "
        "tensors must use sgl_lora.fused_align.fused_align_block_size"
    )


def build_virtual_expert_routing(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    factor_expert_count: int,
    max_loras: int,
    block_size: int,
    routed_expert_to_factor_id: torch.Tensor | None = None,
    view: str = ROUTE_ALIGNED,
) -> RouteView:
    """Build exactly the requested route representation and nothing more."""
    if view not in ROUTE_VIEWS:
        raise ValueError(f"unknown route view {view!r}; expected one of {ROUTE_VIEWS}")
    if topk_ids.ndim != 2 or token_lora_mapping.shape != (topk_ids.shape[0],):
        raise ValueError("expected topk_ids [T,K] and token_lora_mapping [T]")
    if min(factor_expert_count, max_loras, block_size) <= 0:
        raise ValueError(
            "factor count, adapter capacity, and block size must be positive"
        )

    common = {
        "view": view,
        "num_virtual_experts": factor_expert_count * max_loras,
        "block_size": block_size,
        "topk_ids": topk_ids,
        "token_slots": token_lora_mapping,
        "factor_expert_count": factor_expert_count,
        "max_loras": max_loras,
        "routed_expert_to_factor_id": routed_expert_to_factor_id,
    }
    if view == ROUTE_RAW:
        return RouteView(**common)

    num_virtual = common["num_virtual_experts"]
    use_fused = (
        view == ROUTE_ALIGNED
        and topk_ids.is_cuda
        and (
            num_virtual >= _FUSED_ALIGN_MIN_VIRTUAL
            or topk_ids.numel() >= _FUSED_ALIGN_MIN_PAIRS
        )
    )
    if use_fused:
        # The fused kernel DERIVES the key itself, so the `virtual_topk_ids`
        # pass is skipped entirely rather than computed and then recomputed
        # inside the kernel. Consumers read only the [T, K] shape off
        # `topk_ids`, never these values. Above the JIT kernel's 32767 ceiling
        # this is also the only CUDA-speed option.
        #
        # PDL measured 2026-07-25 (align_extp_pdl_v1.json): +10-28% in eager
        # mode across every fused cell -- eager is where launch gaps exist to
        # hide -- and +0.5-1% (never negative) under graph replay. It also
        # erases the policy's only decided losses (eager V=8192 small-P).
        from sglang.kernels.jit.utils import is_arch_support_pdl
        from sglang.srt.lora.sgl_lora.fused_align import fused_align_block_size

        sorted_pair_ids, block_virtual_expert_ids, num_pairs_post_padded = (
            fused_align_block_size(
                topk_ids,
                token_lora_mapping,
                factor_expert_count=factor_expert_count,
                max_loras=max_loras,
                block_size=block_size,
                capacity=_routing_capacity(topk_ids.numel(), block_size, num_virtual),
                routed_expert_to_factor_id=routed_expert_to_factor_id,
                use_pdl=is_arch_support_pdl(),
            )
        )
        return RouteView(
            **common,
            maybe_sorted_pair_ids=sorted_pair_ids,
            maybe_block_virtual_expert_ids=block_virtual_expert_ids,
            maybe_num_pairs_post_padded=num_pairs_post_padded,
        )

    virtual_topk_ids = _build_virtual_topk_ids(
        topk_ids,
        token_lora_mapping,
        factor_expert_count,
        max_loras,
        routed_expert_to_factor_id,
    )
    if view == ROUTE_FUSED_IDS:
        return RouteView(**common, maybe_virtual_topk_ids=virtual_topk_ids)

    sorted_pair_ids, block_virtual_expert_ids, num_pairs_post_padded = (
        _align_virtual_topk_ids(virtual_topk_ids, block_size, num_virtual)
    )
    return RouteView(
        **common,
        maybe_virtual_topk_ids=virtual_topk_ids,
        maybe_sorted_pair_ids=sorted_pair_ids,
        maybe_block_virtual_expert_ids=block_virtual_expert_ids,
        maybe_num_pairs_post_padded=num_pairs_post_padded,
    )
