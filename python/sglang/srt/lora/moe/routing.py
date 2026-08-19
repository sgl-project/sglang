"""Virtual-expert routing for MoE LoRA MoE kernels.

TERMS (all of them are about ONE question: which LoRA A/B weight matrix
does a given (token, expert) pair multiply by?)

``lora expert``
    One expert-specific copy of an adapter's LoRA weight. A normal adapter
    carries one copy per local expert; a SHARED-OUTER adapter carries a
    single copy used by every expert.
``num_local_experts`` + ``is_shared_outer``
    The two facts a caller states. Everything else about the key domain
    follows: the key STRIDE is ``1`` when shared and ``num_local_experts``
    otherwise (``RouteView.lora_experts_per_adapter``), the bucket count is
    ``stride * max_loras``, and the routed id is range-checked against
    ``num_local_experts``. That last check is not decoration in the shared
    form: validity ends in ``lora_expert_id < stride``, which at stride 1 is
    ``0 < 1`` -- always true -- so the count is what stops a routed id this
    rank does not own from passing as valid.

    Do NOT infer the form from the stride instead. A per-expert rank owning
    exactly one expert has stride 1 too, and reading that as shared-outer
    silently builds the wrong keys -- the same bucket-count collision the
    route-factory clobber test drives at ``num_local_experts=1``.
``virtual expert id``
    The bucket key a grouped GEMM sorts by, fusing both halves of the
    question into one integer:
    ``adapter_slot * lora_experts_per_adapter + lora_expert_id``,
    or ``-1`` for a pair that must contribute nothing.

A schedule requests only the route representation it consumes (execution plan
sections 7.1 and 29 R1).  Three views exist:

``raw``
    Nothing is materialized.  An indexed schedule derives ``(adapter, factor)``
    per pair inline from ``topk_ids`` and ``token_lora_mapping``; building a plan it
    never reads is pure cost.
``aligned``
    The block plan, for grouped GEMMs. The JIT arm materializes the key array
    on its way there; the fused arm never does, and consumers of either read
    only the ``[T, K]`` shape off ``topk_ids``.

Which view wins is a property of the CONSUMING kernel, not of routing, so this
module offers the menu and never picks from it.  Fields belonging to a view
that was not built are ``None``; read them through the accessors, which name
the view you would have had to request.
"""

from __future__ import annotations

from enum import Enum

import msgspec
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.virtual_experts import (
    _align_block_size_jit,
)
from sglang.srt.lora.moe.workspace import MoeLoraWorkspace

# The shared JIT align primitive supports at most 8191 real expert buckets.
# This backend switches to its own fused builder at V >= 8192, so it does not
# need to widen that legacy primitive merely to serve larger LoRA domains.
#
# There is no separate AOT branch: below 1024 buckets the JIT path selects a
# one-thread-per-expert kernel structurally identical to the AOT one, and the
# two measured 0-3.7% apart across 8 paired cells -- noise. One path is simpler
# (execution plan section 36).
_JIT_ALIGN_MAX_VIRTUAL_EXPERTS = 8191

# The fused builder takes over exactly at that ceiling + 1; the two constants
# must stay adjacent with no gap, which is why they sit together.
FUSED_ALIGN_MIN_VIRTUAL_EXPERTS = 8192
FUSED_ALIGN_MIN_PAIRS = 16384


# ALIGNED-view kernel config, sited by the section 40 redo (15 rung-edge-aware
# V values x P in {8..16384} x 3 seeds x 2 interleaved repeats x graph AND
# eager modes x iid+hotset routes; winner = unanimous sign + >=1.05 geo margin;
# GB300, evidence align_boundary_v1.json / align_p_edge_v1.json).
#
# No scalar V boundary exists: the fused kernel NEVER LOSES the P=16384
# column (decided wins in 41 of 42 cells; the one exception, eager V=4096, is
# a fused-leaning tie at 1.0455), while JIT wins small-P below the EPT 8->16
# rung. The config is therefore two predicates, OR-ed:
#
# * V >= 8192 -- the JIT kernel's EPT 8->16 rung edge, where its serial
#   per-thread leg doubles and the fused kernel passes it (graph-mode wins
#   from 1.07x at the edge to 2.25x at V=32736; graph P=2048 at the edge
#   leans jit but is tied under the gate). Its only decided losses at the
#   edge are eager small-P cells (<=1.09x), a regime production does not run:
#   decode executes under graphs.
# * P >= 16384 -- see above; never lost in either mode. P=8192 measured
#   jit-or-tied at small V except one decided fused win at eager V=1024
#   (1.07x, the config's one accepted give-away in that direction), so per
#   section 13 rule 3 the unsampled (8192, 16384) interior stays JIT.
class RouteViewKind(str, Enum):
    """Which products a :class:`RouteView` materialized.

    Each value IS the string the ``view`` argument takes, and the members ARE
    the accepted set -- ``view in RouteViewKind`` is the membership test, so no
    parallel tuple has to be kept in step with them.

    NOT interchangeable with ``RouteRequirement``: that one crosses these with
    factor ownership (``aligned_per_expert`` vs ``aligned_shared_outer``) and
    lives in the pydantic plan layer -- which the Triton modules comparing
    views here must not import.
    """

    RAW = "raw"
    ALIGNED = "aligned"


def uses_fused_align(
    topk_ids: torch.Tensor,
    *,
    num_virtual_experts: int,
    view: RouteViewKind = RouteViewKind.ALIGNED,
) -> bool:
    """Return the aligned-route dispatch without building either arm."""
    return view == RouteViewKind.ALIGNED and (
        num_virtual_experts >= FUSED_ALIGN_MIN_VIRTUAL_EXPERTS
        or topk_ids.numel() >= FUSED_ALIGN_MIN_PAIRS
    )


class RouteView(msgspec.Struct, frozen=True, kw_only=True):
    """One route representation over ``(adapter, LoRA expert)`` IDs.

    Carries the source tensors unconditionally so a ``raw`` consumer has what
    it needs to fuse the key computation into its own kernel.
    """

    view: RouteViewKind
    block_size: int
    topk_ids: torch.Tensor
    token_lora_mapping: torch.Tensor
    # How many experts THIS RANK owns, and whether the adapter carries one
    # factor for all of them (shared-outer) or one per expert. Everything else
    # about the key domain follows from these two, so they are stored and the
    # rest derived -- passing the same count in one of two differently-named
    # slots, with 1 as a magic width, is what this replaces.
    num_local_experts: int
    is_shared_outer: bool
    max_loras: int
    # Present only for `aligned`.
    maybe_sorted_pair_ids: torch.Tensor | None = None
    maybe_block_virtual_expert_ids: torch.Tensor | None = None
    maybe_num_pairs_post_padded: torch.Tensor | None = None

    @property
    def lora_experts_per_adapter(self) -> int:
        """Key stride: a shared-outer adapter contributes exactly one bucket."""
        return 1 if self.is_shared_outer else self.num_local_experts

    @property
    def num_virtual_experts(self) -> int:
        return self.lora_experts_per_adapter * self.max_loras

    def _require(self, value, field: str, needed: RouteViewKind):
        if value is None:
            raise ValueError(
                f"route view {self.view.value!r} did not build {field}; the "
                f"consumer must request view {needed.value!r} or derive it inline"
            )
        return value

    @property
    def sorted_pair_ids(self) -> torch.Tensor:
        return self._require(
            self.maybe_sorted_pair_ids, "sorted_pair_ids", RouteViewKind.ALIGNED
        )

    @property
    def block_virtual_expert_ids(self) -> torch.Tensor:
        return self._require(
            self.maybe_block_virtual_expert_ids,
            "block_virtual_expert_ids",
            RouteViewKind.ALIGNED,
        )

    @property
    def num_pairs_post_padded(self) -> torch.Tensor:
        return self._require(
            self.maybe_num_pairs_post_padded,
            "num_pairs_post_padded",
            RouteViewKind.ALIGNED,
        )


@triton.jit
def virtual_expert_ids_inline(
    topk_ids_ptr,
    token_lora_mapping_ptr,
    pair_ids,
    pair_mask,
    routed_expert_id_bound,
    LORA_EXPERTS_PER_ADAPTER: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
):
    """Fused ``(adapter, LoRA expert)`` key for a block of pairs.

    A Triton device function, not a kernel: this is the ONE definition of the
    key and of what makes a pair valid.  A ``raw``-view consumer (an indexed
    LoRA GEMM) calls it from inside its own mainloop and materializes nothing;
    the ``fused_ids`` builder below calls it and stores the result.  Sharing the
    body is the point — a second hand-inlined copy is how sentinel and
    validity semantics silently diverge between the plan-free and plan-based
    paths, and such a divergence produces wrong LoRA output rather than an
    error (execution plan section 29 R2).

    ``SHARED_OUTER`` means the factor is slot 0 for every EP-owned routed
    expert, and ``routed_expert_id_bound`` then carries the EXPLICIT
    expert-range bound: with ``LORA_EXPERTS_PER_ADAPTER == 1`` the generic
    ``factor < count`` check degenerates to ``0 < 1`` and would admit any
    routed id.  Otherwise the routed id IS the LoRA expert id -- ids reach
    this engine already EP-local, which ``MoeLoraRunner._admit`` enforces by
    refusing any dispatcher that keeps global ones.

    Returns the fused key per lane, or ``-1`` where the pair must contribute
    nothing.
    """
    token_ids = pair_ids // TOP_K
    # int32 throughout: every operand is bounded by V + 1 < 2**31 (enforced by
    # the fused-align host; the JIT path's own ceiling is far lower), and int64
    # multiplies/divides in this per-pair hot loop cost real issue slots.
    adapter_ids = tl.load(
        token_lora_mapping_ptr + token_ids,
        mask=pair_mask,
        other=-1,
    ).to(tl.int32)
    routed_expert_ids = tl.load(
        topk_ids_ptr + pair_ids,
        mask=pair_mask,
        other=-1,
    ).to(tl.int32)
    if SHARED_OUTER:
        in_range = (routed_expert_ids >= 0) & (
            routed_expert_ids < routed_expert_id_bound
        )
        lora_expert_ids = tl.where(in_range, 0, -1)
    else:
        lora_expert_ids = routed_expert_ids

    valid = (
        (adapter_ids >= 0)
        & (adapter_ids < MAX_LORAS)
        & (lora_expert_ids >= 0)
        & (lora_expert_ids < LORA_EXPERTS_PER_ADAPTER)
    )
    return tl.where(valid, adapter_ids * LORA_EXPERTS_PER_ADAPTER + lora_expert_ids, -1)


@triton.jit
def _build_virtual_topk_ids_kernel(
    topk_ids_ptr,
    token_lora_mapping_ptr,
    virtual_topk_ids_ptr,
    num_pairs,
    routed_expert_id_bound,
    LORA_EXPERTS_PER_ADAPTER: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pair_ids = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    pair_mask = pair_ids < num_pairs
    virtual_ids = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_lora_mapping_ptr,
        pair_ids,
        pair_mask,
        routed_expert_id_bound,
        LORA_EXPERTS_PER_ADAPTER=LORA_EXPERTS_PER_ADAPTER,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        SHARED_OUTER=SHARED_OUTER,
    )
    tl.store(virtual_topk_ids_ptr + pair_ids, virtual_ids, mask=pair_mask)


def _build_virtual_topk_ids(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    num_local_experts: int,
    max_loras: int,
    is_shared_outer: bool = False,
) -> torch.Tensor:
    # The sole caller validated these same objects before branching.
    virtual_topk_ids = torch.empty_like(topk_ids)
    if topk_ids.numel() == 0:
        # A DP-attention rank with no sequences runs ForwardMode.IDLE to stay
        # in step on the collectives, and reaches here with zero tokens. The
        # grid below would be cdiv(0, 1024) == 0, which is not a launchable
        # configuration (the fused builder's own max(num_pairs, 1) is the same
        # accommodation).
        return virtual_topk_ids

    block_size = 1024
    _build_virtual_topk_ids_kernel[(triton.cdiv(topk_ids.numel(), block_size),)](
        topk_ids,
        token_lora_mapping,
        virtual_topk_ids,
        topk_ids.numel(),
        num_local_experts,
        LORA_EXPERTS_PER_ADAPTER=1 if is_shared_outer else num_local_experts,
        MAX_LORAS=max_loras,
        TOP_K=topk_ids.shape[1],
        SHARED_OUTER=is_shared_outer,
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


def build_virtual_expert_routing(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    num_local_experts: int,
    max_loras: int,
    block_size: int,
    is_shared_outer: bool = False,
    view: RouteViewKind = RouteViewKind.ALIGNED,
    workspace: MoeLoraWorkspace | None = None,
    scratch_prefix: str | None = None,
) -> RouteView:
    """Build exactly the requested route representation and nothing more.

    ``is_shared_outer`` selects the form where the adapter carries ONE factor
    for every expert this rank owns, so the key stride is 1 and the routed id
    is range-checked against ``num_local_experts`` instead. Ids arrive
    EP-local either way; ``MoeLoraRunner._admit`` refuses any dispatcher that
    keeps global ones.

    ``workspace`` plus ``scratch_prefix`` name this route's metadata. The
    fused aligned builder REQUIRES them; every other path ignores them, which
    is why the fused branch both demands and allocates them instead of making
    each caller predict the dispatch.
    """
    if view not in RouteViewKind:
        raise ValueError(
            f"unknown route view {view!r}; expected one of "
            f"{tuple(kind.value for kind in RouteViewKind)}"
        )
    view = RouteViewKind(view)
    if topk_ids.ndim != 2 or token_lora_mapping.shape != (topk_ids.shape[0],):
        raise ValueError("expected topk_ids [T,K] and token_lora_mapping [T]")
    if min(num_local_experts, max_loras, block_size) <= 0:
        raise ValueError(
            "local expert count, adapter capacity, and block size must all "
            "be positive"
        )
    common = {
        "view": view,
        "block_size": block_size,
        "topk_ids": topk_ids,
        "token_lora_mapping": token_lora_mapping,
        "num_local_experts": num_local_experts,
        "is_shared_outer": is_shared_outer,
        "max_loras": max_loras,
    }
    lora_experts_per_adapter = 1 if is_shared_outer else num_local_experts
    if view is RouteViewKind.RAW:
        return RouteView(**common)

    num_virtual = lora_experts_per_adapter * max_loras
    use_fused = uses_fused_align(
        topk_ids,
        num_virtual_experts=num_virtual,
        view=view,
    )
    if use_fused:
        # The fused kernel DERIVES the key itself, so the `virtual_topk_ids`
        # pass is skipped entirely rather than computed and then recomputed
        # inside the kernel. Consumers read only the [T, K] shape off
        # `topk_ids`, never these values. Above the JIT kernel's 8191 ceiling
        # this is also the only CUDA-speed option.
        from sglang.srt.lora.moe.fused_align import fused_align_block_size

        # The fused builder is the only path that keeps metadata across
        # calls, so it is the only one needing a workspace -- and it needs one
        # unconditionally: it names its buffers by route, and two routes
        # sharing a name would share a padded-pair scalar, silently producing
        # a short route rather than failing.
        if workspace is None or scratch_prefix is None:
            raise ValueError(
                "the fused aligned builder needs workspace and scratch_prefix "
                f"(view={view.value}, virtual experts={num_virtual}, "
                f"pairs={topk_ids.numel()})"
            )
        sorted_pair_ids, block_virtual_expert_ids, num_pairs_post_padded = (
            fused_align_block_size(
                topk_ids,
                token_lora_mapping,
                num_local_experts=num_local_experts,
                is_shared_outer=is_shared_outer,
                max_loras=max_loras,
                block_size=block_size,
                capacity=_routing_capacity(topk_ids.numel(), block_size, num_virtual),
                workspace=workspace,
                scratch_prefix=scratch_prefix,
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
        num_local_experts,
        max_loras,
        is_shared_outer=is_shared_outer,
    )
    # Reaching here means the aligned view did NOT dispatch to the fused
    # builder, i.e. V < FUSED_ALIGN_MIN_VIRTUAL_EXPERTS, which is identically
    # V <= _JIT_ALIGN_MAX_VIRTUAL_EXPERTS since the two are adjacent.
    sorted_pair_ids, block_virtual_expert_ids, num_pairs_post_padded = (
        _align_block_size_jit(virtual_topk_ids, block_size, num_virtual)
    )
    return RouteView(
        **common,
        maybe_sorted_pair_ids=sorted_pair_ids,
        maybe_block_virtual_expert_ids=block_virtual_expert_ids,
        maybe_num_pairs_post_padded=num_pairs_post_padded,
    )
