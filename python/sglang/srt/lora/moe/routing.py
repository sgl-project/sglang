"""Canonical virtual-expert routing for MoE LoRA MoE kernels.

TERMS (all of them are about ONE question: which LoRA A/B weight matrix
does a given (token, expert) pair multiply by?)

``lora expert``
    One expert-specific copy of an adapter's LoRA weight. A normal adapter
    carries one copy per local expert; a SHARED-OUTER adapter carries a
    single copy used by every expert.
``lora_experts_per_adapter``
    How many such copies each adapter has: ``num_local_experts`` normally,
    ``1`` when shared. It sets the bucket count ``V = per_adapter *
    max_loras`` and appears in the key formula below, so it is the one
    number this module cannot derive on its own.
``virtual expert id``
    The bucket key a grouped GEMM sorts by, fusing both halves of the
    question into one integer:
    ``adapter_slot * lora_experts_per_adapter + lora_expert_id``,
    or ``-1`` for a pair that must contribute nothing.
``lora_expert_map``
    Optional lookup ``routed expert id -> lora expert id`` (sglang's
    ``expert_map`` for the base weights, applied to LoRA copies). NOT USED
    ON THE PRODUCTION PATH TODAY: the dispatcher hands the runner
    LOCAL-domain ids, so the identity holds and callers pass ``None``. It
    is kept as the escape hatch for topologies that deliver GLOBAL expert
    ids, or that own a non-contiguous set of experts, and its only
    coverage is the lab's global-domain cases. Delete it if Step 11 (real
    distributed validation) shows no real topology needs it — the
    ``USE_LORA_EXPERT_MAP`` constexpr compiles out entirely when unused,
    so keeping it costs nothing at runtime.
``shared_outer_local_expert_count``
    Shared-outer only, and NOT decoration: the validity test ends in
    ``lora_expert_id < lora_experts_per_adapter``, which for a shared
    adapter is ``0 < 1`` — always true. Passing the local expert count
    restores the bound the per-expert path gets for free, so a routed id
    this rank does not own cannot be accepted as a valid pair.

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
    when the config selects the fused kernel, the key array is never
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
from sglang.srt.lora.moe.routing_shape import uses_fused_align_shape

# The shared JIT align primitive supports at most 8191 real expert buckets.
# This backend switches to its own fused builder at V >= 8192, so it does not
# need to widen that legacy primitive merely to serve larger LoRA domains.
#
# There is no separate AOT branch: below 1024 buckets the JIT path selects a
# one-thread-per-expert kernel structurally identical to the AOT one, and the
# two measured 0-3.7% apart across 8 paired cells -- noise. One path is simpler
# (execution plan section 36).
_JIT_ALIGN_MAX_VIRTUAL_EXPERTS = 8191

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
# The constants and pure predicate live in ``routing_shape.py`` so CPU-only
# sweep admission cannot drift from this production dispatch.
ROUTE_RAW = "raw"
ROUTE_FUSED_IDS = "fused_ids"
ROUTE_ALIGNED = "aligned"
ROUTE_VIEWS = (ROUTE_RAW, ROUTE_FUSED_IDS, ROUTE_ALIGNED)


def uses_fused_align(
    topk_ids: torch.Tensor,
    *,
    num_virtual_experts: int,
    view: str = ROUTE_ALIGNED,
) -> bool:
    """Return the canonical aligned-route dispatch without building either arm."""
    return (
        view == ROUTE_ALIGNED
        and topk_ids.is_cuda
        and uses_fused_align_shape(
            num_virtual_experts=num_virtual_experts,
            num_pairs=topk_ids.numel(),
        )
    )


class FusedAlignScratch(msgspec.Struct, frozen=True, kw_only=True):
    """Caller-owned metadata mutated by one fused aligned-route builder.

    The histogram/scan/scatter chain maintains ``counts == 0`` between calls.
    Production supplies one instance per semantic route from the runner
    workspace; direct callers may omit it and use the fused kernel's serial
    fallback cache.
    """

    counts: torch.Tensor
    block_cumulative: torch.Tensor
    cursor: torch.Tensor
    bucket_end: torch.Tensor


class RouteView(msgspec.Struct, frozen=True, kw_only=True):
    """One route representation over canonical ``(adapter, LoRA expert)`` IDs.

    Carries the source tensors unconditionally so a ``raw`` consumer has what
    it needs to fuse the key computation into its own kernel.
    """

    view: str
    num_virtual_experts: int
    block_size: int
    topk_ids: torch.Tensor
    token_slots: torch.Tensor
    lora_experts_per_adapter: int
    max_loras: int
    lora_expert_map: torch.Tensor | None = None
    # Section 60.5 shared-outer form: routed ids in [0, range) map to factor
    # slot 0 with NO map tensor; None everywhere else. Mutually exclusive
    # with the LoRA expert map.
    shared_outer_local_expert_count: int | None = None
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


def validate_shared_outer(
    *,
    shared_outer_local_expert_count: int | None,
    lora_expert_map: torch.Tensor | None,
    lora_experts_per_adapter: int,
) -> None:
    """The shared-outer contract, checked identically at every entry point.

    Eleventh S3 review: ``build_virtual_expert_routing`` enforced all three
    conditions while ``fused_align_block_size`` — reachable directly —
    checked only the mutual exclusion, so a caller could reach the kernels
    with ``lora_experts_per_adapter != 1`` and silently build keys against
    the wrong bucket count.
    """
    if shared_outer_local_expert_count is None:
        return
    if lora_expert_map is not None:
        raise ValueError(
            "shared_outer_local_expert_count replaces the lora expert map "
            "(section 60.5); passing both is contradictory"
        )
    if lora_experts_per_adapter != 1:
        raise ValueError(
            "the shared-outer form has exactly one LoRA expert per adapter; "
            f"lora_experts_per_adapter={lora_experts_per_adapter} is not it"
        )
    if shared_outer_local_expert_count <= 0:
        raise ValueError("shared_outer_local_expert_count must be positive")


@triton.jit
def virtual_expert_ids_inline(
    topk_ids_ptr,
    token_slots_ptr,
    lora_expert_map_ptr,
    pair_ids,
    pair_mask,
    routed_expert_id_bound,
    LORA_EXPERTS_PER_ADAPTER: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    USE_LORA_EXPERT_MAP: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
):
    """Canonical fused ``(adapter, LoRA expert)`` key for a block of pairs.

    A Triton device function, not a kernel: this is the ONE definition of the
    key and of what makes a pair valid.  A ``raw``-view consumer (an indexed
    LoRA GEMM) calls it from inside its own mainloop and materializes nothing;
    the ``fused_ids`` builder below calls it and stores the result.  Sharing the
    body is the point — a second hand-inlined copy is how sentinel and
    validity semantics silently diverge between the plan-free and plan-based
    paths, and such a divergence produces wrong LoRA output rather than an
    error (execution plan section 29 R2).

    ``SHARED_OUTER`` is the section 60.5 cleanup: a shared-outer factor is
    slot 0 for every EP-owned routed expert, so the map lookup that always
    produced 0 is replaced by the constant — while ``routed_expert_id_bound`` keeps
    carrying the EXPLICIT expert-range bound, because with
    ``LORA_EXPERTS_PER_ADAPTER == 1`` the generic ``factor < count`` check
    degenerates to ``0 < 1`` and would admit any routed id.  Callers with
    global-domain ids localize them FIRST (production's convention; the
    dispatcher guarantees it via ``skip_local_expert_mapping == False``).

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
    if SHARED_OUTER:
        in_range = (routed_expert_ids >= 0) & (
            routed_expert_ids < routed_expert_id_bound
        )
        lora_expert_ids = tl.where(in_range, 0, -1)
    elif USE_LORA_EXPERT_MAP:
        in_map = (routed_expert_ids >= 0) & (routed_expert_ids < routed_expert_id_bound)
        lora_expert_ids = tl.load(
            lora_expert_map_ptr + routed_expert_ids,
            mask=pair_mask & in_map,
            other=-1,
        ).to(tl.int32)
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
    lora_expert_map_ptr,
    virtual_topk_ids_ptr,
    num_pairs,
    routed_expert_id_bound,
    LORA_EXPERTS_PER_ADAPTER: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    USE_LORA_EXPERT_MAP: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pair_ids = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    pair_mask = pair_ids < num_pairs
    virtual_ids = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_lora_mapping_ptr,
        lora_expert_map_ptr,
        pair_ids,
        pair_mask,
        routed_expert_id_bound,
        LORA_EXPERTS_PER_ADAPTER=LORA_EXPERTS_PER_ADAPTER,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        USE_LORA_EXPERT_MAP=USE_LORA_EXPERT_MAP,
        SHARED_OUTER=SHARED_OUTER,
    )
    tl.store(virtual_topk_ids_ptr + pair_ids, virtual_ids, mask=pair_mask)


def _build_virtual_topk_ids(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    lora_experts_per_adapter: int,
    max_loras: int,
    lora_expert_map: torch.Tensor | None,
    shared_outer_local_expert_count: int | None = None,
) -> torch.Tensor:
    validate_shared_outer(
        shared_outer_local_expert_count=shared_outer_local_expert_count,
        lora_expert_map=lora_expert_map,
        lora_experts_per_adapter=lora_experts_per_adapter,
    )
    virtual_topk_ids = torch.empty_like(topk_ids)
    if topk_ids.numel() == 0:
        return virtual_topk_ids

    if not topk_ids.is_cuda:
        adapter_valid = (token_lora_mapping >= 0) & (token_lora_mapping < max_loras)
        if shared_outer_local_expert_count is not None:
            in_range = (topk_ids >= 0) & (topk_ids < shared_outer_local_expert_count)
            lora_expert_ids = torch.where(in_range, 0, -1)
        elif lora_expert_map is None:
            lora_expert_ids = topk_ids
        elif lora_expert_map.numel() == 0:
            lora_expert_ids = torch.full_like(topk_ids, -1)
        else:
            in_map = (topk_ids >= 0) & (topk_ids < lora_expert_map.numel())
            safe_ids = topk_ids.clamp(min=0, max=lora_expert_map.numel() - 1)
            lora_expert_ids = torch.where(
                in_map,
                lora_expert_map[safe_ids],
                -1,
            )
        factor_valid = (lora_expert_ids >= 0) & (
            lora_expert_ids < lora_experts_per_adapter
        )
        virtual_ids = (
            token_lora_mapping[:, None] * lora_experts_per_adapter + lora_expert_ids
        )
        return torch.where(adapter_valid[:, None] & factor_valid, virtual_ids, -1)

    block_size = 1024
    use_map = lora_expert_map is not None
    # The kernel always takes a tensor for the map pointer; when the map is
    # unused it reads nothing, so topk_ids stands in and no buffer is
    # allocated. Kept in its OWN name: reassigning lora_expert_map here
    # would shadow the parameter that the two reads below test.
    map_arg = lora_expert_map if use_map else topk_ids
    _build_virtual_topk_ids_kernel[(triton.cdiv(topk_ids.numel(), block_size),)](
        topk_ids,
        token_lora_mapping,
        map_arg,
        virtual_topk_ids,
        topk_ids.numel(),
        (
            shared_outer_local_expert_count
            if shared_outer_local_expert_count is not None
            else (map_arg.numel() if use_map else 0)
        ),
        LORA_EXPERTS_PER_ADAPTER=lora_experts_per_adapter,
        MAX_LORAS=max_loras,
        TOP_K=topk_ids.shape[1],
        USE_LORA_EXPERT_MAP=use_map,
        SHARED_OUTER=shared_outer_local_expert_count is not None,
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
        "tensors must use lora.moe.fused_align.fused_align_block_size"
    )


def build_virtual_expert_routing(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    lora_experts_per_adapter: int,
    max_loras: int,
    block_size: int,
    lora_expert_map: torch.Tensor | None = None,
    shared_outer_local_expert_count: int | None = None,
    view: str = ROUTE_ALIGNED,
    use_pdl: bool | None = None,
    num_pairs_post_padded_out: torch.Tensor | None = None,
    fused_align_scratch: FusedAlignScratch | None = None,
) -> RouteView:
    """Build exactly the requested route representation and nothing more.

    ``shared_outer_local_expert_count`` requests the section 60.5 shared-outer
    form: every routed id inside ``[0, range)`` maps to LoRA expert 0 with
    NO map tensor (the gather was degenerate on the shipping path).
    Requires ``lora_experts_per_adapter == 1`` and local-domain ids — global
    callers localize first, production's convention.

    ``num_pairs_post_padded_out`` is an optional graph-stable destination for
    the fused aligned path. The JIT aligned path already owns its scalar and
    leaves this buffer unused. ``fused_align_scratch`` similarly replaces the
    fused path's process-global serial fallback with caller-owned metadata.
    """
    if view not in ROUTE_VIEWS:
        raise ValueError(f"unknown route view {view!r}; expected one of {ROUTE_VIEWS}")
    if topk_ids.ndim != 2 or token_lora_mapping.shape != (topk_ids.shape[0],):
        raise ValueError("expected topk_ids [T,K] and token_lora_mapping [T]")
    if min(lora_experts_per_adapter, max_loras, block_size) <= 0:
        raise ValueError(
            "LoRA experts per adapter, adapter capacity, and block size "
            "must all be positive"
        )
    validate_shared_outer(
        shared_outer_local_expert_count=shared_outer_local_expert_count,
        lora_expert_map=lora_expert_map,
        lora_experts_per_adapter=lora_experts_per_adapter,
    )

    common = {
        "view": view,
        "num_virtual_experts": lora_experts_per_adapter * max_loras,
        "block_size": block_size,
        "topk_ids": topk_ids,
        "token_slots": token_lora_mapping,
        "lora_experts_per_adapter": lora_experts_per_adapter,
        "max_loras": max_loras,
        "lora_expert_map": lora_expert_map,
        "shared_outer_local_expert_count": shared_outer_local_expert_count,
    }
    if view == ROUTE_RAW:
        return RouteView(**common)

    num_virtual = common["num_virtual_experts"]
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
        #
        # Standard-route PDL is deliberately opt-in. ``None`` and ``False``
        # remain the default-off control; explicit ``True`` is admitted only
        # through bounded composed off/on twins until promotion evidence exists.
        from sglang.srt.lora.moe.fused_align import fused_align_block_size

        sorted_pair_ids, block_virtual_expert_ids, num_pairs_post_padded = (
            fused_align_block_size(
                topk_ids,
                token_lora_mapping,
                lora_experts_per_adapter=lora_experts_per_adapter,
                max_loras=max_loras,
                block_size=block_size,
                capacity=_routing_capacity(topk_ids.numel(), block_size, num_virtual),
                lora_expert_map=lora_expert_map,
                shared_outer_local_expert_count=shared_outer_local_expert_count,
                num_pairs_post_padded_out=num_pairs_post_padded_out,
                scratch=fused_align_scratch,
                use_pdl=bool(use_pdl),
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
        lora_experts_per_adapter,
        max_loras,
        lora_expert_map,
        shared_outer_local_expert_count=shared_outer_local_expert_count,
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


# --------------------------------------------------------------------------
# Dual-GRANULARITY aligned routes: one hist -> scan -> expand pass building
# TWO block plans over the SAME per-expert pair keys.
#
# The consumer is a plan whose grouped gate/up-A prefers a different M tile
# than every other aligned site (GB300 large-prefill:
# gate_a_routing_block_size=64 vs routing_block_size=16).  Two granularities
# share the pair sources, the fused key, and the bucket count; only the
# block-padding arithmetic differs.  Running the fused builder twice repeats
# the whole T*topk key derivation and both bulk passes, so this pass fuses
# them: the histogram derives each key ONCE and bumps one counter array per
# granularity, one two-CTA launch scans both padded cumsums, and one launch
# labels both block tables and scatters each pair into both plans.
#
# This is the dual-granularity sibling of ``joint_routing.py`` (which fuses
# two OWNERSHIPS at one granularity); the kernel bodies deliberately mirror
# that qualified R10 structure and ``fused_align.py``'s audited invariants:
# counts self-restore to zero inside the scan, every stage writes buffers
# disjoint from the ones it bulk-reads, padding fills stay coalesced 2D
# stores, and intra-bucket scatter order is atomic-cursor nondeterministic
# (permitted: block labels and padding are deterministic, consumers index by
# pair id).
_DUAL_HIST_BLOCK = 512
_DUAL_HIST_WARPS = 8
_DUAL_EXPAND_BLOCK = 128
_DUAL_EXPAND_WARPS = 4
_DUAL_SCAN_CHUNK = 2048
_DUAL_SCAN_WARPS = 4


@triton.jit
def _dual_granularity_hist_kernel(
    topk_ids_ptr,
    token_slots_ptr,
    first_counts_ptr,
    second_counts_ptr,
    num_pairs,
    NUM_BUCKETS: tl.constexpr,
    LORA_EXPERTS_PER_ADAPTER: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Derive the per-expert key once, bump both granularities' histograms.

    Both counter arrays arrive ZEROED (each scan restores its own invariant),
    so there is no memset pass.  The key domain is identical for the two
    granularities — only the downstream block padding differs — which is why
    a single derivation feeds two atomics rather than two full passes.
    """
    pair_ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pair_mask = pair_ids < num_pairs
    virtual_ids = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_slots_ptr,
        topk_ids_ptr,
        pair_ids,
        pair_mask,
        0,
        LORA_EXPERTS_PER_ADAPTER=LORA_EXPERTS_PER_ADAPTER,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        USE_LORA_EXPERT_MAP=False,
        SHARED_OUTER=False,
    )
    buckets = tl.where(virtual_ids < 0, NUM_BUCKETS - 1, virtual_ids)
    tl.atomic_add(first_counts_ptr + buckets, 1, mask=pair_mask)
    tl.atomic_add(second_counts_ptr + buckets, 1, mask=pair_mask)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _dual_granularity_scan_one(
    counts_ptr,
    block_cumulative_ptr,
    cursor_ptr,
    bucket_end_ptr,
    padded_pairs_ptr,
    num_buckets,
    BLOCK_SIZE_M: tl.constexpr,
    CHUNK: tl.constexpr,
):
    running = 0
    for base in range(0, num_buckets, CHUNK):
        offsets = base + tl.arange(0, CHUNK)
        mask = offsets < num_buckets
        counts = tl.load(counts_ptr + offsets, mask=mask, other=0)
        # This restores the zero-count invariant for the next forward.
        tl.store(counts_ptr + offsets, 0, mask=mask)
        blocks = (counts + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        block_start = running + tl.cumsum(blocks) - blocks
        tl.store(block_cumulative_ptr + offsets, block_start, mask=mask)
        slot_start = block_start * BLOCK_SIZE_M
        tl.store(cursor_ptr + offsets, slot_start, mask=mask)
        tl.store(bucket_end_ptr + offsets, slot_start + counts, mask=mask)
        running += tl.sum(blocks)
    tl.store(block_cumulative_ptr + num_buckets, running)
    tl.store(padded_pairs_ptr, running * BLOCK_SIZE_M)


@triton.jit
def _dual_granularity_scan_kernel(
    first_counts_ptr,
    first_block_cumulative_ptr,
    first_cursor_ptr,
    first_bucket_end_ptr,
    first_padded_pairs_ptr,
    second_counts_ptr,
    second_block_cumulative_ptr,
    second_cursor_ptr,
    second_bucket_end_ptr,
    second_padded_pairs_ptr,
    num_buckets,
    BLOCK_SIZE_M_FIRST: tl.constexpr,
    BLOCK_SIZE_M_SECOND: tl.constexpr,
    CHUNK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Both padded exclusive scans on one two-CTA launch.

    The two granularities' scans are independent columns of work over the
    same bucket count, so a 2-wide program axis extends the incumbent
    single-CTA scan without touching its serial body; each program keeps its
    own ``BLOCK_SIZE_M`` as a constexpr.
    """
    if USE_PDL:
        # Both scan CTAs consume histogram output immediately.
        tl.extra.cuda.gdc_wait()
        # The expand/scatter kernel can launch now. Its pair path recomputes
        # virtual keys without scan output, then waits immediately before the
        # first cursor load; its label paths wait before their first access.
        tl.extra.cuda.gdc_launch_dependents()
    if tl.program_id(0) == 0:
        _dual_granularity_scan_one(
            first_counts_ptr,
            first_block_cumulative_ptr,
            first_cursor_ptr,
            first_bucket_end_ptr,
            first_padded_pairs_ptr,
            num_buckets,
            BLOCK_SIZE_M=BLOCK_SIZE_M_FIRST,
            CHUNK=CHUNK,
        )
    else:
        _dual_granularity_scan_one(
            second_counts_ptr,
            second_block_cumulative_ptr,
            second_cursor_ptr,
            second_bucket_end_ptr,
            second_padded_pairs_ptr,
            num_buckets,
            BLOCK_SIZE_M=BLOCK_SIZE_M_SECOND,
            CHUNK=CHUNK,
        )


@triton.jit
def _dual_granularity_label_blocks(
    pid,
    block_cumulative_ptr,
    bucket_end_ptr,
    sorted_pair_ids_ptr,
    block_virtual_expert_ids_ptr,
    num_blocks,
    num_pairs,
    NUM_BUCKETS: tl.constexpr,
    NUM_VIRTUAL_EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    block_ids = pid * BLOCK + tl.arange(0, BLOCK)
    block_mask = block_ids < num_blocks
    low = tl.zeros(block_ids.shape, dtype=tl.int32)
    high = tl.full(block_ids.shape, NUM_BUCKETS, dtype=tl.int32)
    for _ in range(SEARCH_STEPS):
        midpoint = (low + high) // 2
        bound = tl.load(
            block_cumulative_ptr + tl.minimum(midpoint + 1, NUM_BUCKETS),
            mask=block_mask,
            other=0,
        )
        take_upper = block_ids >= bound
        low = tl.where(take_upper & (low < high), midpoint + 1, low)
        high = tl.where(take_upper | (low >= high), high, midpoint)
    owner = tl.minimum(low, NUM_BUCKETS - 1)
    total_blocks = tl.load(block_cumulative_ptr + NUM_BUCKETS)
    in_plan = block_mask & (block_ids < total_blocks)
    tl.store(
        block_virtual_expert_ids_ptr + block_ids,
        tl.where(in_plan & (owner < NUM_VIRTUAL_EXPERTS), owner, -1),
        mask=block_mask,
    )
    # Coalesced 2D fill of every in-plan block's padding tail, sentinel
    # bucket included (a -1 block's slots ARE read by the aligned B kernels).
    real_end = tl.load(bucket_end_ptr + owner, mask=in_plan, other=0)
    slots = block_ids[:, None] * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[None, :]
    tl.store(
        sorted_pair_ids_ptr + slots,
        num_pairs,
        mask=in_plan[:, None] & (slots >= real_end[:, None]),
    )


@triton.jit
def _dual_granularity_expand_scatter_kernel(
    topk_ids_ptr,
    token_slots_ptr,
    first_cursor_ptr,
    first_bucket_end_ptr,
    first_block_cumulative_ptr,
    first_sorted_ptr,
    first_block_ids_ptr,
    num_first_blocks,
    first_label_programs,
    second_cursor_ptr,
    second_bucket_end_ptr,
    second_block_cumulative_ptr,
    second_sorted_ptr,
    second_block_ids_ptr,
    num_second_blocks,
    second_label_programs,
    num_pairs,
    NUM_BUCKETS: tl.constexpr,
    NUM_VIRTUAL_EXPERTS: tl.constexpr,
    LORA_EXPERTS_PER_ADAPTER: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_SIZE_M_FIRST: tl.constexpr,
    BLOCK_SIZE_M_SECOND: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Three scan consumers on one grid, split by program id.

    Two label halves (one per granularity) binary-search their own
    ``block_cumulative`` and fill their own padding; the pair half derives
    the shared key ONCE, then claims a slot from EACH granularity's cursor.
    Both plans' key/validity semantics therefore cannot diverge — there is
    exactly one derivation.  All stores land in per-granularity buffers that
    no half bulk-reads (the documented in-place-store hazard).
    """
    pid = tl.program_id(0)
    if pid < first_label_programs:
        if USE_PDL:
            # The label path immediately consumes scan outputs.
            tl.extra.cuda.gdc_wait()
        _dual_granularity_label_blocks(
            pid,
            first_block_cumulative_ptr,
            first_bucket_end_ptr,
            first_sorted_ptr,
            first_block_ids_ptr,
            num_first_blocks,
            num_pairs,
            NUM_BUCKETS=NUM_BUCKETS,
            NUM_VIRTUAL_EXPERTS=NUM_VIRTUAL_EXPERTS,
            BLOCK=BLOCK,
            BLOCK_SIZE_M=BLOCK_SIZE_M_FIRST,
            SEARCH_STEPS=SEARCH_STEPS,
        )
        return
    if pid < first_label_programs + second_label_programs:
        if USE_PDL:
            # The label path immediately consumes scan outputs.
            tl.extra.cuda.gdc_wait()
        _dual_granularity_label_blocks(
            pid - first_label_programs,
            second_block_cumulative_ptr,
            second_bucket_end_ptr,
            second_sorted_ptr,
            second_block_ids_ptr,
            num_second_blocks,
            num_pairs,
            NUM_BUCKETS=NUM_BUCKETS,
            NUM_VIRTUAL_EXPERTS=NUM_VIRTUAL_EXPERTS,
            BLOCK=BLOCK,
            BLOCK_SIZE_M=BLOCK_SIZE_M_SECOND,
            SEARCH_STEPS=SEARCH_STEPS,
        )
        return

    pair_ids = (pid - first_label_programs - second_label_programs) * BLOCK + tl.arange(
        0, BLOCK
    )
    pair_mask = pair_ids < num_pairs
    virtual_ids = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_slots_ptr,
        topk_ids_ptr,
        pair_ids,
        pair_mask,
        0,
        LORA_EXPERTS_PER_ADAPTER=LORA_EXPERTS_PER_ADAPTER,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        USE_LORA_EXPERT_MAP=False,
        SHARED_OUTER=False,
    )
    buckets = tl.where(virtual_ids < 0, NUM_BUCKETS - 1, virtual_ids)
    if USE_PDL:
        # Key recomputation above is independent of the scan. Cursors below
        # are the first scan-produced values this path consumes.
        tl.extra.cuda.gdc_wait()
    first_slots = tl.atomic_add(first_cursor_ptr + buckets, 1, mask=pair_mask)
    tl.store(first_sorted_ptr + first_slots, pair_ids, mask=pair_mask)
    second_slots = tl.atomic_add(second_cursor_ptr + buckets, 1, mask=pair_mask)
    tl.store(second_sorted_ptr + second_slots, pair_ids, mask=pair_mask)


def _validate_dual_granularity_inputs(
    *,
    scratches: tuple[FusedAlignScratch, FusedAlignScratch],
    num_pairs_post_padded_outs: tuple[torch.Tensor, torch.Tensor],
    num_buckets: int,
    device: torch.device,
) -> None:
    """Caller-owned metadata contract, checked before the first launch.

    Same tensor contract as the single-granularity fused builder, plus
    disjoint storage: each granularity's scan destructively re-zeroes its counts
    and claims its cursors, so aliased scratch would double-count in the
    histogram and corrupt both plans silently.
    """
    for index, (scratch, padded) in enumerate(
        zip(scratches, num_pairs_post_padded_outs)
    ):
        contracts = (
            ("counts", scratch.counts, (num_buckets,)),
            ("block_cumulative", scratch.block_cumulative, (num_buckets + 1,)),
            ("cursor", scratch.cursor, (num_buckets,)),
            ("bucket_end", scratch.bucket_end, (num_buckets,)),
            ("num_pairs_post_padded_out", padded, (1,)),
        )
        for name, tensor, shape in contracts:
            if (
                tensor.shape != shape
                or tensor.dtype is not torch.int32
                or tensor.device != device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"dual-granularity route {index} {name} must be a "
                    f"contiguous int32 {list(shape)} tensor on {device}"
                )
    pointer_sets = [
        {
            scratch.counts.data_ptr(),
            scratch.block_cumulative.data_ptr(),
            scratch.cursor.data_ptr(),
            scratch.bucket_end.data_ptr(),
            padded.data_ptr(),
        }
        for scratch, padded in zip(scratches, num_pairs_post_padded_outs)
    ]
    if pointer_sets[0] & pointer_sets[1]:
        raise ValueError(
            "dual-granularity aligned routes need disjoint scratch per "
            "granularity; sharing metadata would double-count the histogram"
        )


def build_dual_granularity_aligned_routes(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    lora_experts_per_adapter: int,
    max_loras: int,
    block_sizes: tuple[int, int],
    num_pairs_post_padded_outs: tuple[torch.Tensor, torch.Tensor],
    scratches: tuple[FusedAlignScratch, FusedAlignScratch],
    use_pdl: bool | None = None,
) -> tuple[RouteView, RouteView]:
    """Build two per-expert ALIGNED views over the same pairs in one pass.

    Returns one :class:`RouteView` per entry of ``block_sizes``, both
    satisfying exactly the plan contract of a standalone fused aligned build
    at that granularity (deterministic block labels, padding, and padded
    count; intra-bucket scatter order nondeterministic as always).  Scratch
    and the retained padded-count scalars are caller-owned — production hands
    in per-route ``MoeLoraWorkspace`` tensors so the pass is CUDA-graph
    stable — and there is no serial fallback cache: this builder exists only
    for the retained dual-granularity route bundle.

    Per-expert ownership only.  The one shipping consumer is grouped
    per-expert gate/up-A against the canonical per-expert route; shared-outer
    or expert-mapped duals have no consumer and would multiply the constexpr
    surface (the joint shared-outer builder already covers mixed ownership).

    ``use_pdl`` keeps the standard-route contract: ``None`` and ``False``
    both mean off, explicit ``True`` forms the K1->K2->K3 edges exactly like
    the single-granularity fused builder.
    """
    if topk_ids.ndim != 2 or token_lora_mapping.shape != (topk_ids.shape[0],):
        raise ValueError("expected topk_ids [T,K] and token_lora_mapping [T]")
    if (
        len(block_sizes) != 2
        or len(num_pairs_post_padded_outs) != 2
        or len(scratches) != 2
    ):
        raise ValueError(
            "dual-granularity routes take exactly two block sizes, two "
            "padded-count outputs, and two scratch sets"
        )
    if min(lora_experts_per_adapter, max_loras, *block_sizes) <= 0:
        raise ValueError(
            "LoRA experts per adapter, adapter capacity, and both block "
            "sizes must all be positive"
        )
    device = topk_ids.device
    num_pairs = topk_ids.numel()
    top_k = topk_ids.shape[1]
    num_virtual = lora_experts_per_adapter * max_loras
    num_buckets = num_virtual + 1
    capacities = tuple(
        _routing_capacity(num_pairs, block_size, num_virtual)
        for block_size in block_sizes
    )
    # int32 key math holds only below 2**31 (same bound as the single fused
    # builder); a wrapped key would silently land in a valid-looking bucket.
    if num_buckets >= 2**31 or max(capacities) >= 2**31:
        raise ValueError(
            f"dual-granularity align uses int32 plan math: num_buckets="
            f"{num_buckets} and capacities {capacities} must all be < 2**31"
        )
    _validate_dual_granularity_inputs(
        scratches=scratches,
        num_pairs_post_padded_outs=num_pairs_post_padded_outs,
        num_buckets=num_buckets,
        device=device,
    )
    use_pdl = bool(use_pdl)
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
    num_blocks = tuple(
        capacity // block_size for capacity, block_size in zip(capacities, block_sizes)
    )
    sorted_pair_ids = tuple(
        torch.empty(capacity, dtype=torch.int32, device=device)
        for capacity in capacities
    )
    block_virtual_expert_ids = tuple(
        torch.empty(blocks, dtype=torch.int32, device=device) for blocks in num_blocks
    )

    _dual_granularity_hist_kernel[(triton.cdiv(max(num_pairs, 1), _DUAL_HIST_BLOCK),)](
        topk_ids,
        token_lora_mapping,
        scratches[0].counts,
        scratches[1].counts,
        num_pairs,
        NUM_BUCKETS=num_buckets,
        LORA_EXPERTS_PER_ADAPTER=lora_experts_per_adapter,
        MAX_LORAS=max_loras,
        TOP_K=top_k,
        BLOCK=_DUAL_HIST_BLOCK,
        USE_PDL=use_pdl,
        num_warps=_DUAL_HIST_WARPS,
    )
    _dual_granularity_scan_kernel[(2,)](
        scratches[0].counts,
        scratches[0].block_cumulative,
        scratches[0].cursor,
        scratches[0].bucket_end,
        num_pairs_post_padded_outs[0],
        scratches[1].counts,
        scratches[1].block_cumulative,
        scratches[1].cursor,
        scratches[1].bucket_end,
        num_pairs_post_padded_outs[1],
        num_buckets,
        BLOCK_SIZE_M_FIRST=block_sizes[0],
        BLOCK_SIZE_M_SECOND=block_sizes[1],
        CHUNK=_DUAL_SCAN_CHUNK,
        USE_PDL=use_pdl,
        num_warps=_DUAL_SCAN_WARPS,
        **pdl_kwargs,
    )
    label_programs = tuple(
        triton.cdiv(max(blocks, 1), _DUAL_EXPAND_BLOCK) for blocks in num_blocks
    )
    pair_programs = triton.cdiv(max(num_pairs, 1), _DUAL_EXPAND_BLOCK)
    _dual_granularity_expand_scatter_kernel[
        (label_programs[0] + label_programs[1] + pair_programs,)
    ](
        topk_ids,
        token_lora_mapping,
        scratches[0].cursor,
        scratches[0].bucket_end,
        scratches[0].block_cumulative,
        sorted_pair_ids[0],
        block_virtual_expert_ids[0],
        num_blocks[0],
        label_programs[0],
        scratches[1].cursor,
        scratches[1].bucket_end,
        scratches[1].block_cumulative,
        sorted_pair_ids[1],
        block_virtual_expert_ids[1],
        num_blocks[1],
        label_programs[1],
        num_pairs,
        NUM_BUCKETS=num_buckets,
        NUM_VIRTUAL_EXPERTS=num_virtual,
        LORA_EXPERTS_PER_ADAPTER=lora_experts_per_adapter,
        MAX_LORAS=max_loras,
        TOP_K=top_k,
        BLOCK=_DUAL_EXPAND_BLOCK,
        BLOCK_SIZE_M_FIRST=block_sizes[0],
        BLOCK_SIZE_M_SECOND=block_sizes[1],
        # The search interval is [0, NUM_BUCKETS] — NUM_BUCKETS + 1 states —
        # so a power-of-two bucket count needs the extra iteration
        # (num_buckets - 1).bit_length() would drop; see fused_align.py.
        SEARCH_STEPS=max(1, num_buckets.bit_length()),
        USE_PDL=use_pdl,
        num_warps=_DUAL_EXPAND_WARPS,
        **pdl_kwargs,
    )

    def _view(index: int) -> RouteView:
        return RouteView(
            view=ROUTE_ALIGNED,
            num_virtual_experts=num_virtual,
            block_size=block_sizes[index],
            topk_ids=topk_ids,
            token_slots=token_lora_mapping,
            lora_experts_per_adapter=lora_experts_per_adapter,
            max_loras=max_loras,
            maybe_sorted_pair_ids=sorted_pair_ids[index],
            maybe_block_virtual_expert_ids=block_virtual_expert_ids[index],
            maybe_num_pairs_post_padded=num_pairs_post_padded_outs[index],
        )

    return _view(0), _view(1)
