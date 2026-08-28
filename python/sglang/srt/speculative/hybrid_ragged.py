"""hybrid retrieval ragged (variable-length) verify -- the compact-forward seam.

The hybrid chain (``hybrid_chain.py``) is extremely bimodal in how many verify
columns it actually needs: a request whose retrieval graft FAILED only needs the
``1 + num_steps`` MTP columns, while a request whose graft SUCCEEDED wants the
full ``L`` width. Padding every request to ``L`` burns ~40% of the verify tokens
on columns that are known-dead before the forward even starts.

This module makes the target-verify forward *ragged*: request ``r`` contributes
only ``w_r`` rows, packed back to back. Everything is organised around ONE seam,
mirroring DSPARK (``dspark_verify.py:run_compact``)::

    1. build_hybrid_ragged_layout   -> per-request w_r + a RaggedVerifyLayout
    2. build_hybrid_ragged_window   -> compact verify_ids / positions / cache_loc
    3. target forward                  compact logits/hidden [total_tokens, ...]
    4. scatter_hybrid_verify_outputs-> fixed stride  [bs * L, ...]     <-- SEAM
    5. everything downstream (accept, retrieve_index, scheduler stride,
       draft-extend, grammar, logprob) sees the SAME [bs, L] rectangle it sees
       today and is byte-for-byte unchanged.

Only one piece of NEW semantics is required past the seam: the columns beyond
``w_r`` were never computed, so they are filled with ``fill_value`` and MUST NOT
be acceptable. That is :func:`cap_accept_to_verify_lens`, a straight port of
DSPARK's ``CapCorrectLen``. This is not defensive programming -- the padded
logits are zeros so ``argmax`` is token 0, the padded candidate columns are also
0, and ``0 == 0`` chains all the way to ``L``. Without the cap the failure is
systematic, not occasional.

Graph-tier mode adds token-tier CUDA graphs and DP cross-rank tier agreement:

  * **The tier grid.** A captured graph has ONE token count, so the step's
    ``sum(w_r)`` is rounded UP to a captured tier. The substrate's default grid
    is ``{bs * L}``, whose spacing is ``L`` while the real totals move in
    quanta of ``L - floor``. A quantum-aligned grid therefore preserves more of
    the ragged saving, so hybrid retrieval builds its own.
  * **Two length vectors.** ``sum(w_r)`` rarely lands exactly on a tier, and the
    substrate disposes of the difference by inflating the LAST REAL request's
    verify_len (``pad_verify_lens_to_bucket``, the ``num_pad_reqs == 0`` branch)
    -- which can push a real request past ``L`` and make the attention read KV
    slots the scheduler never reserved. Hybrid retrieval instead spends the slack
    itself, spread across requests and capped at ``L`` each, so the substrate's
    leftover is zero and nothing is ever inflated out of its reservation. The
    price is that the forward now computes columns that are real geometry but
    hold zero-padded candidates, so the accept cutoff must use the ORIGINAL
    widths: hence ``accept_verify_lens`` (true ``w_r``, drives the cutoff and
    the length contract) alongside ``layout.verify_lens`` (padded, drives the
    forward geometry).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import msgspec
import torch

from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

# The compact-window / scatter / cutoff kernels are algorithm-agnostic but live
# under dspark_components, whose package root pulls in the whole DFlash+DSPARK
# stack. Import them inside the call sites so an EAGLE-only deployment does not
# pay for that at import time (and so this module cannot participate in an
# import cycle through eagle_worker_common).

# Padded logits/hidden rows never reach a consumer: the accept cap keeps every
# read inside [0, w_r), and draft-extend gathers by accept_index. 0.0 matches
# what DSPARK fills so the two paths stay comparable under a debugger.
RAGGED_FILL_VALUE = 0.0


def compute_hybrid_verify_lens(
    *,
    num_keep_drafts: torch.Tensor,
    graft_ok: torch.Tensor,
    num_retrieval_tokens: torch.Tensor,
    num_steps: int,
    verify_width: int,
) -> torch.Tensor:
    """Per-request verify width ``w_r`` for one hybrid step -- ``[bs]`` int32.

    This is the device-side twin of what ``splice_hybrid_chain`` laid out, so
    the two MUST agree column for column. The spliced chain is::

        column 0                      the bonus token (always present)
        columns 1 .. effective_keep   MTP drafts
        columns effective_keep+1 ..   the retrieval continuation, but only when
                                      the agreement gate passed; zero pad
                                      otherwise

    with ``effective_keep = num_keep_drafts`` on a successful graft and
    ``num_steps`` (the full, untruncated MTP chain) on a failed one. So::

        graft failed:   w_r = 1 + num_steps                       (the floor)
        graft ok:       w_r = 1 + min(num_retrieval_tokens, L-1)

    The lower bound ``w_r >= 1 + num_keep_drafts`` holds for free: a successful
    graft means ``num_agree_drafts >= num_keep_drafts``, and agreement is only
    possible on columns the retrieval chain actually carries, hence
    ``num_retrieval_tokens >= num_keep_drafts``. In other words ragged NEVER
    verifies fewer tokens than the kept MTP prefix -- the worst case degrades to
    plain MTP, never to something shorter.

    ``num_retrieval_tokens`` is the retriever's OWN reported continuation length
    (``HybridRetriever`` returns it alongside the chain). It must not be derived
    by counting non-zero entries: token id 0 is a real vocabulary token in DSV4
    (``<|begin_of_sentence|>``, present in the trie because the full prompt is
    indexed), so a zero inside a chain is indistinguishable from tail padding.
    Calibration data contained real token-0 chain positions. They are rare, but
    treating them as padding would silently shorten ``w_r``.

    The ``max(..., 1 + num_keep_drafts)`` at the end is defence in depth for
    exactly that class of bug: whatever the length source says, ``w_r`` can never
    come out shorter than the MTP prefix that was actually spliced in.
    """
    keep = torch.where(
        graft_ok,
        torch.minimum(
            num_retrieval_tokens.to(num_keep_drafts.dtype),
            torch.full_like(num_keep_drafts, verify_width - 1),
        ),
        torch.full_like(num_keep_drafts, num_steps),
    )
    keep = torch.maximum(keep, num_keep_drafts)
    return (keep + 1).clamp_(1, verify_width).to(torch.int32)


def build_hybrid_ragged_layout(
    *,
    verify_lens_cpu: Sequence[int],
    device: str | torch.device,
    tier_num_tokens: Optional[int] = None,
) -> RaggedVerifyLayout:
    """Wrap the per-request FORWARD widths in the substrate's layout.

    ``verify_lens_cpu`` is the forward geometry (``forward_verify_lens`` from
    :func:`plan_hybrid_ragged_tier`), not the accept widths -- see the module
    docstring on why the two differ.

    ``tier_num_tokens`` is the captured graph's token count. Without it (no
    CUDA graph: eager mode, or ``--disable-cuda-graph``) the layout is EXACT, i.e.
    ``graph_num_tokens == sum(w_r)`` and ``tier_slack == 0``; the singleton grid
    satisfies the substrate's ``round_up_grid`` contract without special-casing
    it. Passing a tier is what makes the batch replayable on the token-keyed
    graph captured at that tier.
    """
    verify_lens_list = [int(v) for v in verify_lens_cpu]
    total = sum(verify_lens_list)
    tier = total if tier_num_tokens is None else int(tier_num_tokens)
    assert tier >= total, (
        f"hybrid ragged tier {tier} < sum(forward verify_lens) {total}: the "
        "compact window would overflow the captured graph's token buffers"
    )
    return RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=verify_lens_list,
        device=device,
        grid=[tier],
    )


# compact_row_index_triton resolves the row -> request mapping with a FIXED
# 11-round binary search (`_SEARCH_NBITS`, dspark_verify_window.py:219), which
# covers at most 2048 boundaries and fails SILENTLY above that (the torch
# reference has no such limit, so the two implementations would diverge). Half
# that, as headroom; the real batches here are <= 32.
MAX_RAGGED_BS = 1024


class HybridRaggedWindow(msgspec.Struct, frozen=True):
    """One ragged target-verify forward's inputs.

    The first three are COMPACT (length ``layout.graph_num_tokens``) and feed the
    forward. ``fixed_cache_loc`` is the ordinary ``[bs * L]`` map that every
    stage AFTER the scatter seam still expects -- see the class docstring note
    below and ``run_eagle_verify``.
    """

    verify_ids: torch.Tensor
    positions: torch.Tensor
    verify_cache_loc: torch.Tensor
    fixed_cache_loc: torch.Tensor


def build_hybrid_ragged_window(
    *,
    batch,
    layout: RaggedVerifyLayout,
    draft_token: torch.Tensor,
    verify_width: int,
    req_to_token: torch.Tensor,
    device: str | torch.device,
) -> HybridRaggedWindow:
    """Pack the ``[bs, L]`` verify rectangle down to ``sum(w_r)`` live rows.

    ``draft_token`` stays the FIXED-STRIDE ``[bs * L]`` chain that
    ``build_tree_kernel_efficient`` produced -- it is what ``eagle_sample``
    reshapes into ``candidates`` after the seam, so it must not be compacted.
    Only the forward's inputs are compacted here.

    **The cache map is built FIXED-WIDTH FIRST and the compact one is gathered
    out of it.** That is deliberate, on two counts:

    1. The fixed map is byte-for-byte the same tensor today's fixed-width path
       builds (same kernel, same ``end_offset = seq_lens + L``), so ragged does
       not introduce a second KV-slot allocation path. The compact map is a pure
       subset of it.
    2. The fixed map is REQUIRED after the forward. ``prepare_for_draft_extend``
       (``eagle_worker_common.py``, the non-widening branch) deliberately does
       NOT rebuild ``batch.out_cache_loc``: on the fixed-width path the verify
       map and the draft-extend map are the same ``bs * L`` slots, so it just
       reuses whatever verify left behind. Handing that stage a compact map
       leaves it with ``bs * L`` tokens and only ``sum(w_r)`` locations. So the
       scatter seam restores this one; see ``run_eagle_verify``.

    Unlike DSPARK's ``compact_verify_ids`` (which splices a separate anchor
    tensor with a per-block draft matrix) the hybrid chain already carries its
    anchor in column 0, so the gather is a plain 2-D index. Reusing DSPARK's
    kernel would mean passing ``draft_block_ids`` with a row stride that its
    triton kernel derives from ``draft_tokens.shape[1]``; the strides differ and
    the mismatch is silent, so gather directly instead.
    """
    from sglang.srt.speculative.dspark_components.kernels.dspark_verify_window import (
        CompactRowIndex,
    )

    bs = layout.bs
    assert bs <= MAX_RAGGED_BS, (
        f"hybrid ragged verify: bs={bs} exceeds MAX_RAGGED_BS={MAX_RAGGED_BS}. "
        "compact_row_index_triton resolves the row->request mapping with a fixed "
        "11-round binary search and fails SILENTLY past 2048 boundaries; raise "
        "_SEARCH_NBITS in dspark_verify_window.py before lifting this."
    )
    total = layout.graph_num_tokens
    req_id, within, valid = CompactRowIndex.execute(
        verify_lens=layout.verify_lens, padded_total=total, device=device
    )
    safe_req = req_id.clamp(max=bs - 1)

    prefix_lens = batch.seq_lens
    positions = torch.where(
        valid,
        prefix_lens.to(torch.int64)[safe_req] + within,
        torch.zeros_like(within),
    )

    # Identical to the fixed-width path's own call: the L slots at
    # [seq_len, seq_len + L) were already reserved by the scheduler, this only
    # reads them out of req_to_token.
    fixed_cache_loc = assign_extend_cache_locs_func(
        req_pool_indices=batch.req_pool_indices,
        req_to_token=req_to_token,
        start_offset=prefix_lens,
        end_offset=prefix_lens + verify_width,
        batch_size=bs,
        draft_token_num=verify_width,
        device=device,
    )
    fixed_2d = fixed_cache_loc.view(bs, verify_width)
    verify_cache_loc = fixed_2d[safe_req, within]
    verify_cache_loc = torch.where(
        valid, verify_cache_loc, torch.zeros_like(verify_cache_loc)
    )

    draft_token_2d = draft_token.view(bs, verify_width)
    verify_ids = draft_token_2d[safe_req, within]
    verify_ids = torch.where(valid, verify_ids, torch.zeros_like(verify_ids))

    return HybridRaggedWindow(
        verify_ids=verify_ids.to(torch.int64),
        positions=positions,
        verify_cache_loc=verify_cache_loc,
        fixed_cache_loc=fixed_cache_loc,
    )


def scatter_hybrid_verify_outputs(
    *,
    logits_output,
    layout: RaggedVerifyLayout,
    verify_width: int,
) -> None:
    """THE SEAM. Re-inflate the compact forward outputs to the ``[bs * L, ...]``
    fixed stride, in place on ``logits_output``.

    Ordering is a HARD constraint, not a style choice: ``eagle_sample`` sizes its
    ``predict`` buffer from ``next_token_logits.shape[:-1]``
    (``eagle_utils.py:641``) while ``retrieve_index`` holds global row ids up to
    ``bs * L - 1``. If the verify kernel ran against a compact
    ``[graph_num_tokens]`` predict buffer it would write OUT OF BOUNDS. So this
    must run immediately after the forward and before anything reads
    ``logits_output``.
    """
    from sglang.srt.speculative.dspark_components.kernels.dspark_verify_window import (
        ScatterCompactToStrided,
    )

    compact_logits = logits_output.next_token_logits
    logits_output.next_token_logits = ScatterCompactToStrided.execute(
        compact=compact_logits,
        layout=layout,
        fill_value=RAGGED_FILL_VALUE,
        verify_num_draft_tokens=verify_width,
    )
    compact_hidden = logits_output.hidden_states
    if compact_hidden is not None:
        logits_output.hidden_states = ScatterCompactToStrided.execute(
            compact=compact_hidden,
            layout=layout,
            fill_value=RAGGED_FILL_VALUE,
            verify_num_draft_tokens=verify_width,
        )


def cap_accept_to_verify_lens(
    *,
    num_correct_drafts: torch.Tensor,
    accept_index: torch.Tensor,
    verify_lens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forbid accepting a column the forward never computed (DSPARK's
    ``CapCorrectLen``, ``dspark_accept.py:816``).

    Request ``r`` computed columns ``[0, w_r)``, so the last acceptable DRAFT
    index is ``w_r - 1`` and ``num_correct_drafts`` is clamped there. The
    accepted chain then spans columns ``0 .. num_correct_drafts``, all real, and
    the bonus token (``predict`` at the last accepted column) is real too.

    ``accept_index`` is truncated rather than rebuilt from ``arange``. For a
    linear chain the two are identical -- ``accept_index[b, i] == b * L + i`` --
    but truncating preserves whatever the verify kernel actually produced, so a
    future tree-shaped chain cannot silently get an arange grafted onto it.
    """
    from sglang.srt.speculative.dspark_components.kernels.dspark_accept import (
        CapCorrectLen,
    )

    capped, _cap_trim_lens = CapCorrectLen.execute(
        correct_len=num_correct_drafts, verify_lens=verify_lens
    )
    capped = capped.to(num_correct_drafts.dtype)
    depth = accept_index.shape[1]
    col = torch.arange(depth, device=accept_index.device)
    keep = col[None, :] <= capped.to(torch.int64)[:, None]
    accept_index = torch.where(keep, accept_index, torch.full_like(accept_index, -1))
    return capped, accept_index


# =============================================================================
# graph-tier mode: graph tiers
# =============================================================================

# Gathered value meaning "this rank refuses ragged for this step" (non-greedy
# batch, no captured grid, a total the grid cannot hold, ...). Any rank vetoing
# takes the WHOLE dp group back to the fixed-width path for that step, because
# a per-rank decision would have the ranks replay differently-shaped graphs.
# Mirrors DSPARK's sentinel (`dp_global_verify_tier_num_tokens`: `any(t < 0)`
# -> None -> pinned to the full rectangle).
HYBRID_RAGGED_TIER_VETO = -1


class HybridRaggedTierParams(msgspec.Struct, frozen=True):
    floor: int
    tier_step: int
    max_tiers: int
    explicit_tiers: Tuple[int, ...]


def resolve_hybrid_ragged_tier_params(
    *, server_args, verify_width: int, capture_bs: Sequence[int]
) -> HybridRaggedTierParams:
    """Turn the ``--speculative-hybrid-ragged-*`` args into grid parameters.

    The two derived defaults are the measured shape of the hybrid chain, not
    tuning knobs picked by feel:

      ``floor = num_steps + 1``   the width a graft-FAILED request needs (the
                                  bonus token plus the full MTP chain), which is
                                  also every request's lower bound
      ``tier_step = width - floor`` one QUANTUM: the extra tokens a request costs
                                  when its graft succeeds, i.e. the spacing the
                                  step totals actually move on

    Both are overridable so the grid can be reproduced from the command line
    without touching code.
    """
    floor = server_args.speculative_hybrid_ragged_floor
    if floor is None:
        floor = int(server_args.speculative_num_steps) + 1
    tier_step = server_args.speculative_hybrid_ragged_tier_step
    if tier_step is None:
        tier_step = max(verify_width - int(floor), 1)
    raw_tiers = (server_args.speculative_hybrid_ragged_tiers or "").strip()
    explicit = tuple(int(piece) for piece in raw_tiers.split(",") if piece.strip())
    max_tiers = int(server_args.speculative_hybrid_ragged_max_tiers)
    if max_tiers < 0:
        # AUTO: at most twice the number of verify graphs the fixed-width path
        # would capture, floor 16. The floor keeps the small deployments the
        # measurements came from (max_running=4 -> capture_bs [1,2,3,4] -> 14
        # tiers) on the exact grid, while the ratio stops a large
        # --cuda-graph-max-bs from turning a near-dense union into thousands of
        # captures. `0` still means genuinely unlimited, for sweeps.
        max_tiers = max(16, 2 * len(capture_bs))
    return HybridRaggedTierParams(
        floor=int(floor),
        tier_step=int(tier_step),
        max_tiers=max_tiers,
        explicit_tiers=explicit,
    )


class HybridRaggedTierGrid(msgspec.Struct, frozen=True):
    """The captured tier grid plus what coarsening it took to get there.

    The diagnostics are not decoration: with an unbounded union running to
    thousands of tiers, "which knob fired and how far" is the only way to read a
    startup log and tell whether the grid is quantum-aligned or a coarsened
    fallback.
    """

    tiers: List[int]
    effective_step: int
    anchor_level: int
    num_anchors: int
    num_capture_bs: int
    max_tiers: int

    @property
    def anchors_thinned(self) -> bool:
        return self.anchor_level > 0 or self.num_anchors < self.num_capture_bs

    def step_widened(self, requested_step: int) -> bool:
        return self.effective_step > requested_step

    def coarsened(self, requested_step: int) -> bool:
        """Whether the grid is NOT the requested quantum-aligned one.

        Takes the requested step because widening it is the FIRST coarsening
        lever and the more common one -- a property that only looked at anchor
        thinning reported `coarsened=False` on the H100 default, where the step
        had in fact gone from 5 to 2080. Any caller deciding whether it got the
        requested grid needs both.
        """
        return self.anchors_thinned or self.step_widened(requested_step)

    def describe(self) -> str:
        return (
            f"{len(self.tiers)} tiers, effective_step={self.effective_step}, "
            f"anchors={self.num_anchors}/{self.num_capture_bs} "
            f"(thin_level={self.anchor_level})"
        )

    def describe_coarsening(self, requested_step: int) -> str:
        if not self.coarsened(requested_step):
            return "not coarsened"
        parts = []
        if self.step_widened(requested_step):
            parts.append(f"step {requested_step} -> {self.effective_step}")
        if self.anchors_thinned:
            parts.append(f"anchors {self.num_capture_bs} -> {self.num_anchors}")
        return ", ".join(parts)


def build_hybrid_ragged_tier_grid(
    *,
    capture_bs: Sequence[int],
    verify_width: int,
    floor: int,
    tier_step: int,
    max_tiers: int = 0,
    explicit_tiers: Sequence[int] = (),
) -> List[int]:
    """The tier list alone; see :func:`build_hybrid_ragged_tier_plan`."""
    return build_hybrid_ragged_tier_plan(
        capture_bs=capture_bs,
        verify_width=verify_width,
        floor=floor,
        tier_step=tier_step,
        max_tiers=max_tiers,
        explicit_tiers=explicit_tiers,
    ).tiers


def build_hybrid_ragged_tier_plan(
    *,
    capture_bs: Sequence[int],
    verify_width: int,
    floor: int,
    tier_step: int,
    max_tiers: int = 0,
    explicit_tiers: Sequence[int] = (),
) -> HybridRaggedTierGrid:
    """The captured token grid, aligned to the hybrid chain's own quantum.

    A step's total is ``Σ_r w_r`` where each ``w_r`` is either ``floor``
    (``1 + num_steps``, the graft-failed width) or up to ``verify_width``. So
    per request the total moves by one QUANTUM of ``verify_width - floor``, and
    for ``bs`` requests the reachable totals cluster on
    ``bs * floor + j * quantum``. That is the grid this builds::

        grid = { bs * floor + j * tier_step } ∪ { bs * verify_width }
               for every captured bs, sorted and deduplicated

    ``tier_step`` defaults to one quantum (see ``ServerArgs``); a larger step
    trades saving for fewer captured graphs. The substrate's own default grid is
    ``{bs * verify_width}``, i.e. spacing ``verify_width`` against a signal that
    moves in ``verify_width - floor`` -- which is why it collects almost none of
    the available saving.

    ``max(capture_bs) * verify_width`` is always the LAST tier: it is the grid's
    ceiling, and a step total above the ceiling has no graph to replay.

    **Bounding the size matters more than it looks.** The union runs over every
    captured batch size, and ``bs * floor mod tier_step`` walks every residue as
    ``bs`` varies, so an unbounded union is very nearly one tier per integer:
    at the H100 default ``--cuda-graph-max-bs 512`` the naive union is thousands
    of tiers against ~67 fixed-width graphs. Every tier is a captured graph, and
    while they share the graph memory pool (so the high-water mark does not move)
    each one still costs a warmup + capture at startup. So ``max_tiers``
    coarsens, in this order:

      1. **widen the step**, up to ``max(anchors) * (verify_width - floor)``.
         Past that width every per-bs series has collapsed to its two endpoints
         and the grid is the SKELETON ``{bs*floor} ∪ {bs*verify_width}``.
      2. only if that still does not fit, **thin the anchor batch sizes**.

    That order is deliberate and is the opposite of what looks right. Widening
    the step costs intra-bs resolution, but the skeleton it leaves behind is
    dense in absolute terms -- 67 anchors give 134 tiers spread over the whole
    range, and a bs=4 step needing 21 tokens still lands on 24. Dropping anchors
    instead punches HOLES: thinning 67 anchors down to {1, 512} leaves the grid
    ``[4, 9, 2048, 2068, ...]``, and that same 21-token step rounds up to 2048 --
    a 97x blow-up. Resolution inside one batch size is worth much less than
    having any tier at all near the batch sizes actually in flight.

    Thinning is therefore the last resort, and when it does run it keeps a
    MULTIPLICATIVELY spread subset (see :func:`_thin_anchor_batch_sizes`) so the
    relative gap it introduces stays bounded instead of collapsing to the two
    endpoints. With the auto ``max_tiers`` (``max(16, 2 * len(capture_bs))``) the
    skeleton always fits and thinning never runs at all; only an explicitly small
    ``--speculative-hybrid-ragged-max-tiers`` reaches it.

    If even the coarsest combination exceeds ``max_tiers`` the grid is returned
    as is and the caller warns. Truncating instead would drop the ceiling and
    strand the largest batches with no graph.
    """
    num_capture_bs = len(set(int(bs) for bs in capture_bs))
    if explicit_tiers:
        tiers = _validate_hybrid_ragged_tier_grid(
            tiers=[int(t) for t in explicit_tiers],
            capture_bs=capture_bs,
            verify_width=verify_width,
            source="--speculative-hybrid-ragged-tiers",
        )
        return HybridRaggedTierGrid(
            tiers=tiers,
            effective_step=0,
            anchor_level=0,
            num_anchors=num_capture_bs,
            num_capture_bs=num_capture_bs,
            max_tiers=max_tiers,
        )

    assert 1 <= floor <= verify_width, (
        f"hybrid ragged floor must be in [1, verify_width={verify_width}], got "
        f"{floor}"
    )
    assert tier_step >= 1, f"hybrid ragged tier_step must be >= 1, got {tier_step}"

    anchors_full = sorted(set(int(bs) for bs in capture_bs))
    coarsest_step = max(max(anchors_full) * (verify_width - floor), 1)
    level = 0
    while True:
        anchors = _thin_anchor_batch_sizes(anchors_full, level=level)
        step = tier_step
        while True:
            tiers = _hybrid_ragged_tier_series(
                capture_bs=anchors,
                verify_width=verify_width,
                floor=floor,
                tier_step=step,
            )
            fits = max_tiers <= 0 or len(tiers) <= max_tiers
            exhausted = step >= coarsest_step and len(anchors) <= 2
            if fits or exhausted:
                return HybridRaggedTierGrid(
                    tiers=_validate_hybrid_ragged_tier_grid(
                        tiers=tiers,
                        capture_bs=capture_bs,
                        verify_width=verify_width,
                        source="derived grid",
                    ),
                    effective_step=step,
                    anchor_level=level,
                    num_anchors=len(anchors),
                    num_capture_bs=num_capture_bs,
                    max_tiers=max_tiers,
                )
            if step >= coarsest_step:
                break
            step += tier_step
        level += 1


def _thin_anchor_batch_sizes(anchors: List[int], *, level: int) -> List[int]:
    """Keep a MULTIPLICATIVELY spread subset of the captured batch sizes.

    Spread by ratio, not by list index. Index striding looks equivalent but is
    not: ``capture_bs`` is itself geometric-ish, so a large index stride keeps
    only the two endpoints and leaves a hole spanning most of the range -- and a
    hole is far more expensive than a coarse step, because a batch landing in it
    rounds up by a multiple rather than by a few tokens.

    Level ``k`` keeps anchors at least ``2 ** k`` apart in ratio, so the relative
    gap this introduces is bounded by that factor no matter how many anchors are
    dropped. The first and last are always kept: the last sets the grid's
    ceiling, and small batches are where most of the measured saving occurs.
    """
    if level <= 0:
        return list(anchors)
    ratio = float(2**level)
    kept = [anchors[0]]
    for bs in anchors[1:]:
        if bs >= kept[-1] * ratio:
            kept.append(bs)
    if kept[-1] != anchors[-1]:
        kept.append(anchors[-1])
    return kept


def _hybrid_ragged_tier_series(
    *,
    capture_bs: Sequence[int],
    verify_width: int,
    floor: int,
    tier_step: int,
) -> List[int]:
    tiers = set()
    for bs in capture_bs:
        lo = bs * floor
        hi = bs * verify_width
        tier = lo
        while tier < hi:
            tiers.add(tier)
            tier += tier_step
        tiers.add(hi)
    return sorted(tiers)


def _validate_hybrid_ragged_tier_grid(
    *,
    tiers: List[int],
    capture_bs: Sequence[int],
    verify_width: int,
    source: str,
) -> List[int]:
    if not tiers:
        raise ValueError(f"hybrid ragged tier grid is empty ({source})")
    if sorted(set(tiers)) != tiers:
        raise ValueError(
            f"hybrid ragged tier grid must be strictly increasing and unique "
            f"({source}); got {tiers}"
        )
    if tiers[0] < 1:
        raise ValueError(
            f"hybrid ragged tier grid must be positive ({source}); got {tiers}"
        )
    ceiling = max(capture_bs) * verify_width
    if tiers[-1] != ceiling:
        raise ValueError(
            f"hybrid ragged tier grid must top out at max(capture_bs) * "
            f"verify_width = {ceiling} ({source}); got {tiers[-1]}. round_up_grid "
            "raises on any total above the last tier, so the largest batch would "
            "have no graph to replay."
        )
    # Deliberately NOT required: that every `bs * verify_width` be a member. A
    # batch CAN round up to a tier above its own `bs * verify_width`, leaving
    # slack that plan_hybrid_ragged_tier cannot spend inside the per-request cap
    # -- but that is safe, because `tier > bs * verify_width` forces
    # `slots = min(tier, max_bs) > bs` (tier > bs*width >= bs gives it when
    # tier <= max_bs; tier <= max_bs*width with tier > bs*width gives bs < max_bs
    # otherwise), so the substrate's `pad_verify_lens_to_bucket` hands the
    # remainder to PAD requests rather than inflating a real one. Requiring
    # membership would force the grid to contain one tier per captured batch
    # size, which is a large part of what makes the naive union explode.
    return tiers


class HybridRaggedTierPlan(msgspec.Struct, frozen=True):
    """One verify step's resolved ragged geometry.

    ``accept_verify_lens`` are the true ``w_r``: they drive the accept cutoff
    and the length contract the acceptance script checks. ``forward_verify_lens``
    are those widths grown to fill the tier (each still ``<= verify_width``);
    they drive the compact window, the attention geometry and the KV writes.
    They are equal exactly when ``sum(w_r)`` already sits on a tier.
    """

    accept_verify_lens: List[int]
    forward_verify_lens: List[int]
    tier_num_tokens: int


def plan_hybrid_ragged_tier(
    *,
    accept_verify_lens: Sequence[int],
    verify_width: int,
    tier_num_tokens: int,
) -> HybridRaggedTierPlan:
    """Grow ``w_r`` to fill ``tier_num_tokens``, never past ``verify_width``.

    The slack (``tier - Σ w_r``) has to go somewhere: the substrate's
    ``pad_verify_lens_to_bucket`` puts ALL of it on the last real request when
    the batch already fills the captured slots, which can push that request's
    width past ``verify_width``. A request only ever had ``verify_width`` KV
    slots reserved at ``[seq_len, seq_len + verify_width)``, so a wider window
    makes the attention read slots the scheduler never gave it.

    Spending the slack here instead keeps every request inside its reservation
    and leaves the substrate's leftover at zero, so nothing is inflated later.
    Whatever cannot be spent (only possible when ``tier > bs * verify_width``,
    i.e. a DP peer with more requests set the tier) is left for the substrate --
    and in exactly that case the captured slots strictly exceed ``bs``, so the
    substrate hands the remainder to PAD requests, never to a real one.

    Allocation is lowest-index-first, which is arbitrary but must be
    DETERMINISTIC: every TP rank runs this on the same broadcast widths and the
    result selects the forward's shape, so a rank computing a different split
    would give the group differently-shaped collectives.
    """
    lens = [int(w) for w in accept_verify_lens]
    tier = int(tier_num_tokens)
    total = sum(lens)
    assert tier >= total, (
        f"hybrid ragged tier {tier} < sum(w_r) {total}; the tier must be a "
        "round-UP of the step's total"
    )
    spendable = min(tier, len(lens) * verify_width) - total
    forward = list(lens)
    index = 0
    while spendable > 0 and index < len(forward):
        room = verify_width - forward[index]
        take = min(room, spendable)
        forward[index] += take
        spendable -= take
        index += 1
    return HybridRaggedTierPlan(
        accept_verify_lens=lens,
        forward_verify_lens=forward,
        tier_num_tokens=tier,
    )


def resolve_hybrid_ragged_tier(
    *,
    total_verify_tokens: int,
    tier_grid: Optional[Sequence[int]],
    max_tier: Optional[int] = None,
) -> Optional[int]:
    """Round the step's total up to a captured tier, or None to stay eager.

    None means "no token-keyed graph can hold this step": nothing was captured
    (``--disable-cuda-graph``, or the runner is not in ragged mode), the total is
    above the grid's ceiling, or no grid member fits under ``max_tier``. All are
    correctness-neutral -- ``build_hybrid_ragged_layout`` then makes the layout
    exact and the verify runs eager, which is the eager mode path. Returning None rather
    than letting ``round_up_grid`` raise matters: a batch the grid cannot serve is
    a throughput event, not a crash.

    ``max_tier`` is how the caller says "a tier above this is only safe if the
    graph is CERTAIN to run". Pass ``bs * verify_width`` on any path where the
    verify might fall back to eager: a tier above that leaves slack that
    :func:`plan_hybrid_ragged_tier` cannot spend, and the eager DSV4 path calls
    ``padded_to_bucket(padded_bs=bs)`` with NO pad requests, so the substrate
    folds the whole remainder into the LAST REAL request -- widening it past
    ``verify_width`` and past the KV window the scheduler reserved for it. On the
    graph path the same tier is safe, because ``tier > bs * verify_width`` forces
    the captured slots above ``bs`` and the remainder lands on pad requests
    instead.
    """
    from sglang.srt.speculative.ragged_verify import round_up_grid

    if not tier_grid:
        return None
    ceiling = tier_grid[-1] if max_tier is None else min(tier_grid[-1], max_tier)
    if total_verify_tokens > ceiling:
        return None
    tier = round_up_grid(total=total_verify_tokens, grid=tier_grid)
    return tier if tier <= ceiling else None


def hybrid_ragged_capture_tiers(*, target_worker) -> Optional[List[int]]:
    """The target decode runner's captured token grid, or None.

    None whenever no token-keyed verify graph exists: cuda graphs disabled, or
    the runner did not enter ragged mode (``supports_ragged_verify()`` false, or
    this is the draft runner, which stays fixed width by design).
    """
    runner = target_worker.model_runner.decode_cuda_graph_runner
    if runner is None or not runner.ragged_verify_mode:
        return None
    return runner.capture_num_tokens


def hybrid_ragged_capture_max_bs(*, target_worker) -> int:
    """Largest request count any captured ragged verify graph can hold.

    This is the runner's ``max_bs = max(capture_bs)``, i.e. ``--cuda-graph-max-bs``
    clamped to the request pool -- NOT ``--max-running-requests``, which can be
    larger. It bounds the runner's per-rank admission test
    (``batch_size <= _ragged_capture_slots(tier) == min(tier, max_bs)``); since
    every ``w_r >= 1`` implies ``tier >= bs``, that test reduces exactly to
    ``bs <= max_bs``.

    Returns 0 when there is no ragged runner, so a caller comparing ``bs >`` this
    treats "no graphs" as "cannot admit", which is the safe direction.
    """
    runner = target_worker.model_runner.decode_cuda_graph_runner
    if runner is None or not runner.ragged_verify_mode:
        return 0
    return int(runner.max_bs)


def hybrid_ragged_dp_tier_enabled(*, server_args) -> bool:
    """Whether this process must agree the graph tier with its DP peers.

    Mirrors DSPARK's ``_dp_tier_gather_enabled`` (``dspark_planner.py``) with two
    deliberate differences:

      * DSPARK additionally requires ``not disable_overlap_schedule``, because it
        hides the all-gather inside the overlap pipeline. hybrid retrieval's tier is
        EXACT (T2): it is only known after the draft forward, so the gather is
        synchronous by construction and the overlap condition does not apply.
        future no-sync mode's lagged budget is where that changes.
      * The whole thing is additionally gated on an explicit opt-in
        (``--speculative-hybrid-ragged-dp-tier``). A ragged tier that the ranks
        disagree on hangs the group with no assertion firing first, so turning it
        on is a deliberate act rather than a side effect of passing
        ``--enable-dp-attention``.

    ``attn_tp_size == 1`` is load-bearing, not inherited caution: the gather runs
    over the tp group, so one rank per DP rank is what makes the gathered vector
    the per-DP-rank vector.
    """
    if not server_args.speculative_hybrid_ragged:
        return False
    if not server_args.speculative_hybrid_ragged_dp_tier:
        return False

    from sglang.srt.environ import envs
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled
    from sglang.srt.runtime_context import get_parallel

    # From utils.common, NOT layers.dp_attention -- the latter does not re-export
    # it, so importing it from there is an ImportError at worker init. This is
    # the same module server_args._check_hybrid_ragged_tier_server_args and
    # dspark_planner import it from; the startup guard and this runtime gate MUST
    # agree, and sharing the import site is half of what makes that true.
    from sglang.srt.utils.common import require_mlp_tp_gather

    return (
        is_dp_attention_enabled()
        and require_mlp_tp_gather(server_args)
        and get_parallel().attn_tp_size == 1
        and get_parallel().attn_cp_size == 1
        and not server_args.speculative_skip_dp_mlp_sync
        and server_args.disaggregation_mode == "null"
        and server_args.pp_size == 1
        and not envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get()
    )


def agree_hybrid_ragged_dp_tier(
    *,
    local_total_verify_tokens: int,
    tier_grid: Optional[Sequence[int]],
    cpu_group,
) -> Optional[int]:
    """All-gather the per-rank totals and return the tier EVERY rank must use.

    Under DP attention all ranks replay ONE graph shape: the decode runner keys
    the ragged verify graph off ``layout.graph_num_tokens`` with no cross-rank
    max of its own (unlike the fixed-width path, which takes
    ``max(original_global_num_tokens_cpu)``), so the agreement has to be baked
    into the layout before it gets there. Ranks that disagree select different
    graphs, the in-graph MoE all-gather shapes mismatch, and the group hangs --
    with no assertion firing first, which is why this is a collective rather
    than a local computation.

    Contract on the gathered value, mirroring DSPARK:
      ``> 0``  this rank's ``Σ w_r``
      ``0``    this rank has nothing to verify (idle) but does not object
      ``< 0``  this rank VETOES ragged this step

    Returns None when any rank vetoed, when no rank had work, or when the agreed
    maximum does not fit the grid. None means the whole group takes the
    fixed-width path this step -- the decision is derived from the gathered
    vector, so it is identical on every rank by construction.

    ``cpu_group`` MUST be passed in, and must be the TARGET worker's tp group
    (gloo side). Resolving it here with ``get_tp_group()`` -- which is what DSPARK
    does from the scheduler -- would be wrong at this call site: hybrid retrieval gathers
    from inside ``EagleDraftWorker.draft()``, which under DP attention runs within
    ``draft_tp_context`` (``eagle_worker_v2.py``), and that context PATCHES the
    process-global tp group to the draft runner's own group. With
    ``attn_tp_size == 1`` that group has world_size 1, so ``get_tp_group()`` here
    would give every rank a private one-element gather -- each rank would then
    "agree" its own tier and the group would hang, i.e. exactly the failure this
    function exists to prevent, silently.

    hybrid retrieval's DP support requires ``attn_tp_size == 1`` (validated at startup),
    so the target tp group has exactly one rank per DP rank and the gathered
    vector is the per-DP-rank vector.
    """
    import torch.distributed as dist

    world_size = dist.get_world_size(group=cpu_group)
    local = torch.tensor([int(local_total_verify_tokens)], dtype=torch.int64)
    gathered = torch.empty((world_size,), dtype=torch.int64)
    dist.all_gather_into_tensor(gathered, local, group=cpu_group)

    totals = gathered.tolist()
    if any(total < 0 for total in totals):
        return None
    peak = max(totals, default=0)
    if peak <= 0:
        return None
    return resolve_hybrid_ragged_tier(total_verify_tokens=peak, tier_grid=tier_grid)


def build_hybrid_ragged_idle_layout(
    *,
    tier_num_tokens: int,
    target_worker,
) -> RaggedVerifyLayout:
    """The layout an IDLE DP rank rides so it replays the busy ranks' graph.

    An idle rank has no requests, but it still has to enter the same captured
    graph as everybody else or the in-graph DP collectives mismatch. The layout's
    only job is to carry the agreed tier: the batch itself stays empty, the
    decode runner copies zero rows into the static buffers, and the verify seam
    and the accept cutoff both skip an idle batch.

    It delegates to the runner's OWN capture-layout builder rather than inventing
    a shape, so the widths an idle rank presents are byte-for-byte the ones the
    graph was captured with (``num_tokens`` spread evenly over the captured
    slots). Inventing one -- e.g. a single fake request holding the whole tier --
    would hand the attention a query block wider than ``verify_width`` for a
    request whose seq_len is the graph's fill value, i.e. a much longer stale KV
    read than the fixed-width idle path has ever done.
    """
    runner = target_worker.model_runner.decode_cuda_graph_runner
    layout = runner._capture_ragged_verify_layout(int(tier_num_tokens))
    assert layout is not None and layout.graph_num_tokens == int(tier_num_tokens), (
        "idle DP rank could not build the agreed tier "
        f"{tier_num_tokens}; got {layout}"
    )
    return layout
