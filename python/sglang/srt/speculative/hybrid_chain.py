"""Hybrid MTP + retrieval verify chain (hybrid retrieval).

The MTP (EAGLE/NextN, topk=1) draft heads the chain. Per request, the chain is
truncated at the first position whose draft probability falls below that
position's threshold tau, and the remaining slots -- up to the fixed verify
width ``L = speculative_num_draft_tokens`` -- are refilled with an
agreement-gated NGRAM retrieval continuation. The spliced linear chain is handed
to the shared ``build_tree_kernel_efficient`` as a topk=1 chain of depth
``L - 1``, so every verify tensor (mask / positions / retrieve_*) matches the
kernels exactly and the target verify is a plain (wider) TARGET_VERIFY forward.

Two invariants keep this safe:

* **Correctness is untouched.** A retrieved token is committed only if it equals
  the target model's argmax at that position, exactly like an MTP draft token.
  Retrieval changes accept LENGTH, never the emitted text.
* **Monotonicity.** Retrieval is appended AFTER the kept MTP block, and
  attention is causal, so a retrieval slot can never change the accept/reject
  decision of an MTP slot. When the agreement gate fails the full MTP chain is
  kept (no truncation), so a hybrid step never accepts fewer tokens than the
  pure-MTP step would have.

Numbering: everything user-facing is indexed by **verify-chain column**. For
``num_steps = S`` the chain is ``[column 0 = bonus token, column 1 = MTP draft
0, ..., column S = MTP draft S-1]``, and retrieval fills columns ``S+1 ..
L-1``. Internally the chain math is indexed by draft position (column - 1).

The FIRST MTP draft (verify column 1) is ALWAYS kept. Its draft logprob is
produced by the previous iteration's draft-extend and would have to be carried
across a batch re-composition boundary. Avoiding that cross-iteration carrier
also lets the draft loop run under a CUDA graph (see ``full_cuda_graph``).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_utils import TreeMaskMode, build_tree_kernel_efficient
from sglang.srt.speculative.hybrid_ragged import compute_hybrid_verify_lens

# Sentinel for "never truncate at this chain position".
TAU_DISABLED = ("off", "disabled", "none", "")


def parse_position_thresholds(
    spec: str, scalar_tau: float, num_steps: int
) -> List[float]:
    """Resolve the per-position truncation thresholds (probabilities).

    ``spec`` is indexed by **verify-chain column**. For
    ``--speculative-num-steps S`` the verify chain that the MTP draft produces
    is::

        column 0  = the bonus token (the target's own last prediction, the
                    chain root) -- never a draft, never truncatable
        column 1  = MTP draft 0
        ...
        column S  = MTP draft S-1

    so ``spec`` must carry exactly ``S + 1`` comma-separated entries. An entry
    in :data:`TAU_DISABLED` (or ``0``) means "never truncate at this column".
    Columns 0 and 1 must both be disabled -- column 0 because it is the bonus
    token, column 1 because its draft logprob comes from the previous
    iteration's draft-extend (see the module docstring).

    Example for ``S = 3`` (verify columns 0,1,2,3): ``off,off,0.40,0.55``
    keeps the bonus and the first draft unconditionally, truncates before the
    second draft when its confidence < 0.40, and before the third when < 0.55.

    An empty ``spec`` falls back to ``scalar_tau`` on columns 2..S.

    Returns ``num_steps`` probabilities indexed by DRAFT position
    (``result[i]`` is the threshold of verify column ``i + 1``); ``0.0`` marks
    a disabled position.
    """
    if not spec.strip():
        return [0.0] + [scalar_tau] * (num_steps - 1)

    entries = [x.strip().lower() for x in spec.split(",")]
    if len(entries) != num_steps + 1:
        raise ValueError(
            "speculative_hybrid_tau_per_pos is indexed by verify-chain column "
            "(column 0 = the bonus token, columns 1..S = the MTP drafts), so it "
            f"must have exactly speculative_num_steps + 1 = {num_steps + 1} "
            f"comma-separated entries; got {len(entries)}: {spec!r}. For S=3 the "
            "calibrated setting is 'off,off,0.40,0.55'."
        )
    columns = []
    for column, entry in enumerate(entries):
        if entry in TAU_DISABLED:
            columns.append(0.0)
            continue
        value = float(entry)
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"speculative_hybrid_tau_per_pos column {column} must be a "
                f"probability in [0, 1] or one of {TAU_DISABLED}; got {entry!r}"
            )
        columns.append(value)
    if columns[0] != 0.0:
        raise ValueError(
            "speculative_hybrid_tau_per_pos column 0 must be disabled (one of "
            f"{TAU_DISABLED}, or 0): column 0 is the bonus token, which the "
            "target model itself produced -- it is never a draft and is never "
            "truncated."
        )
    if columns[1] != 0.0:
        raise ValueError(
            "speculative_hybrid_tau_per_pos column 1 must be disabled (one of "
            f"{TAU_DISABLED}, or 0): the first MTP draft's logprob comes from "
            "the previous iteration's draft-extend and is not carried across "
            "batch re-composition, so it cannot gate a truncation."
        )
    # Drop the bonus column; the chain math is indexed by draft position.
    return columns[1:]


class HybridChainState:
    """Per-worker mutable state of the hybrid chain.

    Holds the truncation thresholds (init-static), the retrieval-extension EMA
    that drives dynamic tau, and the previous step's per-request diagnostics
    that the EMA update reads back after target verification.
    """

    # EMA smoothing of the retrieval-extension rate (dynamic tau).
    EMA_DECAY = 0.95

    def __init__(
        self,
        *,
        thresholds: List[float],
        tau_min: float,
        dynamic_tau: bool,
        device: str,
    ):
        self.thresholds = thresholds  # [num_steps] probabilities, 0 = disabled
        self.tau_min = tau_min
        self.dynamic_tau = dynamic_tau
        # log thresholds for chain positions 1..num_steps-1 (position 0 is
        # always kept, so it never participates in the comparison).
        self._tail_log = torch.tensor(
            [_safe_log(t) for t in thresholds[1:]],
            dtype=torch.float32,
            device=device,
        )
        self._tail_log_min = torch.full_like(self._tail_log, _safe_log(tau_min))
        # Positions configured as "disabled" must stay disabled under dynamic
        # tau as well, so remember them and re-apply -inf after interpolation.
        self._tail_enabled = torch.tensor(
            [t > 0.0 for t in thresholds[1:]], dtype=torch.bool, device=device
        )
        # Interpolation operand with -inf swapped for tau_min: (-inf) - (-inf)
        # is NaN, and even though the NaN is discarded by the torch.where below
        # it would trip the async NaN probes.
        self._tail_log_finite = torch.where(
            self._tail_enabled, self._tail_log, self._tail_log_min
        )

        # Retrieval-extension EMA: 0 = retrieval is not extending anything, so
        # the cold start avoids paying any truncation cost.
        self.extension_ema: float | torch.Tensor = 0.0
        # Stashed [bs] tensors from the last built chain, read by the EMA update.
        self.last_num_keep_drafts: Optional[torch.Tensor] = None
        self.last_graft_ok: Optional[torch.Tensor] = None

    def tail_log_thresholds(self) -> torch.Tensor:
        """Effective log thresholds for chain positions 1..num_steps-1.

        With dynamic tau on, each enabled position drifts between ``tau_min``
        (minimal truncation ~ pure MTP, when retrieval is idle) and its
        configured tau (aggressive handoff, when retrieval is extending), driven
        by the retrieval-extension EMA. Disabled positions stay disabled.
        """
        if not self.dynamic_tau:
            return self._tail_log
        ema = self.extension_ema
        # Interpolate in LOG space: monotone in the same direction as
        # probability space and avoids a device round-trip when the EMA is a
        # device scalar (overlap mode keeps it on GPU to dodge a per-step D2H).
        effective = (
            self._tail_log_min + (self._tail_log_finite - self._tail_log_min) * ema
        )
        return torch.where(self._tail_enabled, effective, self._tail_log)

    def update_extension_ema(
        self, *, num_correct_drafts: torch.Tensor, keep_on_device: bool
    ) -> None:
        """Fold this verify's retrieval-extension rate into the EMA.

        "Extended" = acceptance reached past the kept MTP prefix AND the graft
        actually happened -- i.e. RETRIEVAL contributed. Gating on ``graft_ok``
        is required: on a failed graft the full (un-truncated) MTP chain is
        kept, so ``num_correct_drafts > num_keep_drafts`` there is plain MTP
        acceptance, and counting it would push tau up while retrieval was idle.
        """
        if self.last_num_keep_drafts is None or self.last_graft_ok is None:
            return
        extended = (num_correct_drafts > self.last_num_keep_drafts) & self.last_graft_ok
        rate = extended.float().mean()
        if keep_on_device:
            # Overlap: keep the EMA on GPU -- a per-step .item() would
            # re-serialize the hot path. It is only ever read as a broadcast
            # operand in tail_log_thresholds().
            self.extension_ema = (
                self.EMA_DECAY * self.extension_ema + (1.0 - self.EMA_DECAY) * rate
            )
        else:
            self.extension_ema = (
                self.EMA_DECAY * float(self.extension_ema)
                + (1.0 - self.EMA_DECAY) * rate.item()
            )


def _safe_log(p: float) -> float:
    """log(p) with 0 mapped to -inf (= "never truncate at this position")."""
    return -math.inf if p <= 0.0 else math.log(p)


def splice_hybrid_chain(
    *,
    mtp_draft_tokens: torch.Tensor,
    step_logprobs: Optional[torch.Tensor],
    retrieval_chains: Optional[torch.Tensor],
    retrieval_lens: Optional[torch.Tensor],
    tail_log_thresholds: torch.Tensor,
    verify_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the ``[bs, L-1]`` spliced chain body (verify columns 1..L-1).

    ``mtp_draft_tokens`` is ``[bs, num_steps]`` (column i = chain position i);
    ``step_logprobs`` is ``[bs, num_steps-1]`` (column j = log p_draft of chain
    position j+1), or None when ``num_steps == 1``; ``retrieval_chains`` is
    ``[bs, L]`` with column 0 = the root (verified context's last token) and
    ``retrieval_lens`` is ``[bs]``, the retriever's own count of how many of the
    columns after the root are REAL.

    ``retrieval_lens`` is not optional bookkeeping: the padding value is 0 and
    token id 0 is a legal vocabulary entry, so without it the agreement gate
    would match padding against a genuine MTP token 0 and report a graft that
    never happened.

    Returns ``(spliced, num_keep_drafts, num_agree_drafts, graft_ok)``.
    """
    bs, num_steps = mtp_draft_tokens.shape
    device = mtp_draft_tokens.device

    # Leading run of kept MTP positions. Position 0 is always kept, so start at
    # 1 and extend while each tail position clears its threshold. cumprod over
    # the 0/1 keep mask is exact and device-portable.
    if step_logprobs is None:
        num_keep_drafts = torch.ones(bs, dtype=torch.long, device=device)
    else:
        keep = (step_logprobs >= tail_log_thresholds).to(torch.float32)
        num_keep_drafts = 1 + torch.cumprod(keep, dim=1).sum(dim=1).to(torch.long)

    if retrieval_chains is None:
        retrieval_chains = torch.zeros(
            (bs, verify_width), dtype=mtp_draft_tokens.dtype, device=device
        )
        retrieval_lens = torch.zeros(bs, dtype=torch.long, device=device)
    assert retrieval_lens is not None, (
        "splice_hybrid_chain needs the retriever's own continuation lengths; "
        "they cannot be recovered from the chain because the padding value 0 is "
        "a legal token id"
    )

    # Agreement gate: graft the retrieval tail only when the retrieved chain
    # matches the kept MTP prefix, so the tail is validly conditioned on it.
    # Compare only against columns the retriever actually filled: past its
    # length the chain holds zero PADDING, and a genuine MTP token 0 there would
    # otherwise register as agreement and fake a graft that never happened.
    step_col = torch.arange(1, num_steps + 1, device=device)
    retrieval_valid = step_col[None, :] <= retrieval_lens.to(torch.long)[:, None]
    agree = (
        (retrieval_chains[:, 1 : num_steps + 1] == mtp_draft_tokens) & retrieval_valid
    ).to(torch.float32)
    num_agree_drafts = torch.cumprod(agree, dim=1).sum(dim=1).to(torch.long)
    graft_ok = num_agree_drafts >= num_keep_drafts

    # When the agreement gate FAILS, do NOT truncate: feed the full MTP chain
    # into verify instead of zero-padding the tail. A pad token is always
    # rejected; a real MTP token still has a chance, so a failed graft must not
    # waste the truncated positions.
    effective_keep = torch.where(
        graft_ok, num_keep_drafts, torch.full_like(num_keep_drafts, num_steps)
    )

    # Verify column c (1..L-1) carries chain position c: MTP draft c-1 while
    # c <= effective_keep, otherwise the retrieval continuation (or a zero pad
    # when the graft failed).
    col = torch.arange(1, verify_width, device=device)  # [L-1]
    keep_width = effective_keep[:, None]
    is_mtp = col[None, :] <= keep_width
    mtp_at = mtp_draft_tokens[:, (col - 1).clamp(0, num_steps - 1)]
    retrieval_at = retrieval_chains[:, col]
    use_retrieval = (col[None, :] > keep_width) & graft_ok[:, None]
    spliced = torch.where(
        is_mtp,
        mtp_at,
        torch.where(use_retrieval, retrieval_at, torch.zeros_like(retrieval_at)),
    )
    return spliced, num_keep_drafts, num_agree_drafts, graft_ok


def build_hybrid_verify_input(
    *,
    batch,
    draft_input: EagleDraftInput,
    mtp_draft_tokens: torch.Tensor,
    step_logprobs: Optional[torch.Tensor],
    retrieval_chains: Optional[torch.Tensor],
    retrieval_lens: Optional[torch.Tensor],
    state: HybridChainState,
    topk: int,
    verify_width: int,
    tree_mask_mode: TreeMaskMode,
    target_worker,
    ragged: bool = False,
) -> EagleVerifyInput:
    """Assemble the hybrid ``EagleVerifyInput`` at width ``L = verify_width``.

    Mirrors ``eagle_worker_common.build_eagle_verify_input`` but feeds the
    spliced MTP+retrieval chain and declares ``spec_steps = L - 1`` so the
    accept kernel's row width (``spec_steps + 1``) is L and acceptance is not
    capped at the MTP chain depth.

    With ``ragged=True`` the input additionally carries a ``RaggedVerifyLayout``
    so the target-verify forward only computes each request's live columns; see
    ``hybrid_ragged.py``. The spliced chain itself stays ``[bs, L]`` either way.
    """
    spliced, num_keep_drafts, num_agree_drafts, graft_ok = splice_hybrid_chain(
        mtp_draft_tokens=mtp_draft_tokens,
        step_logprobs=step_logprobs,
        retrieval_chains=retrieval_chains,
        retrieval_lens=retrieval_lens,
        tail_log_thresholds=state.tail_log_thresholds(),
        verify_width=verify_width,
    )
    state.last_num_keep_drafts = num_keep_drafts
    state.last_graft_ok = graft_ok

    bs = mtp_draft_tokens.shape[0]
    device = mtp_draft_tokens.device

    tree_mask_buf, position_buf = (
        target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
    )
    # build_tree_kernel uses seq_lens_sum only to size the (non-preallocated)
    # tree mask; over-sizing is safe. Skip the per-iter .sum().item() D2H.
    seq_lens_sum = batch.seq_lens_sum
    if seq_lens_sum is None:
        if tree_mask_buf is None:
            seq_lens_sum = bs * target_worker.model_runner.attn_backend.max_context_len
        else:
            seq_lens_sum = 0  # preallocated buffer -> the kernel ignores it

    # parent_list / top_scores_index are the runtime-invariant linear chain,
    # same closed form as the topk=1 preallocs but at width L-1.
    parent_list = torch.arange(
        -1, verify_width - 2, dtype=torch.long, device=device
    ).repeat(bs, 1)
    top_scores_index = torch.arange(
        verify_width - 1, dtype=torch.long, device=device
    ).repeat(bs, 1)
    (
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        draft_tokens,
    ) = build_tree_kernel_efficient(
        draft_input.bonus_tokens,
        parent_list,
        top_scores_index,
        spliced,
        batch.seq_lens,
        seq_lens_sum,
        topk,
        verify_width - 1,  # chain depth
        verify_width,  # num_verify_tokens
        tree_mask_mode,
        tree_mask_buf,
        position_buf,
    )

    # The retriever's own length. NOT a non-zero count: token id 0 is a legal
    # vocabulary entry (DSV4's <|begin_of_sentence|>, and it is in the trie
    # because the full prompt is indexed), so counting non-zeros conflates a
    # genuine 0 with tail padding and reports a chain shorter than it is --
    # which would make w_r violate its own `>= 1 + num_keep_drafts` floor.
    num_retrieval_tokens = (
        retrieval_lens.to(num_keep_drafts.dtype)
        if retrieval_lens is not None
        else torch.zeros_like(num_keep_drafts)
    )
    verify_lens = (
        compute_hybrid_verify_lens(
            num_keep_drafts=num_keep_drafts,
            graft_ok=graft_ok,
            num_retrieval_tokens=num_retrieval_tokens,
            num_steps=mtp_draft_tokens.shape[1],
            verify_width=verify_width,
        )
        if ragged
        else None
    )

    # TP consistency: num_keep_drafts thresholds a continuous logsumexp of the
    # TP-sharded all-reduced draft logits (NOT bit-identical across ranks), and
    # mtp_draft_tokens is a per-rank argmax. The GREEDY verify path does not
    # broadcast accept_index, so any per-rank divergence in the spliced
    # candidates would desync seq_lens across the TP group (hang / KV
    # corruption). Broadcast the final candidate chain from rank 0, mirroring
    # what the sampling path does for predict/accept_index. positions /
    # retrieve_* / tree_mask are deterministic (derived from parent_list and
    # seq_lens), so only draft_tokens needs syncing.
    #
    # verify_lens is derived from the SAME non-bit-identical quantities, and
    # under ragged it selects the forward's shape: a per-rank disagreement would
    # give the ranks differently-shaped collectives and hang the group. It is
    # concatenated into the existing broadcast rather than sent separately --
    # one collective, and the two can never be broadcast out of step.
    tp_group = (
        get_parallel().attn_tp_group if is_dp_attention_enabled() else get_tp_group()
    )
    if tp_group.world_size > 1:
        if verify_lens is None:
            tp_group.broadcast(draft_tokens, src=0)
        else:
            payload = torch.cat(
                [draft_tokens.view(-1), verify_lens.to(draft_tokens.dtype)]
            )
            tp_group.broadcast(payload, src=0)
            draft_tokens.view(-1).copy_(payload[: draft_tokens.numel()])
            verify_lens = payload[draft_tokens.numel() :].to(torch.int32)

    accept_verify_lens_cpu = None
    if verify_lens is not None:
        # T2 (exact) tier policy: one small D2H per step. verify_lens_cpu is not
        # optional here -- the eager DSV4 metadata path reads it through
        # compute_ragged_extend_lengths, and RaggedVerifyLayout's invariant
        # checks only run when it is populated. future no-sync mode replaces this with the lagged
        # EMA budget (T1), which needs no sync at all.
        accept_verify_lens_cpu = [int(v) for v in verify_lens.tolist()]

    return EagleVerifyInput(
        draft_token=draft_tokens,
        custom_mask=tree_mask,
        positions=positions,
        retrieve_index=retrieve_index,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        retrieve_cum_len=None,
        # spec_steps = L-1 so max_tree_depth (= spec_steps + 1) is L; otherwise
        # the accept kernel caps acceptance at num_steps + 1 and silently drops
        # every retrieval slot past the MTP chain.
        spec_steps=verify_width - 1,
        topk=topk,
        draft_token_num=verify_width,
        capture_hidden_mode=None,
        seq_lens_sum=None,
        seq_lens_cpu=None,
        draft_probs=None,
        # NO layout yet, on purpose. The layout carries `graph_num_tokens` = the
        # captured tier, and under DP attention that tier is a CROSS-RANK
        # agreement -- which an idle rank must join too, and an idle rank never
        # reaches this function. So the layout is built one level up, in
        # `EagleDraftWorker._finalize_hybrid_ragged_layout`, where the hybrid and
        # the idle branches converge. Only the accept widths (true `w_r`) are
        # settled here.
        ragged_verify_layout=None,
        ragged_accept_verify_lens=verify_lens,
        ragged_accept_verify_lens_cpu=accept_verify_lens_cpu,
    )
