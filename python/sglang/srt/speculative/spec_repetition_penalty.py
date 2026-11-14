"""Repetition penalty for EAGLE-family verify, applied per draft position and over
every committed token, so chain drafts match non-speculative greedy decoding."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties
from sglang.srt.utils.device import async_h2d

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class SpecRepetitionState:
    def __init__(
        self,
        *,
        req_generation: torch.Tensor,
        vocab_size: int,
        device: torch.device | str,
    ):
        # Host tensor bumped on every row allocation: a changed value means the
        # row has a new owner (also a resumed or same-rid request).
        self._req_generation = req_generation
        self._device = device
        # [req_to_token rows, vocab] bool, written only on the forward stream.
        self.seen = torch.zeros(
            (req_generation.shape[0], vocab_size), dtype=torch.bool, device=device
        )
        self._true = torch.ones((), dtype=torch.bool, device=device)
        self._row_generation: Dict[int, int] = {}

    def sync_owners(self, reqs: List[Req]) -> None:
        slots = [req.kv.req_pool_idx for req in reqs]
        generations = self._req_generation[slots].tolist()
        reset_rows: List[int] = []
        rebuild_rows: List[int] = []
        rebuild_tokens: List[int] = []
        for req, slot, generation in zip(reqs, slots, generations):
            if self._row_generation.get(slot) == generation:
                continue
            self._row_generation[slot] = generation
            reset_rows.append(slot)
            # Empty unless the request resumed after a retract.
            rebuild_rows.extend([slot] * len(req.output_ids))
            rebuild_tokens.extend(req.output_ids)
        if not reset_rows:
            return
        # index_fill_ takes the value as a kernel scalar; `seen[rows] = False`
        # would sync the host with the forward stream.
        self.seen.index_fill_(0, self._h2d(reset_rows), False)
        if rebuild_rows:
            self.seen.index_put_(
                (self._h2d(rebuild_rows), self._h2d(rebuild_tokens)), self._true
            )

    def apply_verify(
        self,
        *,
        logits: torch.Tensor,
        slots: torch.Tensor,
        candidates: torch.Tensor,
        penalties: torch.Tensor,
        chain: bool,
    ) -> None:
        # logits [bs * K, V] in place; candidates [bs, K] = root + drafts; penalties [bs, 1].
        bs, k = candidates.shape
        vocab_size = logits.shape[-1]
        assert vocab_size == self.seen.shape[1], (vocab_size, self.seen.shape)
        scale = torch.where(self.seen[slots.long()], penalties, 1.0)
        scale = scale.unsqueeze(1).repeat(1, k, 1)
        cand = candidates.long()
        if chain:
            # Row j also penalizes cand[0..j]; entries i > j point at the root,
            # which row j penalizes anyway, so the scatter stays idempotent.
            prefix = torch.ones((k, k), dtype=torch.bool, device=logits.device).tril()
            index = torch.where(prefix, cand[:, None, :], cand[:, :1, None])
            scale.scatter_(2, index, penalties[:, :, None].expand(bs, k, k))
        else:
            # Tree drafts: ancestors are not tracked, so only the root is added.
            scale.scatter_(
                2,
                cand[:, :1, None].expand(bs, k, 1),
                penalties[:, :, None].expand(bs, k, 1),
            )
        apply_scaling_penalties(
            logits=logits, scaling_penalties=scale.view(bs * k, vocab_size)
        )

    def add_committed_tokens(
        self,
        *,
        slots: torch.Tensor,
        roots: torch.Tensor,
        predict: torch.Tensor,
        accept_index: torch.Tensor,
    ) -> None:
        # accept_index [bs, depth] holds global indices into predict, -1 padded;
        # predict is shared by all ranks, and the bonus is the next verify's root.
        index = accept_index.long()
        emitted = predict.long()[index[:, :-1].clamp_min(0)]
        roots = roots.long()[:, None]
        correct_drafts = torch.where(index[:, 1:] >= 0, emitted, roots)
        tokens = torch.cat([roots, correct_drafts], dim=1)
        rows = slots.long()[:, None].expand_as(tokens)
        # A device value keeps this off the host: `seen[rows, tokens] = True`
        # copies the Python scalar with a stream sync and breaks overlap.
        self.seen.index_put_((rows, tokens), self._true)

    def apply_mixed_tails(
        self,
        *,
        reqs: List[Req],
        slots: torch.Tensor,
        roots: torch.Tensor,
        penalties: torch.Tensor,
        scaling_penalties: torch.Tensor,
    ) -> None:
        # Mixed-chunk tails decode one plain token from their pending root: give
        # them the exact rows of scaling_penalties [n, V] and commit the root.
        self.sync_owners(reqs)
        slots = slots.long()
        roots = roots.long()
        scale = torch.where(self.seen[slots], penalties, 1.0)
        scale.scatter_(1, roots[:, None], penalties)
        scaling_penalties.copy_(scale)
        self.seen.index_put_((slots, roots), self._true)

    def _h2d(self, values: List[int]) -> torch.Tensor:
        return async_h2d(values, dtype=torch.int64, device=self._device)


def create_spec_repetition_state(
    *, req_generation: torch.Tensor, vocab_size: int, device: torch.device | str
) -> Optional[SpecRepetitionState]:
    # Called at attention-backend init, before graph capture, so the table's
    # memory is taken at startup rather than on the first penalized request.
    if not envs.SGLANG_ENABLE_SPEC_EXACT_REPETITION_PENALTY.get():
        return None
    return SpecRepetitionState(
        req_generation=req_generation, vocab_size=vocab_size, device=device
    )
