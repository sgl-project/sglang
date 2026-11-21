"""Sampling penalties of chain speculative decoding.

A verify block of a request has ``W = K + 1`` rows: row ``j`` predicts the token
after the anchor (the last committed token) and the drafts ``d_1 .. d_j``.
Ordinary decoding would penalize that token with ``d_1 .. d_j`` counted, so a
penalty of row ``j`` differs from the block-start tables only for the drafts
``d_1 .. d_j`` and, while min_new_tokens lifts its stop-token ban inside the
block, for the stop tokens. ``SpeculativePenalties.overrides`` computes these
per-row values and ``apply_`` applies the block-start tables with them.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch

from sglang.kernels.ops.sampling.penalties import (
    MinNewTokensState,
    TokenPenalty,
    apply_chain_penalties_,
    chain_penalty_overrides,
)


@dataclasses.dataclass(frozen=True)
class ChainPenaltyOverrides:
    """The penalties of a verify block that differ from its block-start tables.

    ``tokens`` is ``[bs, T]`` int32 (-1 marks an empty slot); ``additive`` and
    ``scaling`` (None without a repetition penalty) are ``[bs, W, T]`` fp32: the
    values for ``tokens[b, t]`` at row ``j`` of request ``b``.
    """

    tokens: torch.Tensor
    additive: torch.Tensor
    scaling: Optional[torch.Tensor]


@dataclasses.dataclass(frozen=True)
class SpeculativePenalties:
    """A batch's penalty state at the start of a chain speculative step.

    ``additive`` and ``scaling`` are the block-start tables of
    ``SamplingBatchInfo.update_penalties``; the other fields reference the
    state that each prepared penalizer adds with ``_add_speculative_state``
    (None when a penalizer is not prepared). The scheduler counts the step's
    accepted drafts into the referenced tables after the forward, on the same
    stream, so read them only during the forward.
    """

    additive: torch.Tensor
    scaling: Optional[torch.Tensor]
    frequency: Optional[TokenPenalty] = None
    presence: Optional[TokenPenalty] = None
    repetition: Optional[TokenPenalty] = None
    min_new_tokens: Optional[MinNewTokensState] = None

    @property
    def vocab_size(self) -> int:
        return self.additive.shape[1]

    def overrides(
        self, draft_tokens: torch.Tensor, num_drafts: Optional[torch.Tensor] = None
    ) -> ChainPenaltyOverrides:
        """The per-row penalties of blocks whose rows ``1 .. K`` follow the drafts
        ``draft_tokens`` (``[bs, K]``). Only the first ``num_drafts[b]`` drafts
        of a request count (all when None); tokens outside ``[0, V)`` never do,
        so a caller keeps a token out of the history by replacing it with -1."""
        tokens, additive, scaling = chain_penalty_overrides(
            draft_tokens,
            num_drafts,
            vocab_size=self.vocab_size,
            frequency=self.frequency,
            presence=self.presence,
            repetition=self.repetition,
            min_new_tokens=self.min_new_tokens,
        )
        return ChainPenaltyOverrides(tokens, additive, scaling)

    def apply_(
        self,
        logits: torch.Tensor,
        overrides: ChainPenaltyOverrides,
        logit_bias: Optional[torch.Tensor] = None,
    ) -> None:
        """Penalize request-major ``[bs * W, V]`` verify logits in place with the
        ops of ``SamplingBatchInfo.apply_logits_bias`` for one row, then add
        ``logit_bias[b]`` (``[bs, V]``) when given."""
        apply_chain_penalties_(
            logits,
            self.additive,
            self.scaling,
            overrides.tokens,
            overrides.additive,
            overrides.scaling,
            logit_bias,
        )
