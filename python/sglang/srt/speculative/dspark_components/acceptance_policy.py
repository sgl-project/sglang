"""Pluggable acceptance policy seam for DSpark verification.

A policy lets a custom speculative algorithm replace how verified draft
tokens are accepted -- e.g. a relaxed or approximate acceptance rule --
while reusing DSpark's drafting, target verification, KV commit, and
finalization paths.

The hot interface must return the same ``(correct_len, bonus,
cap_trim_lens)`` contract as the native greedy verifier, so every
downstream consumer (finalization, KV commit, request tokens, next draft
input) stays unchanged and unaware of the policy.

Policies are injected by overriding
:meth:`DSparkWorkerV2._build_acceptance_policy
<sglang.srt.speculative.dspark_components.dspark_worker_v2.DSparkWorkerV2._build_acceptance_policy>`;
the intended pattern is a custom speculative algorithm registered via
``SpeculativeAlgorithm.register`` whose worker subclasses
``DSparkWorkerV2``. Returning ``None`` (the default) keeps DSpark's native
strict greedy verification with zero overhead.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

AcceptTuple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class DSparkAcceptancePolicy:
    """Base class for substituting DSpark's greedy acceptance rule.

    Subclasses must implement :meth:`accept`. The lifecycle hooks are
    no-ops by default; override them to keep per-request state (e.g.
    budgets) bound to request identities across decode steps.
    """

    def bind_batch(self, batch) -> None:
        """Called at the prefill lifecycle point, before decode steps.

        Override to bind per-request state to request identities. The
        batch carries ``reqs`` (with ``rid`` and ``req_pool_idx``) and the
        request-pool mapping; heavy bookkeeping belongs here rather than
        on the decode hot path.
        """

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        """Called when a request finishes; release its per-request state."""

    def accept(
        self,
        *,
        candidates: torch.Tensor,
        target_logits: Optional[torch.Tensor],
        cutoff_verify_lens: Optional[torch.Tensor],
        req_pool_indices: torch.Tensor,
        all_greedy: bool,
        native_accept: Callable[[], AcceptTuple],
    ) -> AcceptTuple:
        """Choose the acceptance outcome for one verify batch.

        Args:
            candidates: ``[bs, verify_num_draft_tokens]`` candidate ids;
                column 0 is the anchor token, columns ``1..gamma`` the
                drafts under verification.
            target_logits: ``[bs * verify_num_draft_tokens, vocab]``
                next-token logits produced by the target forward, or
                ``None`` when the backend folds them away.
            cutoff_verify_lens: per-request verify budget from the ragged
                planner (``None`` for the static layout).
            req_pool_indices: request-pool device indices of the batch.
            all_greedy: whether every request in the batch is greedy.
            native_accept: callable returning the native strict greedy
                ``(correct_len, bonus, cap_trim_lens)``; policies may
                delegate to it (e.g. on a non-greedy batch).

        Returns:
            ``(correct_len, bonus, cap_trim_lens)`` with the same shapes
            and semantics as the native greedy verifier.
        """
        raise NotImplementedError
