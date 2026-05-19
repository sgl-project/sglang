"""Regression for sglang#20711 — outputs differ across batch composition.

Bug summary: the same prompt produced different logprobs (and
occasionally different next-token argmax) when run as a singleton vs.
as part of a larger batch. Root cause was a batch-dimension-dependent
reduction order in one of the attention paths; the bug was a real
numerical non-determinism across batch shape.

How pseudo-mode catches it (and what it does not): pseudo-mode's
sampler override forces every ``next_token_id`` to the oracle's
``predict_output_token(req, step)``, which is a pure function of
``(seed, req_id, step)``. By construction the **emitted token
sequence** is identical for the same req regardless of batch shape —
so this test cannot observe the raw logits divergence from sglang#20711.
What it *can* observe is the second-order consequence: if any batch-
composition-sensitive code path (block table layout, ``out_cache_loc``
slicing, position broadcast across the batch dimension) feeds the
*wrong* slot or position on one of the two runs, the canary fires.

Concretely the test runs the same prompt twice:

1. as a singleton batch (admit, step to completion, harvest the
   per-step ``active_rids`` ordering),
2. as part of a multi-req batch composed with two filler reqs.

Both runs must finish with zero canary violations, and the
singleton run's per-step ``active_rids`` history must match a trivial
``[req]`` pattern (sanity: the harness really did batch-of-one). The
multi-req run is allowed to interleave with the fillers but the req
under test must appear in every step's ``active_rids`` until it
finishes.
"""

from __future__ import annotations

import logging
import unittest
from test.registered.pseudo_mode._fake_prompt import fake_prompt
from test.registered.pseudo_mode._pseudo_engine import PseudoEngine
from test.registered.pseudo_mode._test_utils import (
    PSEUDO_MODE_MODEL,
    requires_cuda,
)

from sglang.test.ci.ci_register import register_cuda_ci

logger = logging.getLogger(__name__)

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")


_PROMPT_LEN: int = 64
_MAX_NEW_TOKENS: int = 4


@requires_cuda
class TestBatchCompositionCrossBatch(unittest.TestCase):
    """Same prompt in two batch compositions must produce no canary violations."""

    def test_singleton_then_multi_batch_clean(self) -> None:
        """Run prompt P alone, then alongside fillers; assert both clean.

        ``assert_no_canary_violations`` at the end of each run is the
        load-bearing check — under sglang#20711, the multi-batch run's
        block-table / position routing for the second slot would mis-
        align against the oracle-predicted (req, position) pair and
        fire ``INPUT_POSITION_MISMATCH`` or ``INPUT_TOKEN_MISMATCH``.
        """
        prompt = fake_prompt(_PROMPT_LEN, seed=0xC0DE)

        # Run 1: singleton batch.
        singleton_history = self._run_singleton(prompt)
        for active in singleton_history:
            self.assertEqual(
                len(active),
                1,
                f"singleton run unexpectedly had >1 active rid: {active}",
            )

        # Run 2: multi-req batch with the same prompt under test plus
        # two filler reqs that occupy adjacent batch slots.
        multi_history, target_rid = self._run_multi_batch(prompt)
        # The req under test must appear in every step that it could
        # still be active. We only require that it appears at least
        # once and is consistently present until it finishes — there
        # is no global ordering guarantee from the scheduler about
        # which filler runs in which step.
        first_step_seen = None
        last_step_seen = None
        for idx, active in enumerate(multi_history):
            if target_rid in active:
                if first_step_seen is None:
                    first_step_seen = idx
                last_step_seen = idx
        self.assertIsNotNone(
            first_step_seen,
            "multi-batch run never scheduled the target req",
        )
        # Contiguous activity window: once admitted, the target should
        # not disappear and reappear (that would imply force-preempt /
        # resume, which this test does not trigger).
        if first_step_seen is not None and last_step_seen is not None:
            window = multi_history[first_step_seen : last_step_seen + 1]
            for idx_in_window, active in enumerate(window):
                self.assertIn(
                    target_rid,
                    active,
                    f"target rid {target_rid!r} missing from step "
                    f"{first_step_seen + idx_in_window} active_rids={active}",
                )

    def _run_singleton(self, prompt: list[int]) -> list[list[str]]:
        """Run ``prompt`` as a singleton batch and return per-step active rids."""
        with PseudoEngine.launch(
            model=PSEUDO_MODE_MODEL,
            num_hidden_layers=1,
            radix_cache=False,
            cuda_graph=False,
        ) as engine:
            handle = engine.admit(
                prompt=prompt, max_new_tokens=_MAX_NEW_TOKENS, req_id="target"
            )
            results = engine.step_until(handle, n=_MAX_NEW_TOKENS)
            engine.step_until_idle(max_steps=8)
            engine.assert_no_canary_violations()
            return [list(r.active_rids) for r in results]

    def _run_multi_batch(
        self, prompt: list[int]
    ) -> tuple[list[list[str]], str]:
        """Run ``prompt`` alongside two filler reqs; return per-step active rids."""
        with PseudoEngine.launch(
            model=PSEUDO_MODE_MODEL,
            num_hidden_layers=1,
            radix_cache=False,
            cuda_graph=False,
        ) as engine:
            filler_a = engine.admit(
                prompt=fake_prompt(_PROMPT_LEN, seed=0xF111),
                max_new_tokens=_MAX_NEW_TOKENS,
                req_id="filler-a",
            )
            handle = engine.admit(
                prompt=prompt, max_new_tokens=_MAX_NEW_TOKENS, req_id="target"
            )
            filler_b = engine.admit(
                prompt=fake_prompt(_PROMPT_LEN, seed=0xF222),
                max_new_tokens=_MAX_NEW_TOKENS,
                req_id="filler-b",
            )
            del filler_a, filler_b  # handles only matter for cleanup
            results = engine.step_until(handle, n=_MAX_NEW_TOKENS)
            engine.step_until_idle(max_steps=12)
            engine.assert_no_canary_violations()
            return [list(r.active_rids) for r in results], handle.rid


if __name__ == "__main__":
    unittest.main(verbosity=3)
