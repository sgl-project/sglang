"""Smoke test for :class:`PseudoEngine` startup + single-step + shutdown.

One end-to-end happy-path check: launch a tiny Qwen3-0.6B (single
hidden layer, dummy weights) under ``--enable-pseudo-mode``, admit one
fake-prompt request, step twice, assert no canary violations, then
shut down.
"""

from __future__ import annotations

import time
import unittest

from sglang.test.ci.ci_register import register_cuda_ci

from test.registered.pseudo_mode._fake_prompt import fake_prompt
from test.registered.pseudo_mode._pseudo_engine import PseudoEngine

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")


_MODEL = "Qwen/Qwen3-0.6B"
_LAUNCH_BUDGET_S: float = 20.0


class TestPseudoEngineSmoke(unittest.TestCase):
    """Single end-to-end happy path: launch / admit / step / shutdown."""

    def test_launch_admit_step_shutdown(self) -> None:
        # Step 1: launch within the 20-second budget.
        t0 = time.monotonic()
        engine = PseudoEngine.launch(
            model=_MODEL,
            num_hidden_layers=1,
            cuda_graph=False,
        )
        launch_elapsed = time.monotonic() - t0
        self.assertLess(
            launch_elapsed,
            _LAUNCH_BUDGET_S,
            f"PseudoEngine.launch took {launch_elapsed:.1f}s > {_LAUNCH_BUDGET_S}s",
        )

        try:
            # Step 2: admit one request and confirm we got a handle back
            # without the scheduler having fired a forward yet.
            handle = engine.admit(prompt=fake_prompt(64), max_new_tokens=2)
            self.assertEqual(handle.max_new_tokens, 2)
            self.assertEqual(len(handle.prompt), 64)

            # Step 3: step twice; first step typically prefills, second
            # decodes one token (target = max_new_tokens=2 so we should
            # see the req still active after 2 steps unless it finishes
            # early).
            results = engine.step_until(handle, n=2)
            self.assertGreaterEqual(len(results), 1)

            # Step 4: no canary violations should have been recorded.
            engine.assert_no_canary_violations()
        finally:
            # Step 5: shutdown always — even on assertion failure — so
            # the scheduler subprocess does not leak.
            engine.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=3)
