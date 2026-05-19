"""Regression for sglang#24401 — CUDA graph runner does not reset positions.

Bug summary: the CUDA graph runner allocated a static ``positions``
tensor at capture time but did not overwrite every entry on replay.
Reqs that decoded with a shorter prompt than the capture-time batch
re-used stale position values from the previous replay's tail, feeding
the model the wrong RoPE / KV index.

How the canary catches it: the head kernel's ``INPUT_POSITION_MISMATCH``
fail reason compares ``forward_batch.positions[entry]`` (the value the
graph-replay actually passed in) against the oracle's expected
per-(req, step) position. Under the bug the stale-tail entries diverge
from oracle on the first replay that hits a smaller batch shape; with
the fix in place, every replay overwrites every entry.

Test plan: enable CUDA graph capture, admit two reqs whose decode
trajectories span enough decode steps to (a) trigger graph capture for
multiple batch shapes and (b) trigger replays of an already-captured
shape. Assert ``assert_no_canary_violations`` is clean after each step
boundary.
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


@requires_cuda
class TestCudaGraphPositionsReset(unittest.TestCase):
    """Span CUDA-graph capture and replay boundaries with multiple reqs."""

    def test_decode_across_capture_replay_clean(self) -> None:
        """Force batch-shape shrink across capture/replay boundaries.

        sglang#24401 specifically manifested on a replay whose batch
        had fewer reqs than the captured shape: stale tail entries kept
        the previous replay's positions. Admit three reqs of unequal
        depths so the running batch shrinks as the shortest one
        finishes — this is the adversarial pattern that surfaced the
        bug. ``assert_no_canary_violations`` after each drain phase
        catches ``INPUT_POSITION_MISMATCH`` on the head kernel.
        """
        with PseudoEngine.launch(
            model=PSEUDO_MODE_MODEL,
            num_hidden_layers=1,
            cuda_graph=True,
            radix_cache=False,
        ) as engine:
            # Step 1: admit three reqs of unequal max_new_tokens. The
            # shortest finishes mid-stream so the running batch shrinks
            # from 3 -> 2 -> 1 across captured replays — the production
            # pattern that surfaced sglang#24401.
            handle_short = engine.admit(
                prompt=fake_prompt(48, seed=0xAAAA), max_new_tokens=2
            )
            handle_mid = engine.admit(
                prompt=fake_prompt(80, seed=0xBBBB), max_new_tokens=4
            )
            handle_long = engine.admit(
                prompt=fake_prompt(112, seed=0xCCCC), max_new_tokens=8
            )

            # Step 2: drive several joint-decode steps so the captured
            # graph holds the bs=3 shape at least once.
            engine.step_until(handle_short, n=2)
            engine.assert_no_canary_violations()

            # Step 3: short req is finished now; the next decode steps
            # shrink the batch to bs=2 (replays of a previously-captured
            # shape with a now-stale tail entry would be caught here).
            engine.step_until(handle_mid, n=4)
            engine.assert_no_canary_violations()

            # Step 4: shrink again to bs=1 for the final stretch.
            engine.step_until(handle_long, n=8)

            # Step 5: drain anything left so the captured graph has run
            # at least once more after both shape-shrink boundaries.
            engine.step_until_idle(max_steps=8)
            engine.assert_no_canary_violations()


if __name__ == "__main__":
    unittest.main(verbosity=3)
