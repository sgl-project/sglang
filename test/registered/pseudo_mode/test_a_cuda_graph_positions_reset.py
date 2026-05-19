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

import torch

from sglang.test.ci.ci_register import register_cuda_ci

logger = logging.getLogger(__name__)

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")


_MODEL: str = "Qwen/Qwen3-0.6B"


@unittest.skipUnless(torch.cuda.is_available(), "PseudoEngine requires CUDA")
class TestCudaGraphPositionsReset(unittest.TestCase):
    """Span CUDA-graph capture and replay boundaries with multiple reqs."""

    def test_decode_across_capture_replay_clean(self) -> None:
        """Drive decode steps that exercise capture + replay for the
        same batch shape and assert no canary violations.

        With the sglang#24401 fix, every replay rewrites the full
        ``positions`` tensor and the head kernel sees the oracle-expected
        position at every entry. Without the fix, a replay's tail entry
        retains a stale position from the previous replay and the head
        kernel writes ``INPUT_POSITION_MISMATCH``.
        """
        with PseudoEngine.launch(
            model=_MODEL,
            num_hidden_layers=1,
            cuda_graph=True,
            radix_cache=False,
        ) as engine:
            # Step 1: admit two reqs whose prompt lengths differ so the
            # decode batch shape is exercised on both capture and replay.
            handle_a = engine.admit(
                prompt=fake_prompt(48, seed=0xAAAA), max_new_tokens=6
            )
            handle_b = engine.admit(
                prompt=fake_prompt(80, seed=0xBBBB), max_new_tokens=6
            )

            # Step 2: drive prefills + a few decode steps. Each call to
            # ``step`` runs one outer event-loop iteration; the canary
            # is pulled at the end of every step inside step_until.
            engine.step_until(handle_a, n=5)

            # Step 3: drive the second req to a similar depth, which
            # triggers replays of the same captured shape after handle_a
            # has finished or evicted.
            engine.step_until(handle_b, n=5)

            # Step 4: drain anything left so the captured graph has run
            # at least once more after both reqs touched it.
            engine.step_until_idle(max_steps=8)

            # Step 5: final canary check — must be clean across every
            # capture + replay boundary that ran above.
            engine.assert_no_canary_violations()


if __name__ == "__main__":
    unittest.main(verbosity=3)
