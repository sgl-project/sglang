"""Regression for sglang#25015 — EAGLE draft worker positions misalignment.

Bug summary: under EAGLE speculative decoding, the draft worker built
``positions`` as ``[p, p+2, p+3]`` instead of ``[p, p+1, p+2]`` on some
spec steps. The model still ran (positions are just an int64 tensor)
but the KV cache was indexed at the wrong slots, silently corrupting
state for subsequent decodes.

How the canary catches it: the head kernel's ``INPUT_POSITION_MISMATCH``
fail reason compares the live ``forward_batch.positions[entry]`` against
the oracle-expected per-(req, step) position before any KV write
happens. Under the bug, ``positions[entry]`` would skip a value and the
canary would fire on the first draft step that hit the misalignment.

Honesty note: EAGLE end-to-end in pseudo-mode requires a draft model
path, which is not auto-resolved by the harness today. ``launch`` will
forward ``speculative_algorithm="EAGLE"`` but ServerArgs validation
demands ``speculative_draft_model_path``. We mark this test as
``expectedFailure`` until the harness wires a dummy-weights draft model
(tracked as a v1 follow-up; spec decoding is listed under v1 scope item
7 in the testing README).
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
class TestEaglePositionsMisalign(unittest.TestCase):
    """Run a few EAGLE draft+verify decode steps and assert clean canary."""

    @unittest.expectedFailure
    def test_eagle_draft_positions_clean(self) -> None:
        """Drive EAGLE prefill + a handful of decode steps; expect no violations.

        Without sglang#25015 fix, the head kernel's
        INPUT_POSITION_MISMATCH would fire on the first draft step that
        emits a non-contiguous ``positions`` row. With the fix in place,
        ``assert_no_canary_violations`` returns clean.

        Currently expected to fail because EAGLE requires a draft model
        path that the harness does not yet auto-provision.
        """
        with PseudoEngine.launch(
            model=_MODEL,
            num_hidden_layers=1,
            speculative_algorithm="EAGLE",
            radix_cache=False,
            cuda_graph=False,
        ) as engine:
            handle = engine.admit(prompt=fake_prompt(64), max_new_tokens=4)

            # Prefill step.
            engine.step()
            engine.assert_no_canary_violations()

            # Decode steps: each step under EAGLE runs a draft pass
            # (positions = [p, p+1, ..., p+num_draft-1]) followed by a
            # target verify pass. The bug manifested as a position skip
            # inside the draft pass.
            engine.step_until(handle, n=4)
            engine.assert_no_canary_violations()


if __name__ == "__main__":
    unittest.main(verbosity=3)
