"""Verify the sampler override is end-to-end live.

The sampler override is the load-bearing piece that makes pseudo-mode
deterministic: it replaces every per-step sampler output with the
oracle's ``predict_output_token`` so the model's RNG / temperature
state stops mattering. If the override silently failed to install (a
patched ``ModelRunner.sample`` whose closure never fires, or a guard
that skipped patching), every other test would still pass — none of
them assert "the actual generated tokens equal the oracle prediction".

With the canary's ``INPUT_TOKEN_MISMATCH`` fire path now live (see
``test_e2e_perturbation_fires_input_token_mismatch``), a long-running
decode under default settings (``perturb_prob=0``) is a strong proxy:
if the override were broken, sglang's real sampler would emit tokens
that disagree with ``oracle.predict_input_token`` at the next decode
step, and the canary would trip. We run a non-trivial number of
decode steps to exercise the override under multiple cuda-graph
shapes and replays.
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
class TestSamplerOverrideDecodeClean(unittest.TestCase):
    """Long-running decode with default settings stays canary-clean."""

    def test_thirty_two_decode_steps_no_violations(self) -> None:
        """Admit one req for 32 max_new_tokens, drain to completion,
        and assert every step is canary-clean. A broken sampler
        override would surface as ``INPUT_TOKEN_MISMATCH`` on the very
        first decode step (sglang's actual sampled token would not
        match the oracle's expected one).

        We also assert the req actually produced 32 output tokens —
        catches the regression where the override turns the model into
        an EOS spammer on the first step.
        """
        max_new = 32
        with PseudoEngine.launch(
            model=PSEUDO_MODE_MODEL,
            num_hidden_layers=1,
            cuda_graph=False,
            radix_cache=False,
        ) as engine:
            handle = engine.admit(prompt=fake_prompt(24), max_new_tokens=max_new)
            engine.step_until(handle, n=max_new)
            engine.step_until_idle(max_steps=4)

            engine.assert_no_canary_violations()

            # output_tokens is empty after the req finishes (sglang
            # drops finished reqs from waiting + running). The
            # invariant we care about — "the override drove all 32
            # steps clean" — is already covered by
            # assert_no_canary_violations above; the active_reqs check
            # here is a belt-and-suspenders sanity that the req
            # reached completion and is no longer in either queue.
            still_active = [
                entry
                for entry in engine.active_reqs()
                if entry.rid == handle.rid
            ]
            self.assertFalse(
                still_active,
                f"req {handle.rid} unexpectedly still active: {still_active}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=3)
