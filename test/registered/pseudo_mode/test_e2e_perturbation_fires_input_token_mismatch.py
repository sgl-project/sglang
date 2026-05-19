"""Positive fire test for ``INPUT_TOKEN_MISMATCH``.

The other type-a / smoke tests are negative ("assert no violations") —
they would still pass even if the new ``INPUT_TOKEN_MISMATCH`` fail-
reason path were silently dead. This one is the missing positive
witness: under ``SGLANG_PSEUDO_INPUT_PERTURB_PROB=1.0`` the sampler
override deliberately writes the wrong token, sglang feeds that wrong
token as input on the next decode step, and the canary's pre-chain-
hash assert must catch the divergence with
``FailReason.INPUT_TOKEN_MISMATCH``.

If this test ever turns green by chance (e.g. perturbation lands on
a token id that happens to match the oracle's prediction) we'd want
to know — ``prob=1.0`` is intentional so the mask is fully populated
every step.
"""

from __future__ import annotations

import logging
import os
import unittest
from test.registered.pseudo_mode._fake_prompt import fake_prompt
from test.registered.pseudo_mode._pseudo_engine import PseudoEngine
from test.registered.pseudo_mode._test_utils import (
    PSEUDO_MODE_MODEL,
    requires_cuda,
)
from unittest import mock

from sglang.jit_kernel.kv_cache_canary import FailReason
from sglang.test.ci.ci_register import register_cuda_ci

logger = logging.getLogger(__name__)

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")


@requires_cuda
class TestPerturbationFiresInputTokenMismatch(unittest.TestCase):
    """Inject a per-step perturbation; assert canary catches it."""

    def test_full_perturbation_fires_input_token_mismatch(self) -> None:
        """With ``perturb_prob=1.0`` the sampler emits ``(predicted + 1) %
        vocab`` every step. sglang feeds that perturbed token as the
        next decode step's input, but the oracle's
        ``expected_write_token_ids`` still carries the unperturbed
        prediction (the commit hook tracks oracle predictions, not
        ``req.output_ids``). The canary's pre-chain-hash assert in
        ``run_write_req_chain`` then fires ``INPUT_TOKEN_MISMATCH`` on
        the first decode step after the first perturbed sampler call.
        """
        env_patch = {
            "SGLANG_PSEUDO_INPUT_PERTURB_PROB": "1.0",
            "SGLANG_PSEUDO_INPUT_PERTURB_SEED": "0xDEAD",
        }
        with mock.patch.dict(os.environ, env_patch):
            with PseudoEngine.launch(
                model=PSEUDO_MODE_MODEL,
                num_hidden_layers=1,
                cuda_graph=False,
                radix_cache=False,
            ) as engine:
                handle = engine.admit(prompt=fake_prompt(16), max_new_tokens=4)
                # Step 1: prefill. No decode-side input yet, no fire.
                engine.step()
                # Step 2 onwards: decode. The first decode forwards
                # consume the previously-perturbed sampler output as
                # input, so the canary trips here.
                engine.step_until(handle, n=3)

                violations = engine.canary_violations()
                real = [v for v in violations if v.is_real()]
                self.assertTrue(
                    real,
                    "expected at least one canary violation under perturb_prob=1.0, "
                    "got zero — INPUT_TOKEN_MISMATCH fire path likely dead",
                )
                input_token_mismatches = [
                    v
                    for v in real
                    if v.fail_reason_int == int(FailReason.INPUT_TOKEN_MISMATCH)
                ]
                self.assertTrue(
                    input_token_mismatches,
                    f"expected at least one INPUT_TOKEN_MISMATCH violation, got "
                    f"reasons={[v.fail_reason for v in real]}",
                )
                first = input_token_mismatches[0]
                # Field reuse contract: expected_hash = oracle expected,
                # actual_hash = sglang actual. ``token_id`` mirrors the
                # actual. They must differ by exactly the perturb shift
                # (+1 mod vocab) — but vocab_size is model-internal here,
                # so we only assert they differ.
                self.assertNotEqual(
                    first.expected_hash,
                    first.actual_hash,
                    "INPUT_TOKEN_MISMATCH row's expected/actual fields collapsed",
                )


if __name__ == "__main__":
    unittest.main(verbosity=3)
