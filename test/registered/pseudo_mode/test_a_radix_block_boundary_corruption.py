"""Regression for sglang#22819 — radix cache block-boundary corruption.

Bug summary: when two requests shared a prefix whose length was exactly
a multiple of the radix block size, the radix tree's leaf-split routine
copied one block too few from the donor leaf to the new branch. The
recipient req then read garbage KV bytes at the boundary block, and
subsequent decodes silently advanced from corrupt state.

How the canary would catch it: with ``--kv-cache-canary-real-data=bit``
the slot identity stored at the boundary block would not match the
recipient req's ``(req_id, position)`` pair on the first decode after
the prefix hit. The head kernel writes a ``REAL_KV_HASH`` violation
(or ``INPUT_TOKEN_MISMATCH`` if the corruption cascaded into the
input-feed path on the next step).

Honesty note: pseudo-mode v1 disables the radix cache by design. The
oracle is a CPU-side per-(req, position) predictor and does not know
which sub-range of the prefix sglang decided to reuse. Flipping
``radix_cache=True`` here exercises the production path but the
oracle's ``expected_input_tokens`` for the shared-prefix entries does
not yet emit ``SKIP_SENTINEL``, so the head kernel will fire
``INPUT_TOKEN_MISMATCH`` on prefix-cache *hits* (oracle thinks the req
should re-feed the original tokens, sglang correctly skips them).

This test is marked ``expectedFailure`` until the oracle gains a hook
to signal "this entry was supplied by prefix-cache, skip the input
assert" — at which point the assertion becomes a real regression for
sglang#22819.
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


# Page size is 1 by default on sglang Qwen3 path; we choose a prompt
# length that lands the prefix on a likely radix block multiple. The
# exact block size is sglang-internal, so we pick a power of two that
# is large enough to cross at least one allocator block on any
# reasonable configuration.
_SHARED_PREFIX_LEN: int = 128


@requires_cuda
class TestRadixBlockBoundaryCorruption(unittest.TestCase):
    """Two reqs with an exact-block-size shared prefix hitting the radix cache."""

    @unittest.expectedFailure
    def test_shared_prefix_block_boundary_clean(self) -> None:
        """Admit two reqs sharing an exact-multiple-of-block-size prefix.

        With sglang#22819 fix in place AND a v2 oracle that masks
        prefix-cache hits with ``SKIP_SENTINEL`` for the
        ``expected_input_tokens`` buffer, this test would assert clean.
        Currently expected to fail: pseudo-mode v1 disables radix
        cache, and the oracle has no way to signal "this entry is a
        prefix-cache hit" so flipping ``radix_cache=True`` falsely
        fires ``INPUT_TOKEN_MISMATCH`` even on a healthy sglang.
        """
        shared_prefix = fake_prompt(_SHARED_PREFIX_LEN, seed=0xCAFE)
        suffix_a = fake_prompt(16, seed=0xA1)
        suffix_b = fake_prompt(16, seed=0xB2)
        prompt_a = shared_prefix + suffix_a
        prompt_b = shared_prefix + suffix_b

        with PseudoEngine.launch(
            model=PSEUDO_MODE_MODEL,
            num_hidden_layers=1,
            radix_cache=True,
            cuda_graph=False,
        ) as engine:
            # Step 1: admit + prefill req_a so its prefix populates the
            # radix tree.
            handle_a = engine.admit(prompt=prompt_a, max_new_tokens=4)
            engine.step_until(handle_a, n=2)

            # Step 2: admit req_b — its prefill should hit the radix
            # cache for the first _SHARED_PREFIX_LEN tokens and only
            # extend through the unique suffix.
            handle_b = engine.admit(prompt=prompt_b, max_new_tokens=4)
            engine.step_until(handle_b, n=2)

            # Step 3: drain so both reqs decode past the boundary block
            # under any captured graph or eager path.
            engine.step_until_idle(max_steps=12)

            # Step 4: final canary check. Under sglang#22819 fix, no
            # boundary block corruption fires; without it,
            # ``REAL_KV_HASH`` or ``INPUT_TOKEN_MISMATCH`` shows up.
            engine.assert_no_canary_violations()


if __name__ == "__main__":
    unittest.main(verbosity=3)
