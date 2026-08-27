"""--enable-draft-prefetch: pre-run the next round's draft after draft_extend.

With draft prefetch, EAGLEWorkerV2 pre-runs the next round's draft decode
back-to-back with this round's draft_extend (on the forward stream), and
pre-concatenates the per-step topk into the next draft input. The next decode
round then builds its verify input from the pre-concatenated candidate chain
(prepare_verify_input_for_draft_prefetch) instead of re-running the draft.
This file guards that the optimization is lossless:

  - greedy parity vs a non-spec reference server (SpecParityKit);
  - acceptance length / verify bookkeeping (SpecCorrectnessKit);
  - gsm8k quality + accept length (SpecAccuracyKit);
  - radix / abort / grammar features with a strict memory check -- a leak from
    the extra pre-run draft would trip it (SpecFeatureKit).

Constraint surface (enforced by _check_draft_prefetch): EAGLE family (this
file exercises EAGLE3), topk=1, num_steps > 1, non-adaptive, no rejection
sampling. The unit tests for that validation live in
test/registered/unit/spec/test_spec_draft_prefetch_args.py.
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecCorrectnessKit,
    SpecFeatureKit,
    SpecParityKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base

register_cuda_ci(est_time=900, stage="base-b", runner_config="1-gpu-small")


class _DraftPrefetchBase(Eagle3Base):
    """EAGLE3 topk=1 chain with the next-round draft pre-run enabled."""

    extra_args = ("--enable-draft-prefetch",)
    # The pre-run draft allocates/reorders state on the forward stream every
    # round, so leaks are likely here; the strict check turns them into
    # failures.
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagle3DraftPrefetch(
    _DraftPrefetchBase, SpecCorrectnessKit, SpecAccuracyKit, SpecFeatureKit
):
    """Overlap scheduler -- the primary draft-prefetch pipeline."""

    disable_overlap = False


class TestEagle3DraftPrefetchNoOverlap(_DraftPrefetchBase, SpecCorrectnessKit):
    """Synchronous scheduler: the pre-run draft has to work there too."""

    disable_overlap = True


class TestEagle3DraftPrefetchParity(SpecParityKit, _DraftPrefetchBase):
    """Greedy output must equal the non-spec reference (lossless decode).

    SpecParityKit is first in the bases so its setUpClass launches and kills
    the reference server BEFORE the fixture launches the spec server (one
    model resident at a time).
    """

    disable_overlap = False


if __name__ == "__main__":
    unittest.main()
