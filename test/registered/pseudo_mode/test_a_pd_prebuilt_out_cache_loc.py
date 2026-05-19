"""Regression for sglang#24230 — PD prebuilt ``out_cache_loc`` slice error.

Bug summary: in the PD-disaggregation prefill→decode hand-off path the
prebuilt batch's ``out_cache_loc`` was sliced with the wrong length on
chunked or partial prefill resumes. Decodes that consumed the rebuilt
batch ended up writing tokens into KV slots that did not belong to
their own req.

How the canary would catch it: the head kernel's existing slot-identity
verify (and the new ``INPUT_TOKEN_MISMATCH`` field added in Block 1)
fires on the first decode step where ``out_cache_loc[entry]`` resolves
to a slot whose stored ``(req_id, position)`` does not match the
oracle's expected pair.

Honesty note: PD disaggregation is **out of scope for pseudo-mode v1**.
The testing README §"v1 scope" item 7 explicitly defers
``PD / TP / PP`` to v2. The :class:`PseudoEngine` harness wires a
single-process, single-rank scheduler; there is no prefill server or
decode server pair, no KV transfer channel, and no prebuilt batch
exchange in this configuration. Marking this scenario as ``skip`` (not
``expectedFailure``) because the bug path simply cannot be exercised
through the v1 harness — flipping to a real regression requires the
PD plumbing tracked under v2.
"""

from __future__ import annotations

import logging
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

logger = logging.getLogger(__name__)

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "PseudoEngine requires CUDA")
@unittest.skip(
    "sglang#24230 lives on the PD-disaggregation path; pseudo-mode v1 is "
    "single-process / single-rank and does not exercise the prefill->decode "
    "hand-off or prebuilt batch slicing. Re-enable once v2 wires PD."
)
class TestPdPrebuiltOutCacheLoc(unittest.TestCase):
    """Placeholder for the PD prebuilt out_cache_loc slicing regression."""

    def test_pd_prebuilt_slice_clean(self) -> None:
        """Would drive prefill server → KV transfer → decode server and
        assert no canary violations. Requires v2 PD wiring."""
        self.fail(
            "test body intentionally absent: PD path not reachable from v1 "
            "PseudoEngine; class-level skip should fire before this runs"
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
