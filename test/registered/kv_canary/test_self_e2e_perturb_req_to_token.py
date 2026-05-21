from __future__ import annotations

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")


class _PerturbReqToTokenBase(CanaryE2EBase):
    """Perturb point (a): flip the req_to_token slot id of an active req's live position.

    The KV bytes are untouched but the per-forward verify walks the wrong slot, so the
    stored prev-hash chain cannot be reproduced. Expected detection path is the
    per-forward chain_hash check (HEAD or TAIL kernel); since this perturb point is
    group-agnostic, MHA fires only against the FULL group and SWA fires against either
    FULL or SWA. We only assert that *some* per-forward chain_hash violation is logged.
    """

    kv_canary_mode = "log"
    extra_env = {
        "SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB": "0.1",
        "SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS": "0",
    }

    def test_chain_hash_violation_observed(self) -> None:
        self.send_parallel_requests(n=4, assert_all_successs=True, max_new_tokens=200)
        # Per-forward verify emits launch_tag of the form HEAD_<kernel>_<GROUP> or
        # TAIL_<kernel>_<GROUP>. Try HEAD first; if not present, fall back to TAIL.
        try:
            self.assert_violation_logged(
                launch_tag_pattern="HEAD_*",
                fail_reason="chain_hash",
            )
        except AssertionError:
            self.assert_violation_logged(
                launch_tag_pattern="TAIL_*",
                fail_reason="chain_hash",
            )


class TestPerturbReqToTokenMha(_PerturbReqToTokenBase, unittest.TestCase):
    model_mode = "mha"


class TestPerturbReqToTokenSwa(_PerturbReqToTokenBase, unittest.TestCase):
    model_mode = "swa"


if __name__ == "__main__":
    unittest.main()
