"""
Unit tests for profile_utils stage mapping (profile V2).

Usage:
    python test_profile_utils.py
    python -m unittest test_profile_utils.py -v
"""

import unittest

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils.profile_utils import _get_stage_from_forward_mode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGetStageFromForwardMode(CustomTestCase):
    def test_prefill_and_decode_map_to_stages(self):
        self.assertEqual(_get_stage_from_forward_mode(ForwardMode.EXTEND), "prefill")
        self.assertEqual(_get_stage_from_forward_mode(ForwardMode.DECODE), "decode")

    def test_idle_and_prebuilt_are_not_stages(self):
        # IDLE has no forward, and PREBUILT is the disaggregated-decode
        # placeholder whose KV just arrived; it never enters a model forward.
        self.assertIsNone(_get_stage_from_forward_mode(ForwardMode.IDLE))
        self.assertIsNone(_get_stage_from_forward_mode(ForwardMode.PREBUILT))


if __name__ == "__main__":
    unittest.main()
