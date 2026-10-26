"""Unit tests for srt/utils/profile_utils.py"""

import copy
import unittest

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils.profile_utils import ProfileManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestProfileManager(CustomTestCase):
    def test_prebuilt_batch_leaves_running_capture_untouched(self):
        """PREBUILT batches (PD decode admission) run no model forward;
        stepping one must not raise, and must leave a running capture unchanged."""
        with published_topology():
            manager = ProfileManager(cpu_group=None)
            manager.configure(
                output_dir=None,
                start_step=None,
                num_steps=1,
                # An empty activity list starts no profiler backend.
                activities=[],
                with_stack=None,
                record_shapes=None,
                profile_by_stage=True,
                profile_id="test",
                merge_profiles=False,
                profile_prefix="",
                profile_stages=["decode"],
            )
            manager.step(forward_mode=ForwardMode.DECODE)
            # step() updates the running state in place, so compare against a copy.
            capture = copy.copy(manager.stage_based_trigger.running_state)

            manager.step(forward_mode=ForwardMode.PREBUILT)

        self.assertIsNotNone(capture)
        self.assertEqual(manager.stage_based_trigger.running_state, capture)


if __name__ == "__main__":
    unittest.main()
