import unittest

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils.profile_utils import get_profile_stage
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestProfileStage(unittest.TestCase):
    def test_profile_stage(self):
        expected = {
            ForwardMode.EXTEND: "prefill",
            ForwardMode.MIXED: "prefill",
            ForwardMode.SPLIT_PREFILL: "prefill",
            ForwardMode.DLLM_EXTEND: "prefill",
            ForwardMode.DECODE: "decode",
            ForwardMode.TARGET_VERIFY: "decode",
            ForwardMode.DRAFT_EXTEND_V2: "decode",
            ForwardMode.IDLE: None,
        }

        for forward_mode, stage in expected.items():
            with self.subTest(forward_mode=forward_mode):
                self.assertEqual(get_profile_stage(forward_mode), stage)


if __name__ == "__main__":
    unittest.main()
