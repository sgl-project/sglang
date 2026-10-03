import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.frozen_kv_mtp_info import FrozenKVMTPVerifyInput
from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import FrozenKVMTPDraftWorker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFrozenKVMTPIdleDraft(CustomTestCase):
    """An idle batch (a DP-attention rank with no requests) takes the early
    return in FrozenKVMTPDraftWorker.draft. That call must match
    EagleVerifyInput.create_idle_input, which takes the device as well."""

    def test_idle_batch_returns_empty_verify_input(self):
        worker = object.__new__(FrozenKVMTPDraftWorker)
        worker.topk = 1
        worker.speculative_num_steps = 3
        worker.speculative_num_draft_tokens = 4
        worker.device = "cpu"

        verify_input = worker.draft(SimpleNamespace(forward_mode=ForwardMode.IDLE))

        self.assertIsInstance(verify_input, FrozenKVMTPVerifyInput)
        self.assertEqual(verify_input.draft_token.numel(), 0)
        self.assertEqual(verify_input.draft_token.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
