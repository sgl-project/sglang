import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSchedulerHiSparseDecodeBatch(unittest.TestCase):
    def test_build_hisparse_decode_batch_sets_req_pool_indices_cpu(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.device = "cpu"
        scheduler.req_to_token_pool = MagicMock()
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.tree_cache = MagicMock()
        scheduler.model_config = types.SimpleNamespace(vocab_size=128)
        scheduler.enable_overlap = False
        scheduler.spec_algorithm = MagicMock()
        scheduler.future_map = MagicMock()

        req0 = types.SimpleNamespace(
            req_pool_idx=3,
            origin_input_ids=[1, 2, 3],
            output_ids=[4],
            return_logprob=False,
            grammar=None,
            return_hidden_states=False,
            is_prefill_only=False,
        )
        req1 = types.SimpleNamespace(
            req_pool_idx=5,
            origin_input_ids=[6, 7],
            output_ids=[8, 9],
            return_logprob=False,
            grammar=None,
            return_hidden_states=False,
            is_prefill_only=False,
        )

        with patch(
            "sglang.srt.managers.scheduler.SamplingBatchInfo.from_schedule_batch",
            return_value=MagicMock(),
        ):
            batch = scheduler._build_hisparse_decode_batch([req0, req1])

        expected = torch.tensor([3, 5], dtype=torch.int64)
        self.assertTrue(torch.equal(batch.req_pool_indices, expected))
        self.assertTrue(torch.equal(batch.req_pool_indices_cpu, expected))
        self.assertEqual(batch.req_pool_indices_cpu.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
