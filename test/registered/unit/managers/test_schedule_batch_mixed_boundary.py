import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ForwardMode, ScheduleBatch  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestScheduleBatchMixedBoundary(CustomTestCase):
    def test_reused_batch_gets_current_mixed_boundary(self):
        batch = ScheduleBatch(
            reqs=[SimpleNamespace()],
            forward_mode=ForwardMode.EXTEND,
            mix_running_indices_cpu=torch.tensor([0]),
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            out_cache_loc=torch.tensor([3]),
            prefix_lens=[3],
            extend_lens=[1],
            extend_num_tokens=1,
            extend_logprob_start_lens=[0],
        )
        self.assertIsNone(batch.mix_decode_bs)
        running = ScheduleBatch(
            reqs=[
                SimpleNamespace(
                    _refresh_fill_ids=lambda: None,
                    set_extend_range=lambda start, end: None,
                )
                for _ in range(2)
            ],
            seq_lens_cpu=torch.tensor([5, 6]),
            req_pool_indices=torch.tensor([1, 2]),
            req_pool_indices_cpu=torch.tensor([1, 2]),
            out_cache_loc=torch.tensor([4, 5]),
        )
        with patch.object(
            batch,
            "merge_batch",
            side_effect=lambda other: setattr(batch, "reqs", batch.reqs + other.reqs),
        ):
            batch.mix_with_running(running)

        self.assertEqual(batch.forward_mode, ForwardMode.MIXED)
        self.assertEqual(batch.mix_decode_bs, 2)
        batch.forward_mode = ForwardMode.DECODE
        self.assertIsNone(batch.mix_decode_bs)


if __name__ == "__main__":
    unittest.main()
