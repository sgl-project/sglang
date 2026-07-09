import types
import unittest
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_batch(req_names, top_logprobs_nums, token_ids_logprobs):
    reqs = [types.SimpleNamespace(rid=name) for name in req_names]
    bs = len(reqs)
    batch = ScheduleBatch(reqs=reqs)
    batch.model_config = types.SimpleNamespace(is_encoder_decoder=False)
    batch.sampling_info = MagicMock()
    batch.req_pool_indices = torch.arange(bs, dtype=torch.int64)
    batch.req_pool_indices_cpu = batch.req_pool_indices.clone()
    batch.seq_lens = torch.full((bs,), 8, dtype=torch.int64)
    batch.seq_lens_cpu = batch.seq_lens.clone()
    batch.orig_seq_lens = batch.seq_lens.to(torch.int32)
    batch.input_ids = torch.arange(bs, dtype=torch.int64)
    batch.return_logprob = True
    batch.top_logprobs_nums = top_logprobs_nums
    batch.token_ids_logprobs = token_ids_logprobs
    batch.multimodal_inputs = [None] * bs
    batch.spec_info = None
    return batch


class TestMergeBatchOutOfPlace(unittest.TestCase):
    def test_merge_batch_rebinds_lists_without_mutating_either_side(self):
        """merge_batch must build new list objects; neither side's original lists may be mutated."""
        self_batch = _make_batch(["a", "b"], [1, 2], [[10], [20]])
        other_batch = _make_batch(["c"], [3], [[30]])

        self_reqs_before = self_batch.reqs
        self_reqs_snapshot = self_batch.reqs[:]
        self_top_before = self_batch.top_logprobs_nums
        self_top_snapshot = self_batch.top_logprobs_nums[:]
        other_reqs_before = other_batch.reqs
        other_reqs_snapshot = other_batch.reqs[:]

        self_batch.merge_batch(other_batch)

        self.assertEqual(
            [r.rid for r in self_batch.reqs],
            ["a", "b", "c"],
        )
        self.assertEqual(self_batch.top_logprobs_nums, [1, 2, 3])
        self.assertEqual(self_batch.token_ids_logprobs, [[10], [20], [30]])
        self.assertIsNot(self_batch.reqs, self_reqs_before)
        self.assertIsNot(self_batch.top_logprobs_nums, self_top_before)
        self.assertEqual(self_reqs_before, self_reqs_snapshot)
        self.assertEqual(self_top_before, self_top_snapshot)
        self.assertIs(other_batch.reqs, other_reqs_before)
        self.assertEqual(other_batch.reqs, other_reqs_snapshot)
        self.assertEqual(other_batch.top_logprobs_nums, [3])


if __name__ == "__main__":
    unittest.main()
