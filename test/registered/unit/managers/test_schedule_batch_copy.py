"""The batch copy the result path runs on carries what that path reads.

``ScheduleBatch.copy`` keeps only the fields ``process_batch_result`` uses. Lazy
Mamba tracking frees a request's tracking slot there through the batch's token
allocator, so the copy must carry the allocator as well as the request table.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestScheduleBatchCopy(unittest.TestCase):
    def test_the_copy_carries_the_token_allocator(self):
        """BUG REGRESSION. With `--mamba-radix-cache-strategy
        extra_buffer_lazy`, the first decode tracking boundary freed a slot
        through ``batch.token_to_kv_pool_allocator.state_slots`` on the copy,
        which had no allocator: AttributeError on None."""
        allocator = SimpleNamespace(state_slots=SimpleNamespace())
        table = SimpleNamespace()
        batch = ScheduleBatch(
            reqs=[],
            req_to_token_pool=table,
            token_to_kv_pool_allocator=allocator,
            model_config=SimpleNamespace(is_encoder_decoder=False),
            spec_algorithm=SpeculativeAlgorithm.NONE,
            req_pool_indices=torch.zeros(0, dtype=torch.int64),
            req_pool_indices_cpu=torch.zeros(0, dtype=torch.int64),
        )
        copy = batch.copy()
        self.assertIs(copy.token_to_kv_pool_allocator, allocator)
        self.assertIs(copy.req_to_token_pool, table)


if __name__ == "__main__":
    unittest.main()
