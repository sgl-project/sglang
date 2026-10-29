"""Regression tests for the one-batch benchmark's request/KV pool wiring."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.benchmark import one_batch
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestOneBatchAllocation(CustomTestCase):
    def test_extend_allocates_slots_with_the_benchmark_pools(self):
        """Benchmark prefill must allocate and populate real request rows before inference."""
        request_pool = ReqToTokenPool(
            size=2, max_context_len=8, device="cpu", enable_memory_saver=False
        )
        allocator = TokenToKVPoolAllocator(
            size=16, dtype=torch.float32, device="cpu", kvcache=None, need_sort=False
        )
        logits = torch.zeros(2, 32)
        next_ids = torch.tensor([7, 8])
        runner = SimpleNamespace(
            device="cpu",
            req_to_token_pool=request_pool,
            token_to_kv_pool_allocator=allocator,
            model_config=SimpleNamespace(vocab_size=32, is_encoder_decoder=False),
            forward=lambda batch: SimpleNamespace(
                logits_output=SimpleNamespace(next_token_logits=logits)
            ),
            sample=lambda logits_output, batch: next_ids,
        )

        with get_context().override_server_args(
            model_path="dummy", device="cpu", page_size=1
        ):
            reqs = one_batch.prepare_synthetic_inputs_for_latency_test(
                batch_size=2, input_len=3, custom_inputs=[[1, 2, 3], [4, 5, 6]]
            )
            # Inference metadata/model execution is outside this allocation regression.
            with patch.object(
                one_batch.ForwardBatch, "init_new", return_value=object()
            ):
                actual_ids, actual_logits, batch = one_batch.extend(reqs, runner)

        torch.testing.assert_close(batch.out_cache_loc, torch.arange(1, 7))
        for req, slots in zip(reqs, ([1, 2, 3], [4, 5, 6])):
            torch.testing.assert_close(
                request_pool.req_to_token[req.kv.req_pool_idx, :3],
                torch.tensor(slots, dtype=torch.int32),
            )
        self.assertEqual(allocator.available_size(), 10)
        self.assertIs(actual_ids, next_ids)
        self.assertIs(actual_logits, logits)


if __name__ == "__main__":
    unittest.main()
