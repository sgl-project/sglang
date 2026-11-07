"""EagerRunner input registry sizing.

The token-axis input slots (input_ids, positions, out_cache_loc, ...) hold one
eager batch: they are sized by the prefill limit rather than the KV pool, and a
larger batch regrows them, outside the shared input-buffer pool and never above
the pool-based size used before.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    CudaGraphBufferRegistry,
)
from sglang.srt.model_executor.input_buffers import _forward_input_buffer_pool
from sglang.srt.model_executor.runner import eager_runner as eager_runner_module
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CHUNK = 4096
MAX_PREFILL_TOKENS = 16384


class TestEagerRunnerInputRegistry(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            chunked_prefill_size=CHUNK, max_prefill_tokens=MAX_PREFILL_TOKENS
        )
        override.install()
        self.addCleanup(override.restore)
        no_dllm = mock.patch.object(
            eager_runner_module.DllmConfig, "from_server_args", return_value=None
        )
        no_dllm.start()
        self.addCleanup(no_dllm.stop)

    def _runner(self, max_total_num_tokens: int) -> EagerRunner:
        model_runner = SimpleNamespace(
            device="cpu",
            is_draft_worker=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            server_args=None,
            max_running_requests=8,
            max_total_num_tokens=max_total_num_tokens,
            model_config=SimpleNamespace(
                is_encoder_decoder=False, hf_config=SimpleNamespace()
            ),
        )
        return EagerRunner(model_runner, run_warmup=False)

    def _input_ids_numel(self, runner: EagerRunner) -> int:
        return runner._eager_registry.get_slot("input_ids").buffer.numel()

    def test_large_kv_pool_does_not_size_the_registry(self):
        runner = self._runner(max_total_num_tokens=10_000_000)
        self.assertEqual(runner._eager_registry.max_num_tokens, MAX_PREFILL_TOKENS)
        self.assertEqual(self._input_ids_numel(runner), MAX_PREFILL_TOKENS)

    def test_small_kv_pool_keeps_the_chunk_ceiling(self):
        runner = self._runner(max_total_num_tokens=1000)
        self.assertEqual(runner._eager_registry.max_num_tokens, CHUNK)

    def test_larger_batch_regrows_the_registry(self):
        runner = self._runner(max_total_num_tokens=10_000_000)
        batch = SimpleNamespace(
            batch_size=1,
            input_ids=torch.zeros(MAX_PREFILL_TOKENS + 1, dtype=torch.int64),
            input_embeds=None,
        )
        with (
            mock.patch.object(CudaGraphBufferRegistry, "fill_from") as fill_from,
            mock.patch.object(CudaGraphBufferRegistry, "extract_buffer"),
        ):
            runner.load_batch(batch)
            grown = runner._eager_registry
            self.assertEqual(grown.max_num_tokens, 2 * MAX_PREFILL_TOKENS)
            self.assertEqual(self._input_ids_numel(runner), 2 * MAX_PREFILL_TOKENS)
            self.assertEqual(
                fill_from.call_args.kwargs["raw_num_tokens"], MAX_PREFILL_TOKENS + 1
            )

            batch.input_ids = torch.zeros(CHUNK, dtype=torch.int64)
            runner.load_batch(batch)
            self.assertIs(runner._eager_registry, grown)

    def test_regrow_is_capped_and_not_pooled(self):
        pool_tokens = 20_000
        runner = self._runner(max_total_num_tokens=pool_tokens)
        batch = SimpleNamespace(
            batch_size=1,
            input_ids=torch.zeros(MAX_PREFILL_TOKENS + 1, dtype=torch.int64),
            input_embeds=None,
        )
        with (
            mock.patch.object(CudaGraphBufferRegistry, "fill_from"),
            mock.patch.object(CudaGraphBufferRegistry, "extract_buffer"),
        ):
            runner.load_batch(batch)
        # The next power of two (32,768) would exceed the old pool-based size.
        self.assertEqual(runner._eager_registry.max_num_tokens, pool_tokens)
        grown = runner._eager_registry.get_slot("input_ids").buffer
        pooled = {
            b.untyped_storage().data_ptr() for b in _forward_input_buffer_pool.values()
        }
        self.assertNotIn(grown.untyped_storage().data_ptr(), pooled)


if __name__ == "__main__":
    unittest.main()
