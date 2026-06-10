"""Unit tests for TRTLLMHAAttnBackend._resolve_swa_kv_pool."""

import unittest
from unittest.mock import MagicMock

from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-large")

_resolve = TRTLLMHAAttnBackend._resolve_swa_kv_pool


def _mock_runner(
    *,
    active_pool=None,
    is_draft_worker=False,
    spec_algorithm=SpeculativeAlgorithm.NONE,
    allocator_kvcache=None,
):
    runner = MagicMock()
    runner.token_to_kv_pool = active_pool
    runner.is_draft_worker = is_draft_worker
    runner.spec_algorithm = spec_algorithm
    runner.token_to_kv_pool_allocator.get_kvcache.return_value = allocator_kvcache
    return runner


class TestResolveSwaKvPool(CustomTestCase):
    def test_active_pool_is_swa_returns_it(self):
        swa = MagicMock(spec=SWAKVPool)
        runner = _mock_runner(active_pool=swa)
        self.assertIs(_resolve(runner), swa)

    def test_non_swa_active_pool_falls_through_to_allocator(self):
        swa = MagicMock(spec=SWAKVPool)
        runner = _mock_runner(active_pool=MagicMock(), allocator_kvcache=swa)
        self.assertIs(_resolve(runner), swa)

    def test_allocator_kvcache_not_swa_returns_none(self):
        runner = _mock_runner(active_pool=MagicMock(), allocator_kvcache=MagicMock())
        self.assertIsNone(_resolve(runner))

    def test_draft_worker_non_frozen_kv_returns_none(self):
        runner = _mock_runner(
            active_pool=MagicMock(),
            is_draft_worker=True,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            allocator_kvcache=MagicMock(spec=SWAKVPool),
        )
        self.assertIsNone(_resolve(runner))

    def test_draft_worker_frozen_kv_mtp_returns_allocator_swa(self):
        swa = MagicMock(spec=SWAKVPool)
        runner = _mock_runner(
            active_pool=MagicMock(),
            is_draft_worker=True,
            spec_algorithm=SpeculativeAlgorithm.FROZEN_KV_MTP,
            allocator_kvcache=swa,
        )
        self.assertIs(_resolve(runner), swa)

    def test_non_draft_worker_ignores_spec_algorithm(self):
        swa = MagicMock(spec=SWAKVPool)
        runner = _mock_runner(
            active_pool=MagicMock(),
            is_draft_worker=False,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            allocator_kvcache=swa,
        )
        self.assertIs(_resolve(runner), swa)


if __name__ == "__main__":
    unittest.main()
