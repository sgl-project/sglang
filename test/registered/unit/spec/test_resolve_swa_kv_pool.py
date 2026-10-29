"""Unit tests for attention-backend window KV pool resolution."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.layers.attention.base_attn_backend import resolve_window_kv_pool
from sglang.srt.mem_cache.kv.router import KVLayerRouter
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=11, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=8, suite="stage-b-test-1-gpu-large-amd")


def _pool(*, window: bool):
    pool = SimpleNamespace()
    layer_maps = [("full", None, {0: 0})]
    if window:
        layer_maps.append(("swa", None, {1: 0}))
    pool.layer_router = KVLayerRouter.from_layer_maps(pool, layer_maps)
    return pool


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
    def test_active_pool_with_window_returns_it(self):
        swa = _pool(window=True)
        runner = _mock_runner(active_pool=swa)
        self.assertIs(resolve_window_kv_pool(runner), swa)

    def test_active_pool_without_window_falls_through_to_allocator(self):
        swa = _pool(window=True)
        runner = _mock_runner(active_pool=_pool(window=False), allocator_kvcache=swa)
        self.assertIs(resolve_window_kv_pool(runner), swa)

    def test_allocator_kvcache_without_window_returns_none(self):
        runner = _mock_runner(
            active_pool=_pool(window=False), allocator_kvcache=_pool(window=False)
        )
        self.assertIsNone(resolve_window_kv_pool(runner))

    def test_draft_worker_non_frozen_kv_returns_none(self):
        runner = _mock_runner(
            active_pool=_pool(window=False),
            is_draft_worker=True,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            allocator_kvcache=_pool(window=True),
        )
        self.assertIsNone(resolve_window_kv_pool(runner))

    def test_draft_worker_frozen_kv_mtp_returns_allocator_pool(self):
        swa = _pool(window=True)
        runner = _mock_runner(
            active_pool=_pool(window=False),
            is_draft_worker=True,
            spec_algorithm=SpeculativeAlgorithm.FROZEN_KV_MTP,
            allocator_kvcache=swa,
        )
        self.assertIs(resolve_window_kv_pool(runner), swa)

    def test_non_draft_worker_ignores_spec_algorithm(self):
        swa = _pool(window=True)
        runner = _mock_runner(
            active_pool=_pool(window=False),
            is_draft_worker=False,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            allocator_kvcache=swa,
        )
        self.assertIs(resolve_window_kv_pool(runner), swa)


if __name__ == "__main__":
    unittest.main()
