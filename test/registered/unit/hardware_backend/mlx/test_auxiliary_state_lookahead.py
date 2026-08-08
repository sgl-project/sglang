"""Regression tests for auxiliary-state snapshots across MLX lookahead decode."""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None


@unittest.skipUnless(_HAS_MLX, "requires mlx")
class TestAuxiliaryStateLookahead(CustomTestCase):
    def test_finished_step_snapshots_before_already_launched_lookahead(self):
        import mlx.core as mx
        import torch

        from sglang.srt.hardware_backend.mlx.kv_cache import MlxAuxiliaryStatePool
        from sglang.srt.hardware_backend.mlx.model_runner import (
            MlxModelRunner,
            MlxPendingDecode,
        )

        pool = MlxAuxiliaryStatePool(size=2, device="cpu")

        class _ReqPool:
            auxiliary_state_pool = pool

            @staticmethod
            def get_auxiliary_state_indices(_req_pool_idx):
                return torch.tensor([1], dtype=torch.int64)

        cache = SimpleNamespace(state=(mx.array([1], dtype=mx.int32),))
        runner = object.__new__(MlxModelRunner)
        runner._cache_layout = SimpleNamespace(
            has_auxiliary_state=True,
            auxiliary_layer_indices=[0],
        )
        runner._req_to_token_pool = _ReqPool()
        runner._req_pool_idx = {"r0": 0}
        runner._req_caches = {"r0": [cache]}
        runner._req_token_ids = {"r0": [7]}
        runner._decode_step_ct = 0
        runner._clear_steps = 0

        def launch_lookahead(caches, _input_ids, _req_ids):
            # Graph construction for token N+1 advances the shared native cache
            # before token N has gone through scheduler finish detection.
            caches[0][0].state = (mx.array([2], dtype=mx.int32),)
            return mx.array([10], dtype=mx.int32)

        runner._decode_with_hybrid_batching = launch_lookahead
        prev = MlxPendingDecode(
            lazy_tokens=mx.array([9], dtype=mx.int32),
            req_ids=["r0"],
            caches=[[cache]],
        )

        runner.decode_batch_start_chained(prev)
        self.assertEqual(cache.state[0].item(), 2)

        # Token 9 is then found to finish the request. The radix snapshot must
        # describe the committed N boundary (state=1), not discarded N+1.
        runner.decode_batch_finalize(prev)
        runner.store_auxiliary_state_for_request("r0")

        restored = SimpleNamespace(state=(mx.array([0], dtype=mx.int32),))
        self.assertTrue(pool.restore_cache(torch.tensor([1]), [restored], [0]))
        self.assertEqual(restored.state[0].item(), 1)


if __name__ == "__main__":
    unittest.main()
