import unittest
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestCompressStateCustomMemPool(unittest.TestCase):
    def test_hip_allocation_uses_custom_mem_pool(self):
        state_pool = object.__new__(CompressStatePool)
        state_pool._size = 4
        state_pool.last_dim = 8

        custom_pool = object()
        tensor = MagicMock()
        use_mem_pool = MagicMock(return_value=nullcontext())

        with (
            patch("sglang.srt.mem_cache.deepseek_v4_compress_state._is_hip", True),
            patch(
                "sglang.srt.mem_cache.deepseek_v4_compress_state.maybe_init_custom_mem_pool",
                return_value=(True, custom_pool, "test_pool"),
            ) as init_pool,
            patch.object(torch.cuda, "use_mem_pool", use_mem_pool),
            patch.object(torch, "empty", return_value=tensor) as empty,
        ):
            state_pool._alloc_kv_score_buffer(
                dtype=torch.float16,
                device="cuda:0",
                enable_memory_saver=False,
            )

        init_pool.assert_called_once_with(device="cuda:0")
        use_mem_pool.assert_called_once_with(custom_pool)
        empty.assert_called_once_with(
            (state_pool._size, state_pool.last_dim),
            dtype=torch.float16,
            device="cuda:0",
        )
        self.assertIs(state_pool.custom_mem_pool, custom_pool)
        self.assertIs(state_pool.kv_score_buffer.kv_score, tensor)


if __name__ == "__main__":
    unittest.main()
