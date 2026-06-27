import unittest
from unittest.mock import patch

from sglang.srt.model_executor.forward_batch_deepseek_mha_mixin import (
    ForwardBatchDeepSeekMHAMixin,
)
from sglang.srt.utils.common import CUDA_GRID_DIM_YZ_LIMIT


class TestForwardBatchDeepSeekMHAMixin(unittest.TestCase):
    @patch(
        "sglang.srt.model_executor.forward_batch_deepseek_mha_mixin.envs"
        ".SGLANG_MAX_KV_CHUNK_CAPACITY.get",
        return_value=128 * 1024,
    )
    @patch(
        "sglang.srt.model_executor.forward_batch_deepseek_mha_mixin.get_device_sm",
        return_value=100,
    )
    def test_sm100_caps_mha_chunk_capacity(self, _mock_sm, _mock_env):
        mixin = ForwardBatchDeepSeekMHAMixin()

        self.assertEqual(mixin.get_max_chunk_capacity(), CUDA_GRID_DIM_YZ_LIMIT)

    @patch(
        "sglang.srt.model_executor.forward_batch_deepseek_mha_mixin.envs"
        ".SGLANG_MAX_KV_CHUNK_CAPACITY.get",
        return_value=128 * 1024,
    )
    @patch(
        "sglang.srt.model_executor.forward_batch_deepseek_mha_mixin.get_device_sm",
        return_value=90,
    )
    def test_non_sm100_preserves_mha_chunk_capacity(self, _mock_sm, _mock_env):
        mixin = ForwardBatchDeepSeekMHAMixin()

        self.assertEqual(mixin.get_max_chunk_capacity(), 128 * 1024)


if __name__ == "__main__":
    unittest.main()
