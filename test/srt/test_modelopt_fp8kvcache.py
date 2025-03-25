import unittest

from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp8Config,
    ModelOptFp8KVCacheMethod,
)
from sglang.test.test_utils import CustomTestCase


class TestModelOptFp8KVCacheMethod(CustomTestCase):
    def test_kv_cache_method_initialization(self):
        """Test that ModelOptFp8KVCacheMethod can be instantiated and
        inherits from BaseKVCacheMethod."""
        # Create a ModelOptFp8Config object
        quant_config = ModelOptFp8Config(is_checkpoint_fp8_serialized=True)

        # Instantiate the KV cache method
        kv_cache_method = ModelOptFp8KVCacheMethod(quant_config)

        # Check inheritance
        self.assertIsInstance(kv_cache_method, BaseKVCacheMethod)

        # Check that the quant_config is stored
        self.assertEqual(kv_cache_method.quant_config, quant_config)


if __name__ == "__main__":
    unittest.main()
