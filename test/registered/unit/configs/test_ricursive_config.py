"""RicursiveConfig's KV pool split."""

import unittest

from sglang.srt.configs.model_config import (
    get_hybrid_layer_ids,
    is_layer_types_hybrid_swa_model,
)
from sglang.srt.configs.ricursive import RicursiveConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestRicursiveConfig(CustomTestCase):
    def test_kv_pool_split_comes_from_layer_types(self):
        """The checkpoint also ships a precomputed split, which would skip the
        check of layer_types against the attention layers the model builds."""
        config = RicursiveConfig(
            num_hidden_layers=10,
            sliding_window=1024,
            is_hybrid_swa=True,
            hybrid_layer_pattern=[0] * 10,
        )
        self.assertTrue(is_layer_types_hybrid_swa_model(config.architectures, config))
        swa, full = get_hybrid_layer_ids(config.architectures, config)
        self.assertEqual(full, [4, 9])
        self.assertEqual(swa, [0, 1, 2, 3, 5, 6, 7, 8])


if __name__ == "__main__":
    unittest.main()
