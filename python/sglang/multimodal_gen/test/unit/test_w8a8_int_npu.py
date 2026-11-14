# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from sglang.multimodal_gen.runtime.layers.linear import (
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.w8a8_int_npu import (
    NPUOnlineW8A8DiffusionConfig,
    NPUOnlineW8A8DiffusionLinearMethod,
)


class TestOnlineW8A8IgnoredLayers(unittest.TestCase):
    def setUp(self):
        platform = patch(
            "sglang.multimodal_gen.runtime.layers.quantization.w8a8_int_npu."
            "current_platform.is_npu",
            return_value=True,
        )
        platform.start()
        self.addCleanup(platform.stop)

    def test_ignored_fused_layer_names(self):
        for name in ("qkv_proj", "gate_up_proj"):
            prefix = f"blocks.0.attn.{name}"
            for ignored in ([prefix], [name], ["blocks.0.attn"]):
                with self.subTest(name=name, ignored=ignored):
                    config = NPUOnlineW8A8DiffusionConfig(ignored_layers=ignored)
                    layer = ReplicatedLinear(4, 8, quant_config=config, prefix=prefix)
                    self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)

    def test_ignored_shards_preserve_consistency_checks(self):
        config = NPUOnlineW8A8DiffusionConfig(
            ignored_layers=["q_proj", "k_proj", "v_proj"]
        )
        layer = ReplicatedLinear(
            4, 8, quant_config=config, prefix="blocks.0.attn.qkv_proj"
        )
        self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)

        config = NPUOnlineW8A8DiffusionConfig(ignored_layers=["q_proj"])
        with self.assertRaisesRegex(ValueError, "some but not all shards"):
            ReplicatedLinear(4, 8, quant_config=config, prefix="blocks.0.attn.qkv_proj")

        layer = ReplicatedLinear(
            4, 8, quant_config=config, prefix="blocks.0.attn.out_proj"
        )
        self.assertIsInstance(layer.quant_method, NPUOnlineW8A8DiffusionLinearMethod)


if __name__ == "__main__":
    unittest.main()
