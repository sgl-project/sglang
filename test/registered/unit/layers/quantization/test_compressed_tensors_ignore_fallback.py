"""Unit tests for compressed_tensors should_ignore_layer fused-shard fallback (#52209)."""

import unittest

from sglang.srt.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Models without packed_modules_mapping (LFM2, ...): the quant config carries
# no fused mapping, so should_ignore_layer must fall back to the generic
# qkv_proj / gate_up_proj shard names.
GLM_IGNORE = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
]

# Mapping a config that does declare the fusion (e.g. Qwen3-Next): it must
# keep taking precedence over the fallback.
DECLARED_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


class TestShouldIgnoreLayerFallback(CustomTestCase):
    def test_fused_qkv_ignored_without_fused_mapping(self):
        # Regression for #52209: llm-compressor writes the unfused shard
        # names into `ignore`; with an empty fused_mapping the fused layer
        # used to be quantized even though all its shards are ignored.
        layer = "model.layers.0.self_attn.qkv_proj"
        self.assertTrue(should_ignore_layer(layer, GLM_IGNORE, {}))

    def test_fused_gate_up_ignored_without_fused_mapping(self):
        layer = "model.layers.0.mlp.gate_up_proj"
        self.assertTrue(should_ignore_layer(layer, GLM_IGNORE, {}))

    def test_declared_fused_mapping_still_wins(self):
        layer = "model.layers.0.self_attn.qkv_proj"
        self.assertTrue(should_ignore_layer(layer, GLM_IGNORE, DECLARED_MAPPING))
        self.assertFalse(should_ignore_layer(layer, (), {}))

    def test_layer_listed_directly_is_ignored(self):
        # Workaround configs put the fused name itself into ignore.
        layer = "model.layers.0.self_attn.qkv_proj"
        self.assertTrue(should_ignore_layer(layer, [layer], {}))

    def test_unfused_layer_uses_direct_match(self):
        self.assertTrue(
            should_ignore_layer("model.layers.0.mlp.down_proj", GLM_IGNORE, {})
        )
        self.assertFalse(
            should_ignore_layer("model.layers.1.mlp.down_proj", GLM_IGNORE, {})
        )

    def test_partial_shard_ignore_is_rejected(self):
        # Mixing quantized and dense shards inside one fused layer is
        # unloadable; must fail loudly.
        ignore = ["model.layers.0.mlp.gate_proj"]
        with self.assertRaises(ValueError):
            should_ignore_layer("model.layers.0.mlp.gate_up_proj", ignore, {})


if __name__ == "__main__":
    unittest.main()
