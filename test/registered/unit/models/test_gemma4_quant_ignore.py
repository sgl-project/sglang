import unittest

from sglang.srt.layers.quantization.compressed_tensors.utils import should_ignore_layer
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Vision-tower entries of the compressed-tensors `ignore` list in
# cyankiwi/gemma-4-31B-it-AWQ-4bit: clippable linears are named by their `.linear` module.
_CHECKPOINT_IGNORE = [
    f"model.vision_tower.encoder.layers.0.{proj}.linear"
    for proj in (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
]


def _ignored(layer_name: str) -> bool:
    model = Gemma4ForConditionalGeneration
    return should_ignore_layer(
        layer_name,
        ignore=model.hf_to_sglang_mapper.apply_list(_CHECKPOINT_IGNORE),
        fused_mapping=model.packed_modules_mapping,
    )


class TestGemma4QuantIgnore(CustomTestCase):
    def test_ignored_fused_vision_projections_stay_unquantized(self):
        """Fused vision qkv/gate_up must follow the checkpoint's per-shard ignore entries.

        Otherwise they are built as packed int4 layers over bf16 checkpoint shards
        and stay uninitialized (sglang#43342).
        """
        self.assertTrue(
            _ignored("model.vision_tower.encoder.layers.0.self_attn.qkv_proj")
        )
        self.assertTrue(
            _ignored("model.vision_tower.encoder.layers.0.mlp.gate_up_proj")
        )
        self.assertTrue(
            _ignored("model.vision_tower.encoder.layers.0.self_attn.o_proj.linear")
        )

    def test_language_model_fused_projections_stay_quantized(self):
        self.assertFalse(_ignored("model.language_model.layers.0.self_attn.qkv_proj"))
        self.assertFalse(_ignored("model.language_model.layers.0.mlp.gate_up_proj"))


if __name__ == "__main__":
    unittest.main()
