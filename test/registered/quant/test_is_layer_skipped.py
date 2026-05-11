import unittest

from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


# Qwen3-Next FP8 actually publishes the equivalent of this in
# packed_modules_mapping (qwen3_next.py:908-911). in_proj_ba / in_proj_qkvz
# are deliberately omitted because they are real, unified tensors.
QWEN3_NEXT_FUSED_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _qwen3_next_ignored_layers(layer_idx: int, name: str) -> list:
    # Mirrors the normalization in Fp8Config.from_config: each entry is kept in
    # both "model.<...>" and bare "<...>" forms.
    base = f"layers.{layer_idx}.linear_attn.{name}"
    return [base, f"model.{base}"]


class TestIsLayerSkipped(CustomTestCase):
    def test_qwen3_next_in_proj_ba_is_skipped(self):
        # Regression for #23467: in_proj_ba is a unified tensor in the FP8
        # checkpoint. modules_to_not_convert lists it explicitly, so it must
        # bypass FP8 quantization (otherwise validate_block_quant_shapes raises
        # on output_partition_size=8 vs block_n=128 at tp=4).
        prefix = "model.layers.0.linear_attn.in_proj_ba"
        ignored = _qwen3_next_ignored_layers(0, "in_proj_ba")
        self.assertTrue(is_layer_skipped(prefix, ignored, QWEN3_NEXT_FUSED_MAPPING))

    def test_qwen3_next_in_proj_qkvz_is_skipped(self):
        prefix = "model.layers.5.linear_attn.in_proj_qkvz"
        ignored = _qwen3_next_ignored_layers(5, "in_proj_qkvz")
        self.assertTrue(is_layer_skipped(prefix, ignored, QWEN3_NEXT_FUSED_MAPPING))

    def test_mlp_gate_does_not_match_gate_up_proj(self):
        # The motivation for #23467: an entry "mlp.gate" in
        # modules_to_not_convert must NOT skip a sibling "mlp.gate_up_proj".
        ignored = ["mlp.gate"]
        self.assertFalse(
            is_layer_skipped("model.layers.0.mlp.gate_up_proj", ignored, {})
        )
        self.assertTrue(is_layer_skipped("model.layers.0.mlp.gate", ignored, {}))


if __name__ == "__main__":
    unittest.main()
