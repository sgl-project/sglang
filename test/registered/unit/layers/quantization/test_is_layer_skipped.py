import unittest

from sglang.srt.layers.quantization.awq.awq import AWQConfig, is_layer_skipped_awq
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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

    def test_trailing_dot_prefix_matches_child_modules(self):
        # Mixed-precision checkpoints may use a trailing-dot layer prefix to keep
        # every module under the layer in higher precision.
        ignored = ["model.layers.34."]
        self.assertTrue(
            is_layer_skipped("model.layers.34.mlp.experts.0.down_proj", ignored, {})
        )
        self.assertFalse(
            is_layer_skipped("model.layers.340.mlp.experts.0.down_proj", ignored, {})
        )


# modules_to_not_convert from QuantTrio/Qwen3.6-27B-AWQ (and Qwen3.5-27B-AWQ),
# which stores the listed q/k/v shards as dense fp16.
QWEN3_6_AWQ_NOT_CONVERT = [
    "visual",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_a",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "model.layers.0.",
    "mtp",
]
QWEN3_6_LAYER_PREFIX = "model.language_model.layers.3"
# Qwen3_5ForConditionalGeneration.packed_modules_mapping (qwen3_5.py).
QWEN3_5_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}


def _awq_config(modules_to_not_convert, packed_modules_mapping=None):
    config = {
        "quant_method": "awq",
        "bits": 4,
        "group_size": 128,
        "zero_point": True,
        "modules_to_not_convert": modules_to_not_convert,
    }
    if packed_modules_mapping is not None:
        config["packed_modules_mapping"] = packed_modules_mapping
    return AWQConfig.from_config(config)


def _skipped(prefix, modules_to_not_convert, packed_modules_mapping=None):
    return is_layer_skipped_awq(
        prefix,
        modules_to_not_convert=modules_to_not_convert,
        quant_config=_awq_config(modules_to_not_convert, packed_modules_mapping),
    )


class TestIsLayerSkippedAWQ(CustomTestCase):
    def test_fused_layer_follows_its_listed_shards(self):
        """A fused qkv_proj is unquantized when the list names all its shards."""
        for mapping in (QWEN3_5_MAPPING, None):
            with self.subTest(declared_mapping=mapping is not None):
                self.assertTrue(
                    _skipped(
                        f"{QWEN3_6_LAYER_PREFIX}.self_attn.qkv_proj",
                        QWEN3_6_AWQ_NOT_CONVERT,
                        mapping,
                    )
                )

    def test_layers_not_listed_stay_quantized(self):
        for name in (
            "self_attn.o_proj",
            "mlp.gate_up_proj",
            "linear_attn.in_proj_qkvz",
        ):
            with self.subTest(name=name):
                self.assertFalse(
                    _skipped(
                        f"{QWEN3_6_LAYER_PREFIX}.{name}",
                        QWEN3_6_AWQ_NOT_CONVERT,
                        QWEN3_5_MAPPING,
                    )
                )

    def test_listed_linear_attention_shards_skip_in_proj_ba(self):
        self.assertTrue(
            _skipped(
                f"{QWEN3_6_LAYER_PREFIX}.linear_attn.in_proj_ba",
                QWEN3_6_AWQ_NOT_CONVERT,
                QWEN3_5_MAPPING,
            )
        )

    def test_fused_name_the_model_does_not_declare_is_one_tensor(self):
        """Qwen2.5-VL leaves vision attn.qkv_proj out of its mapping: it is not split."""
        not_convert = ["attn.q_proj", "attn.k_proj", "attn.v_proj"]
        mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
        self.assertFalse(
            _skipped("visual.blocks.0.attn.qkv_proj", not_convert, mapping)
        )

    def test_shard_name_match_does_not_decide_for_the_other_shards(self):
        """Whisper: "v_proj" inside "kv_proj" leaves K quantized, so it is rejected."""
        mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "kv_proj": ["k_proj", "v_proj"],
        }
        with self.assertRaisesRegex(ValueError, "same precision"):
            _skipped("model.decoder.layers.0.encoder_attn.kv_proj", ["v_proj"], mapping)

    def test_k_eq_v_layer_needs_no_v_proj_entry(self):
        """Gemma 4 attention_k_eq_v layers ship q_proj and k_proj only."""
        not_convert = [
            "model.layers.5.self_attn.q_proj",
            "model.layers.5.self_attn.k_proj",
        ]
        self.assertTrue(
            _skipped("model.layers.5.self_attn.qkv_proj", not_convert, QWEN3_5_MAPPING)
        )

    def test_partially_listed_fused_layer_raises(self):
        with self.assertRaisesRegex(ValueError, "same precision"):
            _skipped(
                f"{QWEN3_6_LAYER_PREFIX}.self_attn.qkv_proj",
                ["self_attn.q_proj"],
                QWEN3_5_MAPPING,
            )

    def test_from_config_reads_the_model_packed_modules_mapping(self):
        """A fusion only the model's packed_modules_mapping knows is expanded."""
        not_convert = ["block.x_proj", "block.y_proj"]
        prefix = f"{QWEN3_6_LAYER_PREFIX}.block.xy_proj"
        self.assertTrue(
            _skipped(prefix, not_convert, {"xy_proj": ["x_proj", "y_proj"]})
        )
        self.assertFalse(_skipped(prefix, not_convert))


if __name__ == "__main__":
    unittest.main()
