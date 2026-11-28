import copy
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

    @staticmethod
    def _qwen2_5_vl_mapping():
        from sglang.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration.packed_modules_mapping

    def test_qwen2_5_vl_text_qkv_follows_its_listed_shards(self):
        """Listing the text q/k/v makes the text self_attn.qkv_proj dense."""
        not_convert = ["attn.q_proj", "attn.k_proj", "attn.v_proj"]
        self.assertTrue(
            _skipped(
                "model.layers.0.self_attn.qkv_proj",
                not_convert,
                self._qwen2_5_vl_mapping(),
            )
        )

    def test_qwen2_5_vl_vision_qkv_is_one_tensor(self):
        """The vision attn.qkv_proj is not stacked, so text shard names leave it quantized."""
        not_convert = ["attn.q_proj", "attn.k_proj", "attn.v_proj"]
        self.assertFalse(
            _skipped(
                "visual.blocks.0.attn.qkv_proj",
                not_convert,
                self._qwen2_5_vl_mapping(),
            )
        )
        self.assertTrue(
            _skipped(
                "visual.blocks.0.attn.qkv_proj", ["visual"], self._qwen2_5_vl_mapping()
            )
        )

    def test_shard_name_match_does_not_decide_for_the_other_shards(self):
        """Whisper: "v_proj" inside "kv_proj" leaves K quantized, so it is rejected."""
        mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "kv_proj": ["k_proj", "v_proj"],
        }
        with self.assertRaisesRegex(ValueError, "same precision"):
            _skipped("model.decoder.layers.0.encoder_attn.kv_proj", ["v_proj"], mapping)

    def test_unlisted_v_proj_does_not_follow_k_proj(self):
        """Whisper self-attention has a real v_proj, so dense q/k alone is rejected."""
        mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "kv_proj": ["k_proj", "v_proj"],
        }
        with self.assertRaisesRegex(ValueError, "same precision"):
            _skipped(
                "model.decoder.layers.0.self_attn.qkv_proj",
                ["self_attn.q_proj", "self_attn.k_proj"],
                mapping,
            )

    def test_gemma4_k_eq_v_layer_resolves_from_q_and_k(self):
        """Gemma 4 attention_k_eq_v layers ship q_proj and k_proj only."""
        from types import SimpleNamespace

        from sglang.srt.models.gemma4_causal import _declare_k_eq_v_qkv_shards

        mapping_before = copy.deepcopy(QWEN3_5_MAPPING)
        text_config = SimpleNamespace(
            attention_k_eq_v=True,
            layer_types=["sliding_attention", "full_attention"],
        )
        # Gemma4ForCausalLM and Gemma4ForConditionalGeneration text layer paths.
        for layers_prefix in ("model.layers", "model.language_model.layers"):
            with self.subTest(layers_prefix=layers_prefix):
                not_convert = [
                    f"{layers_prefix}.{i}.self_attn.{shard}"
                    for i in (0, 1)
                    for shard in ("q_proj", "k_proj")
                ]
                quant_config = _awq_config(not_convert, QWEN3_5_MAPPING)
                _declare_k_eq_v_qkv_shards(text_config, quant_config, layers_prefix)

                def skipped(prefix):
                    return is_layer_skipped_awq(
                        prefix,
                        modules_to_not_convert=not_convert,
                        quant_config=quant_config,
                    )

                # Full-attention layer 1 has no v_proj in the checkpoint.
                self.assertTrue(skipped(f"{layers_prefix}.1.self_attn.qkv_proj"))
                # Sliding layer 0 has a real v_proj, which the list leaves quantized.
                with self.assertRaisesRegex(ValueError, "same precision"):
                    skipped(f"{layers_prefix}.0.self_attn.qkv_proj")
        # The model class mapping itself is left untouched.
        self.assertEqual(QWEN3_5_MAPPING, mapping_before)

    def test_router_entry_does_not_split_gate_up_proj(self):
        """ "mlp.gate" names the MoE router, not the gate_proj shard of a dense MLP."""
        for not_convert in (["mlp.gate"], ["gate"]):
            with self.subTest(not_convert=not_convert):
                self.assertFalse(
                    _skipped("model.layers.0.mlp.gate_up_proj", not_convert)
                )
                self.assertFalse(
                    _skipped(
                        "model.layers.3.mlp.shared_experts.gate_up_proj", not_convert
                    )
                )

    def test_layer_entry_does_not_reach_later_layers(self):
        """ "model.layers.1" names layer 1, not layers 10-19."""
        not_convert = ["model.layers.1"]
        self.assertTrue(_skipped("model.layers.1.self_attn.qkv_proj", not_convert))
        self.assertFalse(_skipped("model.layers.12.self_attn.qkv_proj", not_convert))

    def test_partial_model_mapping_keeps_the_fallback(self):
        """DeepSeek-V3 declares only fused_qkv_a_proj_with_mqa; gate_up_proj still expands."""
        mapping = {"fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"]}
        not_convert = ["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"]
        self.assertTrue(
            _skipped("model.layers.0.mlp.gate_up_proj", not_convert, mapping)
        )

    def test_leading_dot_mapping_keys_are_leaf_names(self):
        """Gemma3n writes ".qkv_proj": [".q_proj", ...]."""
        mapping = {
            ".qkv_proj": [".q_proj", ".k_proj", ".v_proj"],
            ".gate_up_proj": [".gate_proj", ".up_proj"],
        }
        not_convert = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
        self.assertTrue(
            _skipped("model.layers.0.self_attn.qkv_proj", not_convert, mapping)
        )

    def test_chatglm_prefix_entry_covers_both_dense_layers(self):
        """AutoAWQ matches by substring, so "mlp.dense_" skips dense_h_to_4h and dense_4h_to_h."""
        from sglang.srt.models.chatglm import ChatGLMForCausalLM

        mapping = ChatGLMForCausalLM.packed_modules_mapping
        for name in ("dense_h_to_4h", "dense_4h_to_h"):
            with self.subTest(name=name):
                self.assertTrue(
                    _skipped(
                        f"transformer.encoder.layers.0.mlp.{name}",
                        ["mlp.dense_"],
                        mapping,
                    )
                )

    def test_phi3_fused_tensors_keep_substring_entries(self):
        """Phi-3 stores qkv_proj and gate_up_proj fused, so "attn.qkv_proj" covers it."""
        from sglang.srt.models.llama import Phi3ForCausalLM

        mapping = Phi3ForCausalLM.packed_modules_mapping
        for name, entry in (
            ("self_attn.qkv_proj", "attn.qkv_proj"),
            ("mlp.gate_up_proj", "mlp.gate_up_proj"),
        ):
            with self.subTest(name=name):
                prefix = f"model.layers.0.{name}"
                self.assertTrue(_skipped(prefix, [entry], mapping))
                self.assertFalse(_skipped(prefix, ["o_proj"], mapping))

    def test_bailing_linear_attention_qkv_is_one_tensor(self):
        """Bailing's linear-attention qkv_proj loads one query_key_value tensor."""
        from sglang.srt.models.bailing_moe_linear import BailingMoELinearForCausalLM

        mapping = BailingMoELinearForCausalLM.packed_modules_mapping
        prefix = "model.layers.0.attention.qkv_proj"
        # An MLA-layer entry neither splits nor raises on the linear layer.
        self.assertFalse(_skipped(prefix, ["attention.q_proj"], mapping))
        # "v_proj" is the tail of the runtime name qkv_proj, not a module.
        self.assertFalse(_skipped(prefix, ["v_proj"], mapping))
        self.assertTrue(_skipped(prefix, ["attention.query_key_value"], mapping))
        self.assertTrue(_skipped(prefix, ["attention.qkv_proj"], mapping))

    def test_gemma4_k_eq_v_skips_configs_without_the_awq_matcher(self):
        """w8a8_fp8 has no packed_modules_mapping and is left untouched."""
        from types import SimpleNamespace

        from sglang.srt.layers.quantization.w8a8_fp8 import W8A8Fp8Config
        from sglang.srt.models.gemma4_causal import _declare_k_eq_v_qkv_shards

        text_config = SimpleNamespace(
            attention_k_eq_v=True, layer_types=["full_attention"]
        )
        quant_config = W8A8Fp8Config()
        state_before = copy.deepcopy(vars(quant_config))
        _declare_k_eq_v_qkv_shards(text_config, quant_config, "model.layers")
        self.assertEqual(vars(quant_config), state_before)

    def test_auto_round_k_eq_v_layer_follows_its_q_and_k(self):
        """AutoRound takes Q/K's bits=16 for a K=V qkv_proj, which has no v_proj."""
        from types import SimpleNamespace

        import torch

        from sglang.srt.layers.quantization.auto_round import AutoRoundConfig
        from sglang.srt.models.gemma4_causal import _declare_k_eq_v_qkv_shards

        dense = {"bits": 16}
        quant_config = AutoRoundConfig.from_config(
            {
                "quant_method": "auto-round",
                "bits": 4,
                "group_size": 128,
                "sym": True,
                "extra_config": {
                    "model.layers.1.self_attn.q_proj": dense,
                    "model.layers.1.self_attn.k_proj": dense,
                },
            }
        )
        _declare_k_eq_v_qkv_shards(
            SimpleNamespace(
                attention_k_eq_v=True,
                layer_types=["sliding_attention", "full_attention"],
            ),
            quant_config,
            "model.layers",
        )
        bits, _, _ = quant_config.get_layer_config(
            torch.nn.Linear(1, 1), "model.layers.1.self_attn.qkv_proj"
        )
        self.assertEqual(bits, 16)

    def test_moe_wna16_awq_linear_keeps_the_k_eq_v_shards(self):
        """moe_wna16 rebuilds AWQ per linear layer and must keep the model's update."""
        from types import SimpleNamespace
        from unittest import mock

        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from sglang.srt.models.gemma4_causal import _declare_k_eq_v_qkv_shards

        not_convert = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
        ]
        full_config = {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "modules_to_not_convert": not_convert,
            "packed_modules_mapping": QWEN3_5_MAPPING,
        }
        # The CPU runner has no device capability; AWQ itself needs none here.
        with mock.patch(
            "sglang.srt.layers.quantization.moe_wna16.get_device_capability",
            return_value=(8, 0),
        ):
            quant_config = MoeWNA16Config.from_config(full_config)
        _declare_k_eq_v_qkv_shards(
            SimpleNamespace(attention_k_eq_v=True, layer_types=["full_attention"]),
            quant_config,
            "model.layers",
        )
        layer = mock.MagicMock(spec=LinearBase)
        method = quant_config.get_quant_method(
            layer, "model.layers.0.self_attn.qkv_proj"
        )
        self.assertIsInstance(method, UnquantizedLinearMethod)

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
