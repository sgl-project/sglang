import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.quantization.quark.utils import should_ignore_layer
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP, _mtp_quant_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=13, suite="base-a-test-cpu")

# The real `mtp.*` exclude entries of the two AMD Quark MXFP4 checkpoints.
# Quark names layers in the checkpoint namespace, where the draft is prefixed
# `mtp.`. The two lists are identical apart from the routed experts:
# Qwen3.5-397B excludes all 512x3 draft expert projections, so its whole MTP
# module is bf16; Qwen3.8-2.4T excludes none of them, so its draft experts stay
# MXFP4 while attention, the shared expert and fc are bf16.
_MIXED_EXCLUDES = [  # amd/Qwen3.8-2.4T-A95B-Quark-MXFP4 (all 10 mtp.* entries)
    "mtp.fc",
    "mtp.layers.0.mlp.gate",
    "mtp.layers.0.mlp.shared_expert.down_proj",
    "mtp.layers.0.mlp.shared_expert.gate_proj",
    "mtp.layers.0.mlp.shared_expert.up_proj",
    "mtp.layers.0.mlp.shared_expert_gate",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.o_proj",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.v_proj",
]
_ALL_BF16_EXCLUDES = _MIXED_EXCLUDES + [  # amd/Qwen3.5-397B-A17B-MXFP4
    f"mtp.layers.0.mlp.experts.{expert}.{proj}"
    for expert in range(512)
    for proj in ("gate_proj", "up_proj", "down_proj")
]


class _FakeQuantConfig:
    def __init__(self, name, exclude_layers):
        self._name = name
        self.exclude_layers = list(exclude_layers)

    def get_name(self):
        return self._name


class TestQwen3_5MTPQuantConfig(CustomTestCase):
    def test_mixed_quark_checkpoint_keeps_quantization(self):
        """Routed experts stay MXFP4, so the draft must stay quantized."""
        quant_config = _FakeQuantConfig("quark", _MIXED_EXCLUDES)

        self.assertIs(_mtp_quant_config(quant_config), quant_config)

    def test_fully_bf16_quark_checkpoint_skips_quantization(self):
        """Regression guard for #23146: a bf16 draft must stay unquantized."""
        quant_config = _FakeQuantConfig("quark", _ALL_BF16_EXCLUDES)

        self.assertIsNone(_mtp_quant_config(quant_config))

    def test_quark_checkpoint_without_mtp_excludes_keeps_quantization(self):
        quant_config = _FakeQuantConfig("quark", ["model.layers.0.self_attn.q_proj"])

        self.assertIs(_mtp_quant_config(quant_config), quant_config)

    def test_non_quark_quant_config_is_untouched(self):
        quant_config = _FakeQuantConfig("fp8", _MIXED_EXCLUDES)

        self.assertIs(_mtp_quant_config(quant_config), quant_config)

    def test_the_loader_reads_the_quant_config_the_mtp_keeps(self):
        """Before the draft is built, the loader answers with the quant config
        the MTP head keeps, so its `auto` KV cache dtype can be resolved then:
        a serialized NVFP4 checkpoint whose KV cache is FP8 ships its MTP head
        in BF16, so the target keeps FP8 KV and the draft BF16."""
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
        from sglang.srt.model_loader import loader

        checkpoint = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo="FP8",
            group_size=16,
            exclude_modules=[],
        )
        for model_class, kept in (
            (Qwen3_5ForCausalLM, checkpoint),
            (Qwen3_5ForCausalLMMTP, None),
        ):
            with (
                self.subTest(model_class=model_class.__name__),
                patch.object(
                    loader, "_get_quantization_config", return_value=checkpoint
                ),
                patch.object(
                    loader,
                    "get_model_architecture",
                    return_value=(model_class, model_class.__name__),
                ),
            ):
                built = loader.built_quant_config(
                    SimpleNamespace(hf_config=None), object()
                )
                self.assertIs(built, kept)

    def test_mtp_reuses_target_packed_modules_mapping(self):
        self.assertEqual(
            Qwen3_5ForCausalLMMTP.packed_modules_mapping,
            Qwen3_5ForCausalLM.packed_modules_mapping,
        )

    def test_excluded_fused_attention_projection_is_ignored(self):
        """The mapping has to expand qkv_proj before the exclude list matches.

        Without it `should_ignore_layer` compares the fused name against
        `q_proj`/`k_proj`/`v_proj` entries, finds nothing, and the layer is
        quantized against bf16 checkpoint shards.
        """
        ignored = should_ignore_layer(
            "mtp.layers.0.self_attn.qkv_proj",
            ignore=_MIXED_EXCLUDES,
            fused_mapping=Qwen3_5ForCausalLMMTP.packed_modules_mapping,
        )

        self.assertTrue(ignored)

    def test_unexcluded_fused_attention_projection_is_not_ignored(self):
        ignored = should_ignore_layer(
            "model.layers.0.self_attn.qkv_proj",
            ignore=_MIXED_EXCLUDES,
            fused_mapping=Qwen3_5ForCausalLMMTP.packed_modules_mapping,
        )

        self.assertFalse(ignored)


if __name__ == "__main__":
    unittest.main()
