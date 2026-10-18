import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4LinearMethod,
    ModelOptMixedPrecisionConfig,
)
from sglang.srt.layers.quantization.quark.utils import should_ignore_layer
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP, _mtp_quant_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

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
    def __init__(self, name, exclude_layers, **attrs):
        self._name = name
        self.exclude_layers = list(exclude_layers)
        self.__dict__.update(attrs)

    def get_name(self):
        return self._name


class _DraftModelBoundary(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, input_ids, positions, forward_batch, hidden_states):
        return hidden_states + 1


class TestQwen3_5MTPQuantConfig(CustomTestCase):
    def _make_mtp_model(self, quant_config):
        config = SimpleNamespace(
            hidden_size=16,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True,
            vocab_size=32,
            num_hidden_layers=4,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            full_attention_interval=4,
        )
        parallel = SimpleNamespace(pp_group=SimpleNamespace(is_last_rank=True))

        with (
            mock.patch(
                "sglang.srt.models.qwen3_5_mtp.get_parallel", return_value=parallel
            ),
            mock.patch(
                "sglang.srt.models.qwen3_5_mtp.Qwen3_5ForCausalLM",
                side_effect=lambda child_config, *args, **kwargs: _DraftModelBoundary(
                    child_config
                ),
            ),
            mock.patch(
                "sglang.srt.models.qwen3_5_mtp.LogitsProcessor",
                return_value=lambda input_ids, hidden_states, lm_head, batch: (
                    hidden_states
                ),
            ),
        ):
            return Qwen3_5ForCausalLMMTP(config, quant_config)

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

    def test_serialized_fp4_without_mtp_metadata_stays_bf16(self):
        """Serialized FP4 checkpoints omit MTP weights unless metadata opts in."""
        quant_config = _FakeQuantConfig(
            "modelopt_fp4",
            [],
            is_checkpoint_nvfp4_serialized=True,
            quantized_layers={},
        )

        self.assertIsNone(_mtp_quant_config(quant_config))

    def test_serialized_fp4_with_mtp_metadata_is_quantized(self):
        """An explicit mtp.* entry proves the serialized draft has FP4 weights."""
        quant_config = _FakeQuantConfig(
            "modelopt_fp4",
            [],
            is_checkpoint_nvfp4_serialized=True,
            quantized_layers={"mtp.fc": {"quant_algo": "NVFP4"}},
        )

        self.assertIs(_mtp_quant_config(quant_config), quant_config)

    def test_mixed_precision_still_requires_mtp_metadata(self):
        no_mtp = _FakeQuantConfig("modelopt_mixed", [], quantized_layers={})
        with_mtp = _FakeQuantConfig(
            "modelopt_mixed", [], quantized_layers={"mtp.fc": {}}
        )

        self.assertIsNone(_mtp_quant_config(no_mtp))
        self.assertIs(_mtp_quant_config(with_mtp), with_mtp)

    def test_modelopt_mtp_loads_packed_fc_and_scales(self):
        """A plain fc cannot load the packed NVFP4 draft weight and its scales."""
        quant_config = ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_algo": "MIXED_PRECISION",
                "packed_modules_mapping": Qwen3_5ForCausalLMMTP.packed_modules_mapping,
                "quantized_layers": {
                    "mtp.fc": {"quant_algo": "NVFP4", "group_size": 16}
                },
            }
        )
        model = self._make_mtp_model(quant_config)
        self.assertIsInstance(model.fc, ReplicatedLinear)
        self.assertIsInstance(model.fc.quant_method, ModelOptFp4LinearMethod)
        self.assertEqual(model.fc.weight.shape, (16, 16))
        self.assertEqual(model.fc.weight.dtype, torch.uint8)
        self.assertEqual(model.fc.weight_scale.shape, (16, 2))
        self.assertEqual(model.fc.weight_scale.dtype, torch.float8_e4m3fn)
        for name in ("input_scale", "weight_scale_2"):
            scale = getattr(model.fc, name)
            self.assertEqual(scale.shape, (1,))
            self.assertEqual(scale.dtype, torch.float32)

        packed_weight = (
            torch.arange(256, dtype=torch.int32).reshape(16, 16).to(torch.uint8)
        )
        block_scales = torch.full((16, 2), 1.5).to(torch.float8_e4m3fn)
        loaded = model.load_weights(
            [
                ("mtp.fc.weight", packed_weight),
                ("mtp.fc.weight_scale", block_scales),
                ("mtp.fc.input_scale", torch.tensor(0.5)),
                ("mtp.fc.weight_scale_2", torch.tensor(2.0)),
            ]
        )

        self.assertEqual(
            loaded,
            {"fc.weight", "fc.weight_scale", "fc.input_scale", "fc.weight_scale_2"},
        )
        torch.testing.assert_close(model.fc.weight, packed_weight, rtol=0, atol=0)
        torch.testing.assert_close(
            model.fc.weight_scale.float(), block_scales.float(), rtol=0, atol=0
        )
        torch.testing.assert_close(model.fc.input_scale, torch.tensor([0.5]))
        torch.testing.assert_close(model.fc.weight_scale_2, torch.tensor([2.0]))
        self.assertEqual(model.model.config.num_hidden_layers, 1)
        self.assertEqual(model.model.config.layer_types, ["full_attention"])
        self.assertEqual(model.model.config.full_attention_interval, 1)

    def test_unrelated_fp8_keeps_plain_fc(self):
        """Adding ModelOpt fc support must preserve the existing FP8 draft layout."""
        model = self._make_mtp_model(Fp8Config(is_checkpoint_fp8_serialized=True))

        self.assertIsInstance(model.fc, torch.nn.Linear)
        self.assertEqual(model.fc.weight.shape, (16, 32))

    def test_modelopt_mtp_forward_unpacks_replicated_fc_output(self):
        """The draft decoder needs the tensor from fc, not its (tensor, bias) pair."""
        quant_config = ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_algo": "MIXED_PRECISION",
                "packed_modules_mapping": Qwen3_5ForCausalLMMTP.packed_modules_mapping,
                "quantized_layers": {
                    "mtp.layers.0.self_attn.q_proj": {
                        "quant_algo": "NVFP4",
                        "group_size": 16,
                    }
                },
            }
        )
        model = self._make_mtp_model(quant_config)
        # The explicit child layer opts MTP in, while the omitted fc stays BF16.
        self.assertIsInstance(model.fc, ReplicatedLinear)
        self.assertIsInstance(model.fc.quant_method, UnquantizedLinearMethod)
        model.fc.to(dtype=torch.bfloat16)
        projection = torch.arange(512, dtype=torch.float32).reshape(16, 32) / 512
        projection = projection.to(model.fc.weight.dtype)
        model.load_weights([("mtp.fc.weight", projection)])
        input_embeds = torch.arange(32, dtype=projection.dtype).reshape(2, 16) / 32
        target_hidden = torch.flip(input_embeds, dims=[1])
        batch = SimpleNamespace(
            mm_input_embeds=input_embeds,
            spec_info=SimpleNamespace(hidden_states=target_hidden),
            forward_mode=ForwardMode.IDLE,
        )
        recorder = SimpleNamespace(disable_this_region=lambda: nullcontext())

        with mock.patch(
            "sglang.srt.models.qwen3_5_mtp.get_global_expert_distribution_recorder",
            return_value=recorder,
        ):
            output = model(torch.tensor([1, 2]), torch.tensor([0, 1]), batch)

        expected = (
            torch.nn.functional.linear(
                torch.cat([input_embeds, target_hidden], dim=-1), projection
            )
            + 1
        )
        torch.testing.assert_close(output, expected)

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
