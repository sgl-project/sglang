import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from transformers import PretrainedConfig

from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.modelopt_quant import ModelOptMixedPrecisionConfig
from sglang.srt.layers.quantization.nvfp4_online import NvFp4OnlineConfig
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.model_loader.weight_utils import get_quant_config
from sglang.srt.models import mistral_large_4
from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM
from sglang.srt.models.mistral_large_4 import (
    MistralLarge4ForCausalLM,
    mistral_large_4_modelopt_layers,
    mistral_large_4_weight_name,
)
from sglang.srt.utils.hf_transformers.mistral_utils import _remap_mistral_large_4_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_NUM_LAYERS = 72
_KV_SOURCES = list(range(_NUM_LAYERS))
for _consumer in range(39, _NUM_LAYERS, 4):
    _KV_SOURCES[_consumer] = 35

# One tensor per family of the consolidated checkpoint index. Layer 0 is dense
# with its own indexer, 3 is a NoPE KV provider, 39 reads layer 35's latent cache.
# fmt: off
_CHECKPOINT_TO_PARAM = {
    "tok_embeddings.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
    "lerope.alpha": "model.lerope_alpha",
    "layers.0.attention_norm.weight": "model.layers.0.input_layernorm.weight",
    "layers.0.ffn_norm.weight": "model.layers.0.post_attention_layernorm.weight",
    "layers.0.attention.wq.weight": "model.layers.0.self_attn.q_proj.weight",
    "layers.0.attention.wq.qscale_weight": "model.layers.0.self_attn.q_proj.weight_scale_inv",
    "layers.0.attention.wo.weight": "model.layers.0.self_attn.o_proj.weight",
    "layers.0.attention.wo.qscale_weight": "model.layers.0.self_attn.o_proj.weight_scale_inv",
    "layers.0.attention.wkv_a_with_mqa.weight": "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
    "layers.0.attention.wkv_b.weight": "model.layers.0.self_attn.kv_b_proj.weight",
    "layers.0.attention.ssmax_scaling.s": "model.layers.0.self_attn.ssmax_scaling_s",
    "layers.0.attention.ssmax_scaling.b": "model.layers.0.self_attn.ssmax_scaling_b",
    "layers.0.attention.indexer.wq_b.weight": "model.layers.0.self_attn.indexer.wq_b.weight",
    "layers.0.attention.indexer.wq_b.qscale_weight": "model.layers.0.self_attn.indexer.wq_b.weight_scale_inv",
    "layers.0.attention.indexer.wk.weight": "model.layers.0.self_attn.indexer.wk.weight",
    "layers.0.attention.indexer.weights_proj.weight": "model.layers.0.self_attn.indexer.weights_proj.weight",
    "layers.0.attention.indexer.k_norm.bias": "model.layers.0.self_attn.indexer.k_norm.bias",
    "layers.0.attention.indexer.q_input_norm.weight": "model.layers.0.self_attn.indexer.q_input_norm.weight",
    "layers.0.attention.indexer.ssmax_scaling.s": "model.layers.0.self_attn.indexer.ssmax_scaling_s",
    "layers.0.feed_forward.w1.qscale_weight": "model.layers.0.mlp.gate_proj.weight_scale_inv",
    "layers.0.feed_forward.w3.weight": "model.layers.0.mlp.up_proj.weight",
    "layers.1.gate.weight": "model.layers.1.mlp.gate.weight",
    "layers.1.experts.127.w2.qscale_weight": "model.layers.1.mlp.experts.127.down_proj.weight_scale_inv",
    "layers.1.shared_experts.w1.weight": "model.layers.1.mlp.shared_experts.gate_proj.weight",
    "layers.3.attention.wkv_a_with_mqa.weight": "model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
    "layers.3.attention.kv_a_norm.weight": "model.layers.3.self_attn.kv_a_layernorm.weight",
    "layers.35.attention.wkv_a_with_mqa.weight": "model.layers.35.self_attn.kv_a_proj_with_mqa.weight",
    "layers.39.attention.wkv_a_with_mqa.weight": None,
    "layers.39.attention.kv_a_norm.weight": None,
    "layers.39.attention.wkv_b.weight": "model.layers.39.self_attn.kv_b_proj.weight",
    "layers.0.attention.k_fake_quantizer.qscale_act": None,
    "mtp_dspark.backbone.layers.0.attention.wq.weight": None,
    "mtp_dspark.autoregressive_head.embedding.weight": None,
    "vision_encoder.ln_pre.weight": None,
    "vision_language_adapter.w_in.weight": None,
}
# fmt: on


def _map(name: str):
    return mistral_large_4_weight_name(name, kv_sources=_KV_SOURCES, block_quant=True)


def _checkpoint_quant_config() -> dict:
    return _remap_mistral_large_4_args(
        {
            "num_hidden_layers": _NUM_LAYERS,
            "quantization_config": {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
            },
        }
    )["quantization_config"]


# Layers that own a DSA indexer in the official NVFP4 checkpoint.
_INDEXER_LAYERS = [*range(17), *(n for n in range(19, _NUM_LAYERS) if n % 4 in (0, 3))]


def _official_nvfp4_quantization_config() -> dict:
    """The params.json quantization_config of mistralai/Mistral-Large-4-1T-A52B-NVFP4
    at 4359ec60: routed experts NVFP4, every other quantized linear block FP8."""
    block_fp8 = {"quant_algo": "FP8_PB_WO"}
    layers = {f"layers.0.feed_forward.w{i}": block_fp8 for i in (1, 2, 3)}
    for layer in range(_NUM_LAYERS):
        layers[f"layers.{layer}.attention.wq"] = block_fp8
        layers[f"layers.{layer}.attention.wo"] = block_fp8
        if layer in _INDEXER_LAYERS:
            layers[f"layers.{layer}.attention.indexer.wq_b"] = block_fp8
        if layer > 0:
            layers[f"layers.{layer}.experts"] = {
                "quant_algo": "NVFP4",
                "group_size": 16,
            }
            for i in (1, 2, 3):
                layers[f"layers.{layer}.shared_experts.w{i}"] = block_fp8
    return {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "group_size": 16,
        "quantized_layers": layers,
        "ignore": [],
    }


class _Built(Exception):
    pass


def _official_nvfp4_quant_config() -> ModelOptMixedPrecisionConfig:
    """The quant config as the model's layers see it when they are built."""
    quant_config = ModelOptMixedPrecisionConfig.from_config(
        _official_nvfp4_quantization_config()
    )

    def build(self, config, quant_config, prefix=""):
        raise _Built

    with (
        mock.patch.object(DeepseekV3ForCausalLM, "__init__", build),
        mock.patch.object(mistral_large_4, "_check_context_parallel_layout"),
    ):
        try:
            MistralLarge4ForCausalLM(
                SimpleNamespace(index_kv_dtype="fp8"), quant_config
            )
        except _Built:
            pass
    return quant_config


class TestMistralLarge4Weights(CustomTestCase):
    def test_checkpoint_tensors_map_to_params_or_explicit_skips(self):
        self.assertEqual(
            {name: _map(name) for name in _CHECKPOINT_TO_PARAM}, _CHECKPOINT_TO_PARAM
        )
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            _map("layers.0.attention.sinks")

    def test_scale_names_follow_block_fp8_checkpoint_not_config_class(self):
        names = [n for n in _CHECKPOINT_TO_PARAM if "qscale" in n]
        expected = [_CHECKPOINT_TO_PARAM[n] for n in names if _CHECKPOINT_TO_PARAM[n]]
        for config_cls in (Fp8Config, NvFp4OnlineConfig):
            model = SimpleNamespace(
                config=SimpleNamespace(num_hidden_layers=_NUM_LAYERS),
                quant_config=config_cls.from_config(_checkpoint_quant_config()),
            )
            remapped = MistralLarge4ForCausalLM._remap_weights(
                model, [(n, None) for n in names], set()
            )
            with self.subTest(config=config_cls.__name__):
                self.assertEqual([target for target, _ in remapped], expected)

    def test_only_mla_kv_projections_skip_fp8(self):
        ignored_layers = Fp8Config.from_config(
            _checkpoint_quant_config()
        ).ignored_layers

        for param in filter(None, _CHECKPOINT_TO_PARAM.values()):
            module = param.rsplit(".", 1)[0]
            if not module.startswith("model.layers."):
                continue
            is_kv_projection = module.endswith(("kv_a_proj_with_mqa", "kv_b_proj"))
            with self.subTest(module=module):
                self.assertEqual(
                    is_layer_skipped(module, ignored_layers), is_kv_projection
                )


class TestMistralLarge4OfficialNvfp4(CustomTestCase):
    def test_fixture_matches_the_checkpoint_layer_count(self):
        self.assertEqual(
            len(_official_nvfp4_quantization_config()["quantized_layers"]), 475
        )

    def test_modelopt_layers_resolve_on_model_modules(self):
        quant_config = _official_nvfp4_quant_config()
        expected = {
            "model.layers.1.mlp.experts": "NVFP4",
            "model.layers.71.mlp.experts": "NVFP4",
            "model.layers.0.self_attn.q_proj": "FP8_PB_WO",
            "model.layers.39.self_attn.o_proj": "FP8_PB_WO",
            "model.layers.0.self_attn.indexer.wq_b": "FP8_PB_WO",
            "model.layers.0.mlp.gate_up_proj": "FP8_PB_WO",
            "model.layers.0.mlp.gate_proj": "FP8_PB_WO",
            "model.layers.0.mlp.down_proj": "FP8_PB_WO",
            "model.layers.1.mlp.shared_experts.gate_up_proj": "FP8_PB_WO",
            "model.layers.1.mlp.shared_experts.down_proj": "FP8_PB_WO",
            "model.layers.0.self_attn.kv_a_proj_with_mqa": None,
            "model.layers.0.self_attn.kv_b_proj": None,
            "model.layers.0.self_attn.indexer.wk": None,
            "model.layers.0.self_attn.indexer.weights_proj": None,
            "model.layers.1.mlp.gate": None,
            "lm_head": None,
        }
        self.assertEqual(
            {module: quant_config.resolve_quant_algo(module) for module in expected},
            expected,
        )
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            mistral_large_4_modelopt_layers({"layers.0.attention.sinks": {}})

    def test_fused_shards_must_share_a_scheme(self):
        with self.assertRaisesRegex(ValueError, "different quantization"):
            mistral_large_4_modelopt_layers(
                {
                    "layers.0.feed_forward.w1": {"quant_algo": "FP8_PB_WO"},
                    "layers.0.feed_forward.w3": {"quant_algo": "NVFP4"},
                }
            )

    def test_block_fp8_scales_load_as_2d_scale_inv_and_nvfp4_scales_pass(self):
        model = SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=_NUM_LAYERS),
            quant_config=_official_nvfp4_quant_config(),
        )
        # Shapes from the checkpoint's safetensors headers.
        weights = {
            "layers.0.attention.wq.weight_scale": (
                torch.rand(64, 1, 48, 1),
                "model.layers.0.self_attn.q_proj.weight_scale_inv",
                (64, 48),
            ),
            "layers.0.feed_forward.w1.weight_scale": (
                torch.rand(192, 1, 48, 1),
                "model.layers.0.mlp.gate_proj.weight_scale_inv",
                (192, 48),
            ),
            "layers.5.attention.indexer.wq_b.weight_scale": (
                torch.rand(32, 1, 48, 1),
                "model.layers.5.self_attn.indexer.wq_b.weight_scale_inv",
                (32, 48),
            ),
            "layers.5.shared_experts.w1.weight_scale": (
                torch.rand(48, 1, 48, 1),
                "model.layers.5.mlp.shared_experts.gate_proj.weight_scale_inv",
                (48, 48),
            ),
            "layers.5.experts.0.w1.weight_scale": (
                torch.zeros(6144, 384, dtype=torch.float8_e4m3fn),
                "model.layers.5.mlp.experts.0.gate_proj.weight_scale",
                (6144, 384),
            ),
            "layers.5.experts.0.w1.weight_scale_2": (
                torch.tensor(1.0),
                "model.layers.5.mlp.experts.0.gate_proj.weight_scale_2",
                (),
            ),
            "layers.5.experts.0.w1.input_scale": (
                torch.tensor(1.0),
                "model.layers.5.mlp.experts.0.gate_proj.input_scale",
                (),
            ),
        }
        remapped = MistralLarge4ForCausalLM._remap_weights(
            model, [(name, w[0]) for name, w in weights.items()], set()
        )
        self.assertEqual(
            [(target, tuple(tensor.shape)) for target, tensor in remapped],
            [(target, shape) for _, target, shape in weights.values()],
        )

    def test_flat_params_json_quant_config_needs_no_hf_quant_config_file(self):
        quant = _remap_mistral_large_4_args(
            {
                "num_hidden_layers": _NUM_LAYERS,
                "quantization_config": _official_nvfp4_quantization_config(),
            }
        )["quantization_config"]
        with tempfile.TemporaryDirectory() as checkpoint:
            Path(checkpoint, "params.json").write_text(json.dumps({}))
            model_config = SimpleNamespace(
                quantization="modelopt_mixed",
                hf_config=PretrainedConfig(quantization_config=quant),
                model_path=checkpoint,
                revision=None,
                is_draft_model=False,
                speculative_draft_model_quantization=None,
            )
            quant_config = get_quant_config(
                model_config,
                SimpleNamespace(download_dir=None, model_loader_extra_config=None),
                {},
            )
        self.assertIsInstance(quant_config, ModelOptMixedPrecisionConfig)
        self.assertEqual(len(quant_config.quantized_layers), 475)


if __name__ == "__main__":
    unittest.main()
