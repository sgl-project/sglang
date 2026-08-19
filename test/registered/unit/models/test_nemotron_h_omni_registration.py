import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.srt.arg_groups.overrides import (
    _FLASHINFER_ALLREDUCE_FUSION_ARCHS,
    _MAMBA_EXTRA_BUFFER_ARCHS,
    _MAMBA_RADIX_CACHE_ARCHS,
    _MODEL_OVERRIDE_FNS,
)
from sglang.srt.configs.model_config import is_multimodal_model
from sglang.srt.configs.nano_nemotron_vl import (
    NemotronH_Omni_Reasoning_V3_Config,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptMixedPrecisionConfig,
)
from sglang.srt.models.nano_nemotron_vl import NemotronH_Omni_Reasoning_V3
from sglang.srt.models.nemotron_h import NemotronHForCausalLM
from sglang.srt.models.radio import RadioModel, _map_hf_radio_weight_name
from sglang.srt.multimodal.processors.nano_nemotron_vl import (
    NanoNemotronVLImageProcessor,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _RecordingWeight:
    def __init__(self):
        self.loads = []

    def weight_loader(self, param, weight, shard_id=None):
        self.loads.append((param, weight, shard_id))


class TestNemotronHOmniRegistration(CustomTestCase):
    def test_config_uses_checkpoint_model_type(self):
        config = NemotronH_Omni_Reasoning_V3_Config(
            vision_config={"args": {"model": "radio"}},
            llm_config={},
            architectures=["NemotronH_Omni_Reasoning_V3"],
        )

        self.assertEqual(config.model_type, "nemotron_h_omni")
        self.assertEqual(config.architectures, ["NemotronH_Omni_Reasoning_V3"])

    def test_config_normalizes_current_nemotron_h_layer_names(self):
        llm_config = {
            "layers_block_type": ["linear_attention", "moe", "full_attention"],
            "num_nextn_predict_layers": 1,
            "mtp_layers_block_type": ["full_attention", "moe"],
        }

        config = NemotronH_Omni_Reasoning_V3_Config(
            vision_config={"args": {"model": "radio"}},
            llm_config=llm_config,
        )

        self.assertEqual(
            config.llm_config.layers_block_type,
            ["mamba", "moe", "attention"],
        )
        self.assertEqual(
            config.llm_config.mtp_layers_block_type,
            ["attention", "moe"],
        )
        self.assertEqual(
            llm_config["layers_block_type"],
            ["linear_attention", "moe", "full_attention"],
        )

    def test_model_and_processor_register_new_architecture(self):
        from sglang.srt.models.registry import ModelRegistry

        model_class, architecture = ModelRegistry.resolve_model_cls(
            "NemotronH_Omni_Reasoning_V3"
        )

        self.assertIs(model_class, NemotronH_Omni_Reasoning_V3)
        self.assertEqual(architecture, "NemotronH_Omni_Reasoning_V3")
        self.assertIn(NemotronH_Omni_Reasoning_V3, NanoNemotronVLImageProcessor.models)

    def test_new_architecture_is_multimodal(self):
        self.assertTrue(is_multimodal_model(["NemotronH_Omni_Reasoning_V3"]))

    def test_new_architecture_uses_nemotron_h_runtime_policy(self):
        architecture = "NemotronH_Omni_Reasoning_V3"

        self.assertIn(architecture, _MODEL_OVERRIDE_FNS)
        self.assertIn(architecture, _MAMBA_RADIX_CACHE_ARCHS)
        self.assertIn(architecture, _MAMBA_EXTRA_BUFFER_ARCHS)
        self.assertIn(architecture, _FLASHINFER_ALLREDUCE_FUSION_ARCHS)

    def test_mixed_precision_resolves_fused_qkv_from_split_layers(self):
        quant_config = ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    f"language_model.model.layers.7.mixer.{projection}": {
                        "quant_algo": "FP8"
                    }
                    for projection in ("q_proj", "k_proj", "v_proj")
                },
                "packed_modules_mapping": (
                    NemotronH_Omni_Reasoning_V3.packed_modules_mapping
                ),
            }
        )

        self.assertEqual(
            NemotronH_Omni_Reasoning_V3.packed_modules_mapping,
            NemotronHForCausalLM.packed_modules_mapping,
        )
        self.assertEqual(
            quant_config._resolve_quant_algo(
                "language_model.model.layers.7.mixer.qkv_proj"
            ),
            "FP8",
        )

    def test_mamba_cache_chunk_size_uses_language_config(self):
        server_args = object.__new__(ServerArgs)
        server_args.get_model_config = lambda: SimpleNamespace(
            hf_config=SimpleNamespace(),
            hf_text_config=SimpleNamespace(mamba_chunk_size=128),
        )

        with patch(
            "sglang.srt.server_args.resolved_view",
            return_value=SimpleNamespace(page_size=64),
        ):
            self.assertEqual(server_args.mamba_cache_chunk_size, 128)

    def test_multimodal_wrapper_exposes_language_embed_and_head(self):
        model = object.__new__(NemotronH_Omni_Reasoning_V3)
        nn.Module.__init__(model)
        embed = object()
        head = object()
        model.language_model = SimpleNamespace(
            get_embed_and_head=lambda: (embed, head),
            lm_head=head,
        )

        self.assertEqual(model.get_embed_and_head(), (embed, head))
        self.assertIs(model.lm_head, head)

    def test_super_vision_final_layernorm_is_loaded_and_applied(self):
        model = object.__new__(NemotronH_Omni_Reasoning_V3)
        nn.Module.__init__(model)
        model.mlp1 = nn.Sequential()
        model.vision_final_layernorm = nn.LayerNorm(2)
        model.language_model = SimpleNamespace(load_weights=lambda weights: None)
        model.vision_model = SimpleNamespace(load_weights=lambda weights: None)
        model.sound_encoder = None

        weight = torch.tensor([2.0, 3.0])
        bias = torch.tensor([0.5, -0.5])
        model.load_weights(
            [
                ("vision_projector.vision_final_layernorm.weight", weight),
                ("vision_projector.vision_final_layernorm.bias", bias),
            ]
        )

        features = torch.tensor([[1.0, 3.0]])
        expected = nn.functional.layer_norm(features, (2,), weight, bias)
        torch.testing.assert_close(model._normalize_vision_features(features), expected)

    def test_super_hf_vision_and_projector_names_are_remapped(self):
        remap = NemotronH_Omni_Reasoning_V3._remap_checkpoint_weight_name

        self.assertEqual(
            remap("vision_model.embeddings.position_embedding"),
            "vision_model.radio_model.hf_model.embeddings.position_embedding",
        )
        self.assertEqual(
            remap("vision_model.embeddings.video_patch_projection.weight"),
            "vision_model.radio_model.hf_model.embeddings.video_patch_projection.weight",
        )
        self.assertEqual(
            remap("vision_projector.mlp1.linear1.weight"),
            "mlp1.1.weight",
        )
        self.assertEqual(
            remap("vision_model.radio_model.model.patch_generator.pos_embed"),
            "vision_model.radio_model.model.patch_generator.pos_embed",
        )

    def test_hf_radio_loader_maps_embeddings_and_split_qkv(self):
        model = object.__new__(RadioModel)
        nn.Module.__init__(model)
        position_embedding = _RecordingWeight()
        qkv_weight = _RecordingWeight()
        model.named_parameters = lambda: iter(
            [
                ("model.patch_generator.pos_embed", position_embedding),
                (
                    "model.encoder.layers.0.attn.attn.qkv_proj.weight",
                    qkv_weight,
                ),
            ]
        )

        position = torch.ones(1)
        query, key, value = (torch.full((1,), value) for value in (2, 3, 4))
        loaded = model.load_weights(
            [
                (
                    "radio_model.hf_model.embeddings.position_embedding",
                    position,
                ),
                (
                    "radio_model.hf_model.encoder.layer.0.attention.attention.query.weight",
                    query,
                ),
                (
                    "radio_model.hf_model.encoder.layer.0.attention.attention.key.weight",
                    key,
                ),
                (
                    "radio_model.hf_model.encoder.layer.0.attention.attention.value.weight",
                    value,
                ),
                ("radio_model.hf_model.summary_idxs", torch.tensor([0, 1])),
            ]
        )

        self.assertEqual(
            loaded,
            {
                "model.patch_generator.pos_embed",
                "model.encoder.layers.0.attn.attn.qkv_proj.weight",
            },
        )
        self.assertEqual(
            position_embedding.loads, [(position_embedding, position, None)]
        )
        self.assertEqual(
            qkv_weight.loads,
            [
                (qkv_weight, query, "q"),
                (qkv_weight, key, "k"),
                (qkv_weight, value, "v"),
            ],
        )

    def test_hf_radio_name_mapping_covers_encoder_parameters(self):
        cases = {
            "embeddings.video_patch_projection.weight": (
                "model.patch_generator.video_embedder.weight",
                None,
            ),
            "encoder.layer.1.attention.output.dense.weight": (
                "model.encoder.layers.1.attn.attn.proj.weight",
                None,
            ),
            "encoder.layer.2.layer_scale1.lambda1": (
                "model.encoder.layers.2.ls1",
                None,
            ),
            "encoder.layer.3.layer_scale2.lambda1": (
                "model.encoder.layers.3.ls2",
                None,
            ),
            "encoder.layer.4.mlp.fc1.bias": (
                "model.encoder.layers.4.mlp.fc1.bias",
                None,
            ),
            "encoder.layer.5.norm2.weight": (
                "model.encoder.layers.5.norm2.weight",
                None,
            ),
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(
                    _map_hf_radio_weight_name(f"radio_model.hf_model.{source}"),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
