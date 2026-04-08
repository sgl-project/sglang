import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp8Config,
)
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    _build_transformer_quant_adapters,
    _resolve_quant_config,
)
from sglang.multimodal_gen.runtime.models.dits.flux_2 import Flux2Attention
from sglang.multimodal_gen.runtime.utils.quantization_utils import (
    get_quant_config,
    get_quant_config_from_safetensors_metadata,
)


class TestDiffusionModelOptQuant(unittest.TestCase):
    def test_get_quant_config_resolves_modelopt_fp8_from_flat_config(self):
        model_config = {
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "FP8",
                "ignore": ["proj_out"],
                "config_groups": {
                    "group_0": {
                        "input_activations": {
                            "dynamic": False,
                            "num_bits": 8,
                            "type": "float",
                        },
                        "weights": {
                            "dynamic": False,
                            "num_bits": 8,
                            "type": "float",
                        },
                    }
                },
            }
        }

        quant_config = get_quant_config(model_config, component_model_path="/tmp")

        self.assertIsInstance(quant_config, ModelOptFp8Config)
        self.assertEqual(quant_config.get_name(), "modelopt_fp8")

    def test_get_quant_config_from_safetensors_metadata_reads_quantization_config(self):
        metadata = {
            "quantization_config": json.dumps(
                {
                    "quant_method": "modelopt",
                    "quant_algo": "FP8",
                    "ignore": [],
                    "config_groups": {
                        "group_0": {
                            "input_activations": {
                                "dynamic": False,
                                "num_bits": 8,
                                "type": "float",
                            },
                            "weights": {
                                "dynamic": False,
                                "num_bits": 8,
                                "type": "float",
                            },
                        }
                    },
                }
            )
        }

        with patch(
            "sglang.multimodal_gen.runtime.utils.quantization_utils.get_metadata_from_safetensors_file",
            return_value=metadata,
        ):
            quant_config = get_quant_config_from_safetensors_metadata("/tmp/fake.safetensors")

        self.assertIsInstance(quant_config, ModelOptFp8Config)

    def test_resolve_quant_config_reads_modelopt_transformer_override_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "_class_name": "FluxTransformer2DModel",
                        "quantization_config": {
                            "quant_method": "modelopt",
                            "quant_algo": "FP8",
                            "ignore": [],
                            "config_groups": {
                                "group_0": {
                                    "input_activations": {
                                        "dynamic": False,
                                        "num_bits": 8,
                                        "type": "float",
                                    },
                                    "weights": {
                                        "dynamic": False,
                                        "num_bits": 8,
                                        "type": "float",
                                    },
                                }
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            server_args = SimpleNamespace(transformer_weights_path=tmpdir)
            quant_config = _resolve_quant_config(
                hf_config={},
                server_args=server_args,
                safetensors_list=[],
                component_model_path="/tmp/base-transformer",
            )

        self.assertIsInstance(quant_config, ModelOptFp8Config)

    def test_flux2_attention_uses_fused_qkv_only_for_packed_nvfp4(self):
        class _DummyAttention(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

        packed_quant_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=[],
            checkpoint_uses_packed_qkv=True,
        )
        unpacked_quant_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=[],
            checkpoint_uses_packed_qkv=False,
        )

        with patch(
            "sglang.multimodal_gen.runtime.models.dits.flux_2.get_tp_world_size",
            return_value=1,
        ), patch(
            "sglang.multimodal_gen.runtime.layers.linear.get_tp_group",
            return_value=object(),
        ), patch(
            "sglang.multimodal_gen.runtime.layers.linear.get_group_size",
            return_value=1,
        ), patch(
            "sglang.multimodal_gen.runtime.layers.linear.get_group_rank",
            return_value=0,
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.flux_2.USPAttention",
            _DummyAttention,
        ):
            packed_attn = Flux2Attention(
                query_dim=128,
                num_heads=2,
                dim_head=64,
                added_kv_proj_dim=128,
                quant_config=packed_quant_config,
            )
            unpacked_attn = Flux2Attention(
                query_dim=128,
                num_heads=2,
                dim_head=64,
                added_kv_proj_dim=128,
                quant_config=unpacked_quant_config,
            )

        self.assertTrue(packed_attn.use_fused_qkv)
        self.assertTrue(packed_attn.use_fused_added_qkv)
        self.assertFalse(unpacked_attn.use_fused_qkv)
        self.assertFalse(unpacked_attn.use_fused_added_qkv)

    def test_modelopt_fp8_prepare_disables_incompatible_dit_offload_modes(self):
        server_args = SimpleNamespace(
            dit_cpu_offload=True,
            dit_layerwise_offload=True,
            text_encoder_cpu_offload=True,
            tp_size=1,
            transformer_weights_path="/tmp/modelopt-fp8",
        )
        quant_config = ModelOptFp8Config(
            is_checkpoint_fp8_serialized=True,
            exclude_modules=[],
        )

        adapters = _build_transformer_quant_adapters(
            cls_name="Flux2Transformer2DModel",
            server_args=server_args,
            quant_config=quant_config,
            nunchaku_config=None,
            model_cls=torch.nn.Module,
            safetensors_list=[],
        )
        for adapter in adapters:
            adapter.prepare()

        self.assertFalse(server_args.dit_cpu_offload)
        self.assertFalse(server_args.dit_layerwise_offload)
        self.assertTrue(server_args.text_encoder_cpu_offload)


if __name__ == "__main__":
    unittest.main()
