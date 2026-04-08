import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from sglang.multimodal_gen.tools.convert_modelopt_fp8_checkpoint import (
    build_fp8_scale_map,
    convert_modelopt_fp8_checkpoint,
    is_ignored_by_modelopt,
)


class TestConvertModelOptFp8Checkpoint(unittest.TestCase):
    def test_is_ignored_by_modelopt_matches_module_prefixes(self):
        self.assertTrue(
            is_ignored_by_modelopt("blocks.0.attn2.to_k.weight", ["blocks.0*"])
        )
        self.assertTrue(is_ignored_by_modelopt("proj_out.weight", ["proj_out"]))
        self.assertFalse(
            is_ignored_by_modelopt("blocks.3.attn2.to_k.weight", ["blocks.0*"])
        )

    def test_build_fp8_scale_map_uses_modelopt_amax(self):
        scale_map = build_fp8_scale_map(
            {
                "foo.weight_quantizer._amax": torch.tensor(2.24, dtype=torch.float32),
                "foo.input_quantizer._amax": torch.tensor(224.0, dtype=torch.float32),
            }
        )

        self.assertIn("foo.weight", scale_map)
        self.assertTrue(torch.allclose(scale_map["foo.weight"]["weight_scale"], torch.tensor([0.005])))
        self.assertTrue(torch.allclose(scale_map["foo.weight"]["input_scale"], torch.tensor([0.5])))

    def test_convert_checkpoint_injects_scales_and_restores_flux2_bf16_fallbacks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "modelopt_hf" / "transformer"
            base_dir = root / "base_model" / "transformer"
            output_dir = root / "converted"
            source_dir.mkdir(parents=True)
            base_dir.mkdir(parents=True)

            config = {
                "_class_name": "Flux2Transformer2DModel",
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
            (source_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (base_dir / "config.json").write_text(json.dumps({"_class_name": "Flux2Transformer2DModel"}), encoding="utf-8")

            source_tensors = {
                "proj.weight": torch.full((2, 2), 1.0, dtype=torch.float16),
                "context_embedder.weight": torch.full((2, 2), 9.0, dtype=torch.float16),
            }
            base_tensors = {
                "proj.weight": torch.full((2, 2), 3.0, dtype=torch.bfloat16),
                "context_embedder.weight": torch.full((2, 2), 7.0, dtype=torch.bfloat16),
            }

            shard_name = "diffusion_pytorch_model-00001-of-00001.safetensors"
            save_file(source_tensors, source_dir / shard_name, metadata={"format": "pt"})
            save_file(base_tensors, base_dir / shard_name, metadata={"format": "pt"})

            index_data = {
                "metadata": {"total_size": 0},
                "weight_map": {
                    "proj.weight": shard_name,
                    "context_embedder.weight": shard_name,
                },
            }
            (source_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
                json.dumps(index_data),
                encoding="utf-8",
            )
            (base_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
                json.dumps(index_data),
                encoding="utf-8",
            )

            backbone_ckpt = root / "backbone.pt"
            torch.save(
                {
                    "model_state_dict": {
                        "proj.weight_quantizer._amax": torch.tensor(4.48, dtype=torch.float32),
                        "proj.input_quantizer._amax": torch.tensor(224.0, dtype=torch.float32),
                        "context_embedder.weight_quantizer._amax": torch.tensor(8.96, dtype=torch.float32),
                        "context_embedder.input_quantizer._amax": torch.tensor(112.0, dtype=torch.float32),
                    }
                },
                backbone_ckpt,
            )

            stats = convert_modelopt_fp8_checkpoint(
                modelopt_hf_dir=str(source_dir.parent),
                modelopt_backbone_ckpt=str(backbone_ckpt),
                base_transformer_dir=str(base_dir.parent),
                output_dir=str(output_dir),
            )

            self.assertEqual(stats["bf16_fallback_weights"], 1)
            self.assertEqual(stats["added_scale_tensors"], 2)

            output_shard = output_dir / shard_name
            converted = load_file(output_shard)
            expected_proj = (source_tensors["proj.weight"].float() / 0.01).to(
                torch.float8_e4m3fn
            )
            self.assertEqual(converted["proj.weight"].dtype, torch.float8_e4m3fn)
            self.assertTrue(torch.equal(converted["proj.weight"], expected_proj))
            self.assertTrue(torch.equal(converted["context_embedder.weight"], base_tensors["context_embedder.weight"]))
            self.assertTrue(torch.allclose(converted["proj.weight_scale"], torch.tensor([0.01], dtype=torch.float32)))
            self.assertTrue(torch.allclose(converted["proj.input_scale"], torch.tensor([0.5], dtype=torch.float32)))
            self.assertNotIn("context_embedder.weight_scale", converted)
            self.assertNotIn("context_embedder.input_scale", converted)

            with safe_open(output_shard, framework="pt", device="cpu") as f:
                metadata = f.metadata()
            self.assertIn("quantization_config", metadata)
            self.assertIn("_quantization_metadata", metadata)

            with open(output_dir / "diffusion_pytorch_model.safetensors.index.json", encoding="utf-8") as f:
                output_index = json.load(f)
            self.assertIn("proj.weight_scale", output_index["weight_map"])
            self.assertIn("proj.input_scale", output_index["weight_map"])
            self.assertNotIn("context_embedder.weight_scale", output_index["weight_map"])

    def test_convert_checkpoint_preserves_modelopt_ignored_bf16_layers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "modelopt_hf" / "transformer"
            output_dir = root / "converted"
            source_dir.mkdir(parents=True)

            config = {
                "_class_name": "WanTransformer3DModel",
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "FP8",
                    "ignore": ["blocks.0*", "proj_out"],
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
            (source_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

            source_tensors = {
                "blocks.0.attn2.to_k.weight": torch.full((2, 2), 3.0, dtype=torch.bfloat16),
                "blocks.3.attn2.to_k.weight": torch.full((2, 2), 2.0, dtype=torch.bfloat16),
                "proj_out.weight": torch.full((2, 2), -1.0, dtype=torch.bfloat16),
            }

            shard_name = "diffusion_pytorch_model-00001-of-00001.safetensors"
            save_file(source_tensors, source_dir / shard_name, metadata={"format": "pt"})
            index_data = {
                "metadata": {"total_size": 0},
                "weight_map": {name: shard_name for name in source_tensors},
            }
            (source_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
                json.dumps(index_data),
                encoding="utf-8",
            )

            backbone_ckpt = root / "backbone.pt"
            torch.save(
                {
                    "model_state_dict": {
                        "blocks.0.attn2.to_k.weight_quantizer._amax": torch.tensor(4.48, dtype=torch.float32),
                        "blocks.0.attn2.to_k.input_quantizer._amax": torch.tensor(224.0, dtype=torch.float32),
                        "blocks.3.attn2.to_k.weight_quantizer._amax": torch.tensor(8.96, dtype=torch.float32),
                        "blocks.3.attn2.to_k.input_quantizer._amax": torch.tensor(112.0, dtype=torch.float32),
                        "proj_out.weight_quantizer._amax": torch.tensor(13.44, dtype=torch.float32),
                        "proj_out.input_quantizer._amax": torch.tensor(336.0, dtype=torch.float32),
                    }
                },
                backbone_ckpt,
            )

            stats = convert_modelopt_fp8_checkpoint(
                modelopt_hf_dir=str(source_dir.parent),
                modelopt_backbone_ckpt=str(backbone_ckpt),
                output_dir=str(output_dir),
            )

            self.assertEqual(stats["preserved_ignored_weights"], 2)
            self.assertEqual(stats["quantized_weights"], 1)
            self.assertEqual(stats["added_scale_tensors"], 2)

            converted = load_file(output_dir / shard_name)
            self.assertEqual(
                converted["blocks.0.attn2.to_k.weight"].dtype, torch.bfloat16
            )
            self.assertTrue(
                torch.equal(
                    converted["blocks.0.attn2.to_k.weight"],
                    source_tensors["blocks.0.attn2.to_k.weight"],
                )
            )
            self.assertEqual(converted["proj_out.weight"].dtype, torch.bfloat16)
            self.assertNotIn("blocks.0.attn2.to_k.weight_scale", converted)
            self.assertNotIn("blocks.0.attn2.to_k.input_scale", converted)
            self.assertNotIn("proj_out.weight_scale", converted)
            self.assertNotIn("proj_out.input_scale", converted)
            self.assertEqual(
                converted["blocks.3.attn2.to_k.weight"].dtype, torch.float8_e4m3fn
            )


if __name__ == "__main__":
    unittest.main()
