import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from sglang.multimodal_gen.tools.convert_modelopt_fp8_checkpoint import (
    build_fp8_scale_map,
    convert_modelopt_fp8_checkpoint,
    is_ignored_by_modelopt,
)


def _fp8_quant_config(*, class_name: str, ignore: list[str] | None = None) -> dict:
    return {
        "_class_name": class_name,
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "FP8",
            "ignore": ignore or [],
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


def _write_config(directory: Path, config: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _write_shard(directory: Path, tensors: dict[str, torch.Tensor], shard_name: str) -> None:
    save_file(tensors, directory / shard_name, metadata={"format": "pt"})
    index_data = {
        "metadata": {"total_size": 0},
        "weight_map": {name: shard_name for name in tensors},
    }
    (directory / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(index_data),
        encoding="utf-8",
    )


def test_is_ignored_by_modelopt_matches_module_prefixes():
    assert is_ignored_by_modelopt("blocks.0.attn2.to_k.weight", ["blocks.0*"])
    assert is_ignored_by_modelopt("proj_out.weight", ["proj_out"])
    assert not is_ignored_by_modelopt("blocks.3.attn2.to_k.weight", ["blocks.0*"])


def test_build_fp8_scale_map_uses_modelopt_amax():
    scale_map = build_fp8_scale_map(
        {
            "foo.weight_quantizer._amax": torch.tensor(2.24, dtype=torch.float32),
            "foo.input_quantizer._amax": torch.tensor(224.0, dtype=torch.float32),
        }
    )

    assert "foo.weight" in scale_map
    assert torch.allclose(
        scale_map["foo.weight"]["weight_scale"],
        torch.tensor([0.005], dtype=torch.float32),
    )
    assert torch.allclose(
        scale_map["foo.weight"]["input_scale"],
        torch.tensor([0.5], dtype=torch.float32),
    )


def test_convert_checkpoint_injects_scales_and_restores_flux2_fallbacks(tmp_path):
    modelopt_dir = tmp_path / "modelopt_hf" / "transformer"
    base_dir = tmp_path / "base_model" / "transformer"
    output_dir = tmp_path / "converted"
    shard_name = "diffusion_pytorch_model-00001-of-00001.safetensors"

    _write_config(
        modelopt_dir,
        _fp8_quant_config(class_name="Flux2Transformer2DModel"),
    )
    _write_config(base_dir, {"_class_name": "Flux2Transformer2DModel"})

    source_tensors = {
        "proj.weight": torch.full((2, 2), 1.0, dtype=torch.float16),
        "context_embedder.weight": torch.full((2, 2), 9.0, dtype=torch.float16),
    }
    base_tensors = {
        "proj.weight": torch.full((2, 2), 3.0, dtype=torch.bfloat16),
        "context_embedder.weight": torch.full((2, 2), 7.0, dtype=torch.bfloat16),
    }
    _write_shard(modelopt_dir, source_tensors, shard_name)
    _write_shard(base_dir, base_tensors, shard_name)

    backbone_ckpt = tmp_path / "backbone.pt"
    torch.save(
        {
            "model_state_dict": {
                "proj.weight_quantizer._amax": torch.tensor(4.48, dtype=torch.float32),
                "proj.input_quantizer._amax": torch.tensor(
                    224.0, dtype=torch.float32
                ),
                "context_embedder.weight_quantizer._amax": torch.tensor(
                    8.96, dtype=torch.float32
                ),
                "context_embedder.input_quantizer._amax": torch.tensor(
                    112.0, dtype=torch.float32
                ),
            }
        },
        backbone_ckpt,
    )

    stats = convert_modelopt_fp8_checkpoint(
        modelopt_hf_dir=str(modelopt_dir.parent),
        modelopt_backbone_ckpt=str(backbone_ckpt),
        base_transformer_dir=str(base_dir.parent),
        output_dir=str(output_dir),
    )

    assert stats["bf16_fallback_weights"] == 1
    assert stats["added_scale_tensors"] == 2

    converted = load_file(output_dir / shard_name)
    expected_proj = (source_tensors["proj.weight"].float() / 0.01).to(
        torch.float8_e4m3fn
    )

    assert converted["proj.weight"].dtype == torch.float8_e4m3fn
    assert torch.equal(converted["proj.weight"], expected_proj)
    assert torch.equal(
        converted["context_embedder.weight"],
        base_tensors["context_embedder.weight"],
    )
    assert torch.allclose(
        converted["proj.weight_scale"],
        torch.tensor([0.01], dtype=torch.float32),
    )
    assert torch.allclose(
        converted["proj.input_scale"],
        torch.tensor([0.5], dtype=torch.float32),
    )
    assert "context_embedder.weight_scale" not in converted
    assert "context_embedder.input_scale" not in converted

    with safe_open(output_dir / shard_name, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    assert "quantization_config" in metadata
    assert "_quantization_metadata" in metadata

    with open(
        output_dir / "diffusion_pytorch_model.safetensors.index.json",
        encoding="utf-8",
    ) as handle:
        output_index = json.load(handle)
    assert "proj.weight_scale" in output_index["weight_map"]
    assert "proj.input_scale" in output_index["weight_map"]
    assert "context_embedder.weight_scale" not in output_index["weight_map"]


def test_convert_checkpoint_preserves_modelopt_ignored_layers(tmp_path):
    modelopt_dir = tmp_path / "modelopt_hf" / "transformer"
    output_dir = tmp_path / "converted"
    shard_name = "diffusion_pytorch_model-00001-of-00001.safetensors"

    _write_config(
        modelopt_dir,
        _fp8_quant_config(
            class_name="WanTransformer3DModel",
            ignore=["blocks.0*", "proj_out"],
        ),
    )

    source_tensors = {
        "blocks.0.attn2.to_k.weight": torch.full((2, 2), 3.0, dtype=torch.bfloat16),
        "blocks.3.attn2.to_k.weight": torch.full((2, 2), 2.0, dtype=torch.bfloat16),
        "proj_out.weight": torch.full((2, 2), -1.0, dtype=torch.bfloat16),
    }
    _write_shard(modelopt_dir, source_tensors, shard_name)

    backbone_ckpt = tmp_path / "backbone.pt"
    torch.save(
        {
            "model_state_dict": {
                "blocks.0.attn2.to_k.weight_quantizer._amax": torch.tensor(
                    4.48, dtype=torch.float32
                ),
                "blocks.0.attn2.to_k.input_quantizer._amax": torch.tensor(
                    224.0, dtype=torch.float32
                ),
                "blocks.3.attn2.to_k.weight_quantizer._amax": torch.tensor(
                    8.96, dtype=torch.float32
                ),
                "blocks.3.attn2.to_k.input_quantizer._amax": torch.tensor(
                    112.0, dtype=torch.float32
                ),
                "proj_out.weight_quantizer._amax": torch.tensor(
                    13.44, dtype=torch.float32
                ),
                "proj_out.input_quantizer._amax": torch.tensor(
                    336.0, dtype=torch.float32
                ),
            }
        },
        backbone_ckpt,
    )

    stats = convert_modelopt_fp8_checkpoint(
        modelopt_hf_dir=str(modelopt_dir.parent),
        modelopt_backbone_ckpt=str(backbone_ckpt),
        output_dir=str(output_dir),
    )

    assert stats["preserved_ignored_weights"] == 2
    assert stats["quantized_weights"] == 1
    assert stats["added_scale_tensors"] == 2

    converted = load_file(output_dir / shard_name)
    assert converted["blocks.0.attn2.to_k.weight"].dtype == torch.bfloat16
    assert torch.equal(
        converted["blocks.0.attn2.to_k.weight"],
        source_tensors["blocks.0.attn2.to_k.weight"],
    )
    assert converted["proj_out.weight"].dtype == torch.bfloat16
    assert "blocks.0.attn2.to_k.weight_scale" not in converted
    assert "blocks.0.attn2.to_k.input_scale" not in converted
    assert "proj_out.weight_scale" not in converted
    assert "proj_out.input_scale" not in converted
    assert converted["blocks.3.attn2.to_k.weight"].dtype == torch.float8_e4m3fn
