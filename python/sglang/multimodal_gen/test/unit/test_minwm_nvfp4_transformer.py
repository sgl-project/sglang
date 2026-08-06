import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from sglang.multimodal_gen.tools.build_minwm_nvfp4_transformer import (
    INDEX_FILENAME,
    materialize_minwm_nvfp4_transformer,
    minwm_block_linear_names,
    prepare_calibration_kwargs,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_materialize_minwm_nvfp4_transformer(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    shard_name = "diffusion_pytorch_model-00001-of-00001.safetensors"
    modules = minwm_block_linear_names()
    tensors = {}
    weight_map = {}
    packed_state = {}
    for module_name in modules:
        weight_name = f"{module_name}.weight"
        tensors[weight_name] = torch.ones((2, 16), dtype=torch.bfloat16)
        weight_map[weight_name] = shard_name
        packed_state[weight_name] = torch.zeros((2, 8), dtype=torch.uint8)
        packed_state[f"{module_name}.weight_scale"] = torch.ones(
            (2, 1), dtype=torch.float8_e4m3fn
        )
        packed_state[f"{module_name}.weight_scale_2"] = torch.tensor(
            1.0, dtype=torch.float32
        )
        packed_state[f"{module_name}.input_scale"] = torch.tensor(
            1.0, dtype=torch.float32
        )
    tensors["action_in.proj.weight"] = torch.ones((2, 16), dtype=torch.bfloat16)
    weight_map["action_in.proj.weight"] = shard_name
    save_file(tensors, source / shard_name, metadata={"format": "pt"})
    _write_json(
        source / "config.json", {"_class_name": "MinWMCausalTransformer3DModel"}
    )
    _write_json(
        source / INDEX_FILENAME,
        {"metadata": {"total_size": 1}, "weight_map": weight_map},
    )

    manifest = materialize_minwm_nvfp4_transformer(
        input_dir=str(source),
        packed_state=packed_state,
        output_dir=str(output),
    )

    converted = load_file(output / shard_name, device="cpu")
    assert manifest["packed_weights"] == 300
    assert converted["blocks.0.self_attn.q.weight"].dtype == torch.uint8
    assert converted["blocks.0.self_attn.q.weight"].shape == (2, 8)
    assert converted["blocks.0.self_attn.q.weight_scale"].dtype == torch.float8_e4m3fn
    assert converted["action_in.proj.weight"].dtype == torch.bfloat16
    output_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert output_config["quantization_config"]["quant_algo"] == "NVFP4"
    assert output_config["quantization_config"]["group_size"] == 16


def test_prepare_calibration_kwargs_rebuilds_cold_inference_cache():
    cache = object()

    class Generator:
        def make_kv_cache(self):
            return cache

    record = {
        "x": torch.ones(1),
        "t": torch.zeros(1),
        "context": torch.ones(1),
        "context_lens": torch.ones(1, dtype=torch.int32),
        "output": torch.zeros(1),
    }

    kwargs = prepare_calibration_kwargs(Generator(), record, device="cpu")

    assert kwargs["cache"] is cache
    assert kwargs["self_cache_update"] is None
    assert "output" not in kwargs
