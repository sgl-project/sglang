import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from sglang.multimodal_gen.tools.build_minwm_static_fp8_transformer import (
    INDEX_FILENAME,
    build_minwm_static_fp8_transformer,
)


def _module_names() -> list[str]:
    suffixes = [
        "self_attn.q",
        "self_attn.k",
        "self_attn.v",
        "self_attn.o",
        "cross_attn.q",
        "cross_attn.k",
        "cross_attn.v",
        "cross_attn.o",
        "ffn.0",
        "ffn.2",
    ]
    return [f"blocks.{block}.{suffix}" for block in range(30) for suffix in suffixes]


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_build_minwm_static_fp8_transformer(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    shard_name = "diffusion_pytorch_model-00001-of-00001.safetensors"
    tensors = {}
    weight_map = {}
    modules = _module_names()
    for index, module_name in enumerate(modules, start=1):
        weight_name = f"{module_name}.weight"
        bias_name = f"{module_name}.bias"
        tensors[weight_name] = torch.tensor(
            [[index / 300, -index / 600], [index / 900, -index / 1200]],
            dtype=torch.bfloat16,
        )
        tensors[bias_name] = torch.zeros(2, dtype=torch.bfloat16)
        weight_map[weight_name] = shard_name
        weight_map[bias_name] = shard_name
    tensors["action_in.proj.weight"] = torch.ones((2, 2), dtype=torch.bfloat16)
    weight_map["action_in.proj.weight"] = shard_name
    save_file(tensors, source / shard_name, metadata={"format": "pt"})
    _write_json(
        source / "config.json", {"_class_name": "MinWMCausalTransformer3DModel"}
    )
    _write_json(
        source / INDEX_FILENAME,
        {"metadata": {"total_size": 1}, "weight_map": weight_map},
    )
    calibration = tmp_path / "calibration.json"
    _write_json(
        calibration,
        {
            "format": "minwm-static-fp8-calibration-v1",
            "module_count": 300,
            "modules": {
                name: {"input_amax": float(index + 1), "samples": 4}
                for index, name in enumerate(modules)
            },
        },
    )

    manifest = build_minwm_static_fp8_transformer(
        input_dir=str(source),
        calibration_path=str(calibration),
        output_dir=str(output),
    )

    converted = load_file(output / shard_name, device="cpu")
    assert manifest["quantized_weights"] == 300
    assert converted["blocks.0.self_attn.q.weight"].dtype == torch.float8_e4m3fn
    assert converted["blocks.0.self_attn.q.weight_scale"].shape == (1,)
    assert torch.equal(
        converted["blocks.0.self_attn.q.input_scale"],
        torch.tensor([1.0 / 448.0], dtype=torch.float32),
    )
    assert converted["action_in.proj.weight"].dtype == torch.bfloat16
    output_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert output_config["quantization_config"] == {
        "activation_scheme": "static",
        "ignored_layers": [],
        "quant_method": "fp8",
    }

    ffn_output = tmp_path / "ffn-output"
    ffn_manifest = build_minwm_static_fp8_transformer(
        input_dir=str(source),
        calibration_path=str(calibration),
        output_dir=str(ffn_output),
        module_scope="ffn",
    )
    ffn_converted = load_file(ffn_output / shard_name, device="cpu")
    assert ffn_manifest["module_scope"] == "ffn"
    assert ffn_manifest["quantized_weights"] == 60
    assert ffn_converted["blocks.0.ffn.0.weight"].dtype == torch.float8_e4m3fn
    assert ffn_converted["blocks.0.self_attn.q.weight"].dtype == torch.bfloat16
    assert "blocks.0.ffn.0.input_scale" in ffn_converted
    assert "blocks.0.self_attn.q.input_scale" not in ffn_converted
    ffn_config = json.loads((ffn_output / "config.json").read_text(encoding="utf-8"))
    assert ffn_config["quantization_config"] == {
        "activation_scheme": "static",
        "ignored_layers": ["to_q", "to_k", "to_v", "to_out", "attn2"],
        "quant_method": "fp8",
    }


def test_ffn_scope_selects_runtime_quant_methods():
    from sglang.multimodal_gen.runtime.layers.linear import (
        ReplicatedLinear,
        UnquantizedLinearMethod,
    )
    from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
        Fp8Config,
        Fp8LinearMethod,
    )

    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="static",
        ignored_layers=["to_q", "to_k", "to_v", "to_out", "attn2"],
    )
    layer = ReplicatedLinear.__new__(ReplicatedLinear)

    for prefix in (
        "blocks.0.to_q",
        "blocks.0.to_k",
        "blocks.0.to_v",
        "blocks.0.to_out",
        "blocks.0.attn2.to_q",
        "blocks.0.attn2.to_out",
    ):
        assert isinstance(
            config.get_quant_method(layer, prefix), UnquantizedLinearMethod
        )
    assert isinstance(
        config.get_quant_method(layer, "blocks.0.ffn.fc_in"), Fp8LinearMethod
    )
    assert isinstance(
        config.get_quant_method(layer, "blocks.0.ffn.fc_out"), Fp8LinearMethod
    )
