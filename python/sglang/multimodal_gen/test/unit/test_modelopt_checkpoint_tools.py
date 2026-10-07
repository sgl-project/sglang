# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from sglang.multimodal_gen.tools.build_modelopt_fp8_transformer import (
    build_modelopt_fp8_transformer,
)
from sglang.multimodal_gen.tools.build_modelopt_nvfp4_transformer import (
    build_modelopt_nvfp4_transformer,
)
from sglang.multimodal_gen.tools.modelopt_checkpoint import load_weight_map


def _write_checkpoint(path, config, shards, index_name):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config))
    weight_map = {}
    for i, tensors in enumerate(shards):
        filename = f"weights-{i}.safetensors"
        save_file(tensors, path / filename, metadata={"source": "preserved"})
        weight_map.update(dict.fromkeys(tensors, filename))
    if index_name:
        (path / index_name).write_text(json.dumps({"weight_map": weight_map}))


@pytest.mark.parametrize("kind", ["fp8", "nvfp4"])
@pytest.mark.parametrize("keep_bf16", [False, True])
@pytest.mark.parametrize(
    "index_name",
    [
        None,
        "diffusion_pytorch_model.safetensors.index.json",
        "custom.safetensors.index.json",
    ],
)
def test_modelopt_checkpoint_conversion(tmp_path, kind, keep_bf16, index_name):
    source, base, output = (tmp_path / name for name in ("source", "base", "output"))
    quant = {"quant_method": "modelopt", "quant_algo": kind.upper(), "ignore": []}
    weights = torch.arange(32).reshape(4, 8).to(torch.bfloat16)
    serialized = weights if kind == "fp8" else weights.to(torch.uint8)
    shards = [
        {f"{name}.weight": serialized, f"{name}.weight_scale": torch.tensor(1.0)}
        for name in ("quantized", "fallback")
    ]
    shards[0]["quantized.weight_quantizer._amax"] = torch.tensor(448.0)
    if index_name is None:
        shards = [shards[0] | shards[1]]
    _write_checkpoint(
        source / "transformer", {"quantization_config": quant}, shards, index_name
    )
    _write_checkpoint(base, {}, [{"fallback.weight": weights + 17}], None)
    (source / "transformer" / "assets").mkdir()
    (source / "transformer" / "assets" / "notes.txt").write_text("preserved")
    kwargs = dict(
        modelopt_hf_dir=str(source),
        base_transformer_dir=str(base),
        output_dir=str(output),
        keep_bf16_patterns=["fallback"] if keep_bf16 else [],
    )
    if kind == "fp8":
        backbone = tmp_path / "backbone.pt"
        torch.save(
            {
                "model_state_dict": {
                    f"{name}.{quantizer}_quantizer._amax": torch.tensor(448.0)
                    for name in ("quantized", "fallback")
                    for quantizer in ("weight", "input")
                }
            },
            backbone,
        )
        kwargs.update(modelopt_backbone_ckpt=str(backbone), model_type="none")
        build = build_modelopt_fp8_transformer
    else:
        build = build_modelopt_nvfp4_transformer
    stats = build(**kwargs)
    with pytest.raises(FileExistsError, match="--overwrite"):
        build(**kwargs)
    (output / "stale.txt").write_text("remove on explicit overwrite")
    assert build(**kwargs, overwrite=True) == stats
    assert not (output / "stale.txt").exists()
    assert (output / "assets" / "notes.txt").read_text() == "preserved"
    weight_map, output_index = load_weight_map(str(output))
    assert output_index == (index_name or "weights-0.safetensors.index.json")
    assert not any("_quantizer." in key for key in weight_map)
    tensors = {}
    for filename in set(weight_map.values()):
        with safe_open(output / filename, framework="pt") as shard:
            assert shard.metadata()["source"] == "preserved"
            assert (
                json.loads(shard.metadata()["quantization_config"])["quant_algo"]
                == kind.upper()
            )
            tensors.update({key: shard.get_tensor(key) for key in shard.keys()})
    assert set(tensors) == set(weight_map)
    quantized = weights.to(torch.float8_e4m3fn) if kind == "fp8" else serialized
    for name, expected in (
        ("quantized", quantized),
        ("fallback", weights + 17 if keep_bf16 else quantized),
    ):
        actual = tensors[f"{name}.weight"]
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
    assert ("fallback.weight_scale" in tensors) is not keep_bf16
    index = json.loads((output / output_index).read_text())
    assert index["metadata"]["total_size"] == sum(
        t.numel() * t.element_size() for t in tensors.values()
    )


def test_modelopt_index_selection(tmp_path):
    with pytest.raises(ValueError, match="found 0 shard"):
        load_weight_map(str(tmp_path))
    for name in ("a", "b"):
        save_file({name: torch.ones(1)}, tmp_path / f"{name}.safetensors")
    with pytest.raises(ValueError, match="found 2 shard"):
        load_weight_map(str(tmp_path))
    for name in ("z", "a", "diffusion_pytorch_model", "model"):
        filename = f"{name}.safetensors.index.json"
        mapping = {name: "a.safetensors"}
        (tmp_path / filename).write_text(json.dumps({"weight_map": mapping}))
        assert load_weight_map(str(tmp_path)) == (mapping, filename)
