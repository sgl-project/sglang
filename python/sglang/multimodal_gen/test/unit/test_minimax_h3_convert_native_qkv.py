# SPDX-License-Identifier: Apache-2.0
"""The offline QKV converter matches the loader's reorder and touches nothing else."""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    _reorder_grouped_qkv_to_qkv,
)
from sglang.multimodal_gen.tools import convert_minimax_h3_native_qkv as convert

HEADS, HEAD_DIM, HIDDEN = 2, 4, 6
CONFIG = {
    "num_layers": 2,
    "token_refiner_num_layers": 1,
    "num_attention_heads": HEADS,
    "attention_head_dim": HEAD_DIM,
}


def _qkv(seed):
    generator = torch.Generator().manual_seed(seed)
    rows = 3 * HEADS * HEAD_DIM
    return torch.randn(rows, HIDDEN, generator=generator).to(torch.bfloat16)


def _write_model(root):
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "config.json").write_text(json.dumps(CONFIG))
    shards = [
        {
            "blocks.0.attn.q_norm.weight": torch.arange(4, dtype=torch.float32),
            "blocks.0.attn.qkv_proj.weight": _qkv(0),
            "blocks.0.mlp.fc1.weight": torch.full((3, 2), 1.5, dtype=torch.bfloat16),
        },
        {
            "blocks.1.attn.qkv_proj.weight": _qkv(1),
            "token_refiner.blocks.0.attn.qkv_proj.weight": _qkv(2),
            "blocks.1.attn.out_proj.weight": torch.ones(2, 3, dtype=torch.bfloat16),
        },
    ]
    for index, tensors in enumerate(shards):
        save_file(
            tensors,
            transformer / f"model-{index}.safetensors",
            metadata={"format": "pt"},
        )
    (transformer / "model.safetensors.index.json").write_text("{}")
    (root / "text_encoder").mkdir()
    (root / "text_encoder" / "config.json").write_text("{}")
    (root / "model_index.json").write_text("{}")
    return shards


def test_converts_qkv_rows_and_nothing_else(tmp_path):
    model, out = tmp_path / "FL2VA", tmp_path / "native"
    shards = _write_model(model)

    assert convert.main([str(model), str(out)]) == 0

    for index, tensors in enumerate(shards):
        name = f"model-{index}.safetensors"
        source, target = model / "transformer" / name, out / "transformer" / name
        assert target.stat().st_size == source.stat().st_size
        with safe_open(source, "pt") as a, safe_open(target, "pt") as b:
            assert a.metadata() == b.metadata()
        converted = load_file(target)
        assert converted.keys() == tensors.keys()
        for key, tensor in tensors.items():
            if key.endswith(".attn.qkv_proj.weight"):
                expected = _reorder_grouped_qkv_to_qkv(
                    tensor,
                    num_query_groups=HEADS,
                    heads_per_group=1,
                    head_dim=HEAD_DIM,
                )
                assert not torch.equal(expected, tensor)
            else:
                expected = tensor
            assert torch.equal(converted[key], expected), key
    config = json.loads((out / "transformer" / "config.json").read_text())
    assert config == {**CONFIG, "checkpoint_qkv_layout": "native"}
    assert (out / "transformer" / "model.safetensors.index.json").is_file()
    # components the tool does not change are links to the source
    assert (out / "text_encoder").is_symlink()
    assert (out / "model_index.json").resolve() == (model / "model_index.json")
    assert not list(out.rglob("*.partial"))


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"checkpoint_qkv_layout": "native"}, "already in the native"),
        ({"quantization_config": {"quant_method": "fp8"}}, "not supported"),
    ],
)
def test_rejects_an_already_native_or_quantized_transformer(tmp_path, extra, message):
    model = tmp_path / "model"
    _write_model(model)
    (model / "transformer" / "config.json").write_text(json.dumps({**CONFIG, **extra}))
    with pytest.raises(SystemExit, match=message):
        convert.main([str(model), str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_refuses_an_existing_output_and_unexpected_shapes(tmp_path):
    model = tmp_path / "model"
    _write_model(model)
    (tmp_path / "out").mkdir()
    with pytest.raises(SystemExit, match="already exists"):
        convert.main([str(model), str(tmp_path / "out")])

    save_file(
        {"blocks.0.attn.qkv_proj.weight": torch.ones(5, HIDDEN)},
        model / "transformer" / "model-0.safetensors",
    )
    with pytest.raises(SystemExit, match="grouped QKV rows"):
        convert.main([str(model), str(tmp_path / "other")])
    # a failed conversion leaves nothing behind
    assert not list(tmp_path.glob("other*"))


def test_missing_layers_are_reported(tmp_path):
    model = tmp_path / "model"
    _write_model(model)
    (model / "transformer" / "model-1.safetensors").unlink()
    with pytest.raises(SystemExit, match="converted 1 QKV tensors, expected 3"):
        convert.main([str(model), str(tmp_path / "out")])
