import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.loader.minimax_h3_weights import (
    minimax_h3_safetensors_from_comfy_export,
)

TENSORS = {"blocks.0.attn.qkv.weight": torch.zeros(6, 4, dtype=torch.bfloat16)}


def _write(tmp_path, name, metadata):
    path = tmp_path / name
    save_file(TENSORS, str(path), metadata=metadata)
    return str(path)


def test_comfy_export_is_detected(tmp_path):
    path = _write(tmp_path, "comfy.safetensors", {"config": "{}"})
    assert minimax_h3_safetensors_from_comfy_export([path])


def test_diffusers_export_is_not_detected(tmp_path):
    path = _write(tmp_path, "diffusers.safetensors", {"format": "pt"})
    assert not minimax_h3_safetensors_from_comfy_export([path])


def test_missing_metadata_is_not_detected(tmp_path):
    path = _write(tmp_path, "bare.safetensors", None)
    assert not minimax_h3_safetensors_from_comfy_export([path])


def test_any_comfy_shard_is_detected(tmp_path):
    diffusers = _write(tmp_path, "diffusers-00001.safetensors", {"format": "pt"})
    comfy = _write(tmp_path, "comfy-00002.safetensors", {"config": "{}"})
    assert minimax_h3_safetensors_from_comfy_export([diffusers, comfy])


def test_non_safetensors_paths_are_ignored():
    assert not minimax_h3_safetensors_from_comfy_export(["model.gguf"])
    assert not minimax_h3_safetensors_from_comfy_export([])
