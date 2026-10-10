# SPDX-License-Identifier: Apache-2.0
"""Checkpoint I/O shared by the ModelOpt transformer conversion tools."""

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import torch
from safetensors import safe_open

_INDEX_FILENAMES = (
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
)


def resolve_transformer_dir(path: str) -> str:
    candidate = Path(path).expanduser().resolve()
    if (candidate / "config.json").is_file():
        return str(candidate)
    transformer_dir = candidate / "transformer"
    if (transformer_dir / "config.json").is_file():
        return str(transformer_dir)
    raise FileNotFoundError(f"Could not resolve a transformer directory from: {path}")


def load_weight_map(model_dir: str) -> tuple[dict[str, str], str]:
    index_filename = next(
        (
            name
            for name in _INDEX_FILENAMES
            if os.path.isfile(os.path.join(model_dir, name))
        ),
        None,
    )
    if index_filename is None:
        matches = sorted(
            name
            for name in os.listdir(model_dir)
            if name.endswith(".safetensors.index.json")
        )
        index_filename = matches[0] if matches else None
    if index_filename is not None:
        with open(os.path.join(model_dir, index_filename), encoding="utf-8") as f:
            index_data = json.load(f)
        return dict(index_data["weight_map"]), index_filename

    safetensors_files = sorted(
        filename
        for filename in os.listdir(model_dir)
        if filename.endswith(".safetensors")
    )
    if len(safetensors_files) != 1:
        raise ValueError(
            f"Expected an index file or a single safetensors shard in {model_dir}, "
            f"found {len(safetensors_files)} shard(s)."
        )

    shard_name = safetensors_files[0]
    with safe_open(
        os.path.join(model_dir, shard_name), framework="pt", device="cpu"
    ) as f:
        weight_map = {key: shard_name for key in f.keys()}
    return weight_map, f"{Path(shard_name).stem}.safetensors.index.json"


def load_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def prepare_output_dir(source_dir: str, output_dir: str, *, overwrite: bool) -> Path:
    output_path = Path(output_dir).expanduser().resolve()
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_path}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for entry in os.listdir(source_dir):
        if entry.endswith(".safetensors") or entry in _INDEX_FILENAMES:
            continue
        source_path = os.path.join(source_dir, entry)
        destination = output_path / entry
        if os.path.isdir(source_path):
            shutil.copytree(source_path, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, destination)
    return output_path


def load_selected_tensors(
    model_dir: str,
    weight_map: Mapping[str, str],
    tensor_names: Iterable[str],
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    names_by_file: dict[str, list[str]] = defaultdict(list)
    for name in tensor_names:
        names_by_file[weight_map[name]].append(name)

    for filename, names in names_by_file.items():
        shard_path = os.path.join(model_dir, filename)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name in names:
                tensors[name] = f.get_tensor(name).contiguous()
    return tensors
