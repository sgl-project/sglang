"""Compute disk-side weight checksums comparable to server-side checksums.

The server computes ``compute_weights_checksum(module.named_parameters())``.
This module provides ``compute_disk_checksum(module, weights_dir)`` which
loads safetensors from disk, applies the same transforms the model loader
applies (QKV merge, prefix stripping, buffer filtering, dtype casting),
and produces a directly comparable SHA-256 digest.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Iterator

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.loader.utils import _list_safetensors_files
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    compute_weights_checksum,
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

def _get_stacked_params_mapping(module: nn.Module) -> list[tuple[str, str, str]]:
    """Return the stacked-params mapping (e.g. QKV merge) for *module*, or []."""
    return list(getattr(module, "_stacked_params_mapping", []))


def _merge_stacked_params(
    weights_iter: Iterable[tuple[str, torch.Tensor]],
    stacked_params_mapping: list[tuple[str, str, str]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Merge shard tensors (q/k/v_proj) into fused tensors (qkv_proj).

    Replicates the concatenation that QKVParallelLinear.weight_loader performs
    at model-load time.  Concatenation is along dim 0 (output dimension).
    """
    if not stacked_params_mapping:
        yield from weights_iter
        return

    shard_lookup: dict[str, tuple[str, str]] = {
        shard_name: (merged_name, shard_id)
        for merged_name, shard_name, shard_id in stacked_params_mapping
    }

    shard_order: dict[str, list[str]] = defaultdict(list)
    for merged_name, _, shard_id in stacked_params_mapping:
        if shard_id not in shard_order[merged_name]:
            shard_order[merged_name].append(shard_id)

    buffers: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    merged_name_for_key: dict[str, str] = {}

    for name, tensor in weights_iter:
        matched = False
        for shard_name, (merged_name, shard_id) in shard_lookup.items():
            if shard_name not in name:
                continue
            full_merged_key = name.replace(shard_name, merged_name)
            buffers[full_merged_key][shard_id] = tensor
            merged_name_for_key[full_merged_key] = merged_name
            matched = True
            break
        if not matched:
            yield name, tensor

    for full_key, shard_tensors in buffers.items():
        merged_name = merged_name_for_key[full_key]
        order = shard_order.get(merged_name, sorted(shard_tensors.keys()))
        missing = [sid for sid in order if sid not in shard_tensors]
        if missing:
            logger.warning(
                "Incomplete stacked-param shards for '%s': missing %s – skipping.",
                full_key,
                missing,
            )
            continue
        yield full_key, torch.cat([shard_tensors[sid] for sid in order], dim=0)


def _strip_prefix_if_needed(
    weights: dict[str, torch.Tensor],
    param_names: set[str],
) -> dict[str, torch.Tensor]:
    """Auto-detect and strip a common key prefix when disk names don't match."""
    if not weights or weights.keys() & param_names:
        return weights

    # Auto-detect: take one disk key, try stripping progressively longer
    # dot-delimited prefixes until the remainder matches a param name.
    sample_key = next(iter(weights))
    parts = sample_key.split(".")
    for i in range(1, len(parts)):
        prefix = ".".join(parts[:i]) + "."
        if sample_key[len(prefix):] in param_names:
            return {
                k[len(prefix):] if k.startswith(prefix) else k: v
                for k, v in weights.items()
            }

    return weights


def compute_disk_checksum(
    module: nn.Module,
    weights_dir: str,
) -> str:
    """Compute SHA-256 of on-disk weights, comparable to the server's checksum.

    Transforms applied (matching what the model loader does):
    1. QKV / gate-up merge (via module._stacked_params_mapping)
    2. Prefix stripping (e.g. ``model.`` in HF checkpoints)
    3. Buffer filtering (keep only named_parameters() keys)
    4. Dtype casting (e.g. float16 on disk → float32 for force_upcast VAE)
    5. Augment with memory-only params absent from disk
    """
    files = _list_safetensors_files(weights_dir)
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {weights_dir}")

    # Load and transform
    raw_iter = safetensors_weights_iterator(files)
    stacked_mapping = _get_stacked_params_mapping(module)
    merged = _merge_stacked_params(raw_iter, stacked_mapping)
    weights = dict(merged)

    param_dict = dict(module.named_parameters())
    param_names = set(param_dict)

    weights = _strip_prefix_if_needed(weights, param_names)
    weights = {k: v for k, v in weights.items() if k in param_names}

    # Dtype alignment — checksum hashes raw bytes
    for name, tensor in weights.items():
        target_dtype = param_dict[name].dtype
        if tensor.dtype != target_dtype:
            weights[name] = tensor.to(target_dtype)

    # Augment with params present in model but absent from disk
    for name, param in param_dict.items():
        if name not in weights:
            weights[name] = param.detach().cpu()

    return compute_weights_checksum(weights.items())
