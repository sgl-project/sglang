# SPDX-License-Identifier: Apache-2.0
"""Env-gated parity dump/probe helpers (no-op in production).

Single home for the debug harness that localized the realtime<->batch parity
bugs (previously copy-pasted across streaming/realtime/refiner/sana_wm).

Env vars:
- ``SANAWM_FORK_DUMP_DIR``  — batch-path dumps (init_noise, conditioning,
  per-chunk stage-1, kv/refiner probes, weights fingerprint)
- ``SANAWM_RT_DUMP_DIR``    — realtime-path dumps (same set)
- ``SANAWM_BLOCK_PROBE``    — file path: per-block forward_long checksums on
  the first sink-path call
- ``SANAWM_INJECT_DIR``     — batch-path injection inputs (read, not written)

All checksums reduce in float64 (bf16/fp32 upcasts are lossless, so the
formula is bitwise-stable across both paths).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

ENV_FORK_DUMP = "SANAWM_FORK_DUMP_DIR"
ENV_RT_DUMP = "SANAWM_RT_DUMP_DIR"
ENV_BLOCK_PROBE = "SANAWM_BLOCK_PROBE"
ENV_INJECT = "SANAWM_INJECT_DIR"


def probe_dir(*env_names: str) -> Path | None:
    """First configured dump dir among ``env_names`` (created on demand)."""
    for name in env_names:
        value = os.environ.get(name)
        if value:
            path = Path(value)
            path.mkdir(parents=True, exist_ok=True)
            return path
    return None


def dump_tensor(dirpath: Path | str | None, name: str, tensor) -> None:
    """Save ``tensor`` as float32 CPU under ``dirpath/name.pt`` (None-safe)."""
    if dirpath is None or tensor is None:
        return
    torch.save(tensor.detach().float().cpu(), Path(dirpath) / f"{name}.pt")


def dump_obj(dirpath: Path | str | None, name: str, obj) -> None:
    """Save a picklable object (e.g. a checksum dict) under ``dirpath/name.pt``."""
    if dirpath is None or obj is None:
        return
    torch.save(obj, Path(dirpath) / f"{name}.pt")


def checksum(tensor) -> tuple[tuple[int, ...], float] | None:
    """(shape, float64 sum) of a tensor; None passes through."""
    if tensor is None:
        return None
    return (
        tuple(tensor.shape),
        float(tensor.detach().float().double().sum().item()),
    )


def kv_cache_checksums(chunk_kv, sink_num: int) -> dict:
    """Per-block, per-slot checksums of an accumulated KV cache."""
    probe: dict = {"sink_num": sink_num}
    for block_id, slots in enumerate(chunk_kv):
        for slot_id, tensor in enumerate(slots):
            if tensor is not None:
                probe[f"b{block_id:02d}s{slot_id}"] = checksum(tensor)
    return probe


def weights_fingerprint(module: torch.nn.Module) -> dict[str, float]:
    """Per-parameter |.|-sum fingerprint (catches wrong/mutated weights)."""
    return {
        name: float(param.detach().float().abs().sum().item())
        for name, param in module.named_parameters()
    }
