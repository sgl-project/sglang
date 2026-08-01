"""Tuned launch configs for the Inkling MoE ``silu_and_mul`` Triton kernel.

Configs live in ``configs/layout=<layout>,device_name=<device>.json`` as
``{dtype: {half_dim: {BLOCK_SIZE_M, BLOCK_SIZE_N, num_warps}}}``. Regenerate with
``benchmark/kernels/inkling_silu_and_mul/tuning_inkling_silu_and_mul.py``.

Keyed on N only: the best config barely moves with the row count (one config per
N costs <=0.4% across M in [1, 16384]), so tuning sweeps large M and small M
inherits. A device with no tuned file falls back to ``default_config``.
"""

from __future__ import annotations

import functools
import json
import logging
import os

import torch
import triton

from sglang.srt.utils import get_device_name

logger = logging.getLogger(__name__)

SiluAndMulConfig = dict[str, int]


def config_file_name(use_interleaved: bool, device_name: str | None = None) -> str:
    layout = "interleaved" if use_interleaved else "contiguous"
    device = (device_name or get_device_name()).replace(" ", "_")
    return f"layout={layout},device_name={device}.json"


def configs_dir() -> str:
    return os.environ.get(
        "SGLANG_INKLING_SILU_CONFIG_DIR",
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs"),
    )


@functools.lru_cache
def _load_table(use_interleaved: bool) -> dict[str, dict[str, SiluAndMulConfig]]:
    path = os.path.join(configs_dir(), config_file_name(use_interleaved))
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        table = json.load(f)
    logger.info("Using Inkling silu_and_mul configs from %s.", path)
    return table


def default_config(half_dim: int) -> SiluAndMulConfig:
    """Fallback config: a ~2048-element tile at 8 elements/lane."""
    block_n = min(512, triton.next_power_of_2(half_dim))
    while block_n > 128 and half_dim % block_n != 0:
        block_n //= 2
    block_m = 4
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "num_warps": max(1, min(8, block_m * block_n // 256)),
    }


@functools.lru_cache
def get_config(
    use_interleaved: bool, dtype: torch.dtype, half_dim: int
) -> SiluAndMulConfig:
    per_dtype = _load_table(use_interleaved).get(str(dtype))
    if per_dtype:
        tuned = per_dtype.get(str(half_dim))
        if tuned is not None:
            return tuned
    return default_config(half_dim)
