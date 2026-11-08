"""Exact-width hipBLASLt solution picks for the Muse Spark TP8 BF16 GEMMs.

Decode / verify graphs run each linear at a few fixed row counts (M = draft tokens x
captured batch size). At these widths aiter's tgemm looks the shape up under a padded M,
and torch's hipBLASLt heuristic can add a split-K reduce kernel or pick a tile that leaves
CUs idle. The per-device table configs/muse_spark_width_gemm,device_name=<GPU>.json maps
(N, K, output dtype) and an exact M to (library, solution index, kernel name).

The picks keep BF16 inputs and fp32 accumulation; only the tile / split-K summation order
differs from the default path, so the output is not bitwise equal to it. The table stays
off until the ROCm Muse Spark model enables it (other models share some of these shapes),
and is ignored on another device or ROCm build: hipBLASLt solution indices only hold for
the build they were searched on.
"""

import functools
import json
import logging
import os

import torch

logger = logging.getLogger(__name__)

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
_OUT = {torch.bfloat16: "bf16", torch.float32: "f32"}
_enabled = False


def enable_width_gemm() -> None:
    """Let ``width_gemm_applies`` use this device's table in this process."""
    global _enabled
    _enabled = True


@functools.cache
def _table() -> dict:
    """{(N, K, out): {M: (library, solution index, kernel name)}} for this device and ROCm build."""
    name = torch.cuda.get_device_name().replace(" ", "_")
    path = os.path.join(_CONFIG_DIR, f"muse_spark_width_gemm,device_name={name}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    if raw["_meta"]["rocm"] != torch.version.hip:
        return {}
    table = {}
    for key, by_m in raw.items():
        if key.startswith("_"):
            continue
        n, k, out = key.split(",")
        table[(int(n), int(k), out)] = {int(m): tuple(e) for m, e in by_m.items()}
    logger.info(
        "Muse width GEMM table: %d rows from %s",
        sum(map(len, table.values())),
        os.path.basename(path),
    )
    return table


def width_gemm_applies(
    x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype = torch.bfloat16
) -> bool:
    return (
        _enabled
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.shape[0] in _table().get((*weight.shape, _OUT.get(out_dtype)), ())
    )


def width_gemm(
    x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """``x @ weight.T`` on the table's solution; call only when ``width_gemm_applies``."""
    from aiter.tuned_gemm import solMap

    library, solution, kernel = _table()[(*weight.shape, _OUT[out_dtype])][x.shape[0]]
    config = {"kernelName": kernel, "splitK": None}
    return solMap[library](x, weight, solution, None, out_dtype, config=config)
