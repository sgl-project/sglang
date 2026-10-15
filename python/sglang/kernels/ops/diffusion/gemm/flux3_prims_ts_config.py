# SPDX-License-Identifier: Apache-2.0
"""Measured PrimsTS dense GEMM configs for FLUX.3 image resolutions."""

from __future__ import annotations

import dataclasses
import functools
import inspect

import torch
from flashinfer.prims_ts import gemm as prims_ts_gemm
from flashinfer.prims_ts.gemm import PrimsTsGemmConfig

from sglang.srt.environ import envs

# (format, epilogue, K, packed N, M bucket) ->
# (cluster_m, cluster_n, tile_k, overlap, stages, mma_k, use_clc,
#  epilogue_warps, use_tma_store)
_TACTICS = {
    # This shape was not present in the t2i autotune runs; retain the prior
    # measured tactic until it is swept through the updated FlashInfer search.
    ("fp8_e4m3", "linear", 3072, 9216): {
        1024: (2, 2, 128, False, 6, 64, True, 8, False),
        8192: (2, 2, 128, False, 5, 64, True, 4, False),
        10240: (2, 2, 128, False, 6, 64, True, 4, False),
        32768: (2, 2, 128, False, 6, 64, True, 4, False),
        40960: (2, 2, 128, False, 6, 64, True, 8, False),
    },
    ("fp8_e4m3", "linear", 3072, 3072): {
        1024: (2, 2, 128, False, 6, 64, True, 8, True),
        8192: (2, 2, 128, False, 6, 64, True, 4, True),
        10240: (2, 2, 128, False, 6, 64, True, 4, True),
        32768: (4, 4, 128, False, 6, 64, True, 4, True),
        40960: (4, 4, 128, False, 6, 64, True, 4, True),
    },
    ("fp8_e4m3", "swiglu", 3072, 18432): {
        1024: (2, 2, 128, False, 6, 64, True, 8, False),
        8192: (4, 4, 128, False, 6, 64, True, 8, False),
        10240: (4, 4, 128, False, 5, 64, True, 8, False),
        32768: (2, 2, 128, False, 6, 64, True, 8, False),
        40960: (2, 2, 128, False, 5, 64, True, 8, False),
    },
    ("fp8_e4m3", "linear", 9216, 3072): {
        1024: (2, 2, 128, False, 6, 64, True, 4, True),
        8192: (4, 4, 128, False, 6, 64, True, 4, True),
        10240: (2, 2, 128, False, 6, 64, True, 4, True),
        32768: (2, 2, 128, False, 6, 64, True, 4, False),
        40960: (4, 4, 128, False, 6, 64, True, 4, False),
    },
    ("fp8_e4m3", "qkv_qknorm_rope", 3072, 9216): {
        1024: (2, 2, 128, False, 4, 64, True, 8, True),
        8192: (2, 2, 128, False, 4, 64, True, 8, True),
        10240: (2, 2, 128, False, 4, 64, True, 8, True),
        32768: (4, 4, 128, False, 4, 64, True, 8, True),
        40960: (4, 4, 128, False, 4, 64, True, 8, True),
    },
    # This shape was not present in the t2i autotune runs; retain the prior
    # measured tactic until it is swept through the updated FlashInfer search.
    ("nvfp4_e2m1", "linear", 3072, 9216): {
        1024: (2, 2, 256, True, 5, 64, True, 8, False),
        8192: (2, 2, 256, True, 5, 64, True, 8, False),
        10240: (2, 2, 256, True, 5, 64, True, 8, False),
        32768: (2, 2, 256, True, 5, 64, True, 8, False),
        40960: (4, 4, 256, True, 5, 64, True, 8, False),
    },
    ("nvfp4_e2m1", "linear", 3072, 3072): {
        1024: (2, 2, 256, False, 5, 64, True, 8, True),
        8192: (4, 4, 256, True, 5, 64, True, 8, False),
        10240: (2, 2, 256, True, 5, 64, True, 8, False),
        32768: (4, 4, 256, True, 5, 64, True, 8, False),
        40960: (4, 4, 256, True, 5, 64, True, 8, False),
    },
    ("nvfp4_e2m1", "linear", 9216, 3072): {
        1024: (4, 4, 256, False, 5, 64, True, 4, True),
        8192: (2, 2, 256, True, 5, 64, True, 8, False),
        10240: (2, 2, 256, True, 5, 64, True, 8, False),
        32768: (4, 4, 256, True, 5, 64, True, 8, False),
        40960: (4, 4, 256, True, 5, 64, True, 8, False),
    },
    ("nvfp4_e2m1", "swiglu", 3072, 18432): {
        1024: (4, 4, 256, False, 5, 64, True, 8, False),
        8192: (2, 2, 256, False, 5, 64, True, 8, False),
        10240: (4, 4, 256, False, 5, 64, True, 8, False),
        32768: (4, 4, 256, False, 5, 64, True, 8, False),
        40960: (2, 2, 256, False, 5, 64, True, 8, False),
    },
    ("nvfp4_e2m1", "qkv_qknorm_rope", 3072, 9216): {
        1024: (2, 2, 256, False, 4, 64, True, 8, True),
        8192: (2, 2, 256, False, 4, 64, True, 8, True),
        10240: (2, 2, 256, False, 4, 64, True, 8, True),
        32768: (4, 4, 256, False, 4, 64, True, 8, True),
        40960: (4, 4, 256, False, 4, 64, True, 8, True),
    },
}


@functools.cache
def _tmem_overlap_epilogues() -> frozenset[str]:
    # FlashInfer PR 6255 (SwiGLU) and 6286 (QKV) add tmem_overlap to these prepare
    # functions together with allowing it in the kernel config.
    prepare_fns = {
        "swiglu": prims_ts_gemm.prepare_fp4_linear_swiglu,
        "qkv_qknorm_rope": prims_ts_gemm.prepare_fp4_qkv_qknorm_rope,
    }
    return frozenset(
        epilogue
        for epilogue, prepare_fn in prepare_fns.items()
        if "tmem_overlap" in inspect.signature(prepare_fn).parameters
    )


def _with_tmem_overlap(config: PrimsTsGemmConfig) -> PrimsTsGemmConfig:
    # The next tile's MMA runs during this tile's epilogue (bit-identical). The table
    # was tuned without SwiGLU/QKV overlap, so add it once FlashInfer accepts it.
    if (
        config.operand_format != "nvfp4_e2m1"
        or config.tmem_overlap
        or config.epilogue not in _tmem_overlap_epilogues()
        # Kernel constraints for overlap; QKV also needs MMA-K 64.
        or (config.tile_n, config.tile_k, config.epilogue_warps) != (256, 256, 8)
        or (config.epilogue == "qkv_qknorm_rope" and config.nvfp4_mma_k != 64)
    ):
        return config
    # Overlap and the TMA output store cannot share tensor memory.
    return dataclasses.replace(config, tmem_overlap=True, use_tma_store=False)


def flux3_prims_ts_config(
    x: torch.Tensor,
    *,
    operand_format: str,
    epilogue: str,
    k: int,
    n: int,
    m: int | None = None,
    output_format: str = "bf16",
    head_dim: int | None = None,
    qkv_scale: bool = False,
) -> PrimsTsGemmConfig | None:
    """Return a pinned tactic, or leave selection to FlashInfer autotuning."""
    if x.device.type != "cuda" or envs.SGLANG_ENABLE_FLUX3_PRIMS_TS_AUTOTUNE.get():
        return None
    m = x.numel() // x.shape[-1] if m is None else m
    bucket = next((b for b in (1024, 8192, 10240, 32768, 40960) if m <= b), 40960)
    return _tuned_config(
        arch=_device_arch(x.device),
        operand_format=operand_format,
        output_format=output_format,
        epilogue=epilogue,
        k=k,
        n=n,
        bucket=bucket,
        head_dim=head_dim,
        qkv_scale=qkv_scale,
    )


@functools.cache
def _device_arch(device: torch.device) -> int:
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


# Runs for every GEMM launch: one cached config per shape keeps the host cost to a
# lookup, and the prepared-GEMM cache keys on the same object.
@functools.cache
def _tuned_config(
    *,
    arch: int,
    operand_format: str,
    output_format: str,
    epilogue: str,
    k: int,
    n: int,
    bucket: int,
    head_dim: int | None,
    qkv_scale: bool,
) -> PrimsTsGemmConfig:
    tactic = _TACTICS.get((operand_format, epilogue, k, n), {}).get(bucket)
    if tactic is None:
        return PrimsTsGemmConfig(
            arch=arch,
            operand_format=operand_format,
            output_format=output_format,
            epilogue=epilogue,
            has_bias=False,
            head_dim=head_dim,
            is_neox=False if epilogue == "qkv_qknorm_rope" else None,
            has_qkv_scale=qkv_scale,
            # FlashInfer's default tile_k=128 is below the FP4 TMA K-box.
            **({"tile_k": 256} if operand_format == "nvfp4_e2m1" else {}),
        )
    cm, cn, tile_k, overlap, stages, mma_k, use_clc, warps, use_tma_store = tactic
    config = PrimsTsGemmConfig(
        arch=arch,
        operand_format=operand_format,
        output_format=output_format,
        epilogue=epilogue,
        has_bias=False,
        head_dim=head_dim,
        is_neox=False if epilogue == "qkv_qknorm_rope" else None,
        tile_m=256,
        tile_n=256,
        tile_k=tile_k,
        cluster_shape=(cm, cn, 1),
        scheduler="clc_dynamic" if use_clc else "static",
        has_qkv_scale=qkv_scale,
        nvfp4_mma_k=mma_k,
        epilogue_warps=warps,
        tmem_overlap=overlap,
        ab_stages=stages,
        use_tma_store=use_tma_store,
    )
    return _with_tmem_overlap(config)
