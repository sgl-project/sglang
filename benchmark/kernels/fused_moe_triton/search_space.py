from __future__ import annotations

from typing import Dict, List, Optional


BenchmarkConfig = Dict[str, int]


def estimate_matmul_smem_bytes(
    config: BenchmarkConfig,
    a_dtype_bytes: int,
    b_dtype_bytes: Optional[int] = None,
) -> int:
    """Estimate staged shared memory for the fused MoE matmul tile."""
    if b_dtype_bytes is None:
        b_dtype_bytes = a_dtype_bytes

    block_m = config["BLOCK_SIZE_M"]
    block_n = config["BLOCK_SIZE_N"]
    block_k = config["BLOCK_SIZE_K"]
    num_stages = config["num_stages"]
    return (
        block_m * block_k * a_dtype_bytes + block_k * block_n * b_dtype_bytes
    ) * num_stages


def is_config_within_smem_limit(
    config: BenchmarkConfig,
    smem_limit_bytes: int,
    a_dtype_bytes: int,
    b_dtype_bytes: Optional[int] = None,
) -> bool:
    return (
        estimate_matmul_smem_bytes(config, a_dtype_bytes, b_dtype_bytes)
        <= smem_limit_bytes
    )


def filter_configs_by_smem(
    configs: List[BenchmarkConfig],
    smem_limit_bytes: Optional[int],
    a_dtype_bytes: int,
    b_dtype_bytes: Optional[int] = None,
) -> List[BenchmarkConfig]:
    if smem_limit_bytes is None:
        return configs
    return [
        config
        for config in configs
        if is_config_within_smem_limit(
            config,
            smem_limit_bytes,
            a_dtype_bytes,
            b_dtype_bytes,
        )
    ]
