from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_module(module_name: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ci_register = load_module(
    "ci_register",
    REPO_ROOT / "python" / "sglang" / "test" / "ci" / "ci_register.py",
)
register_cpu_ci = ci_register.register_cpu_ci
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

search_space = load_module(
    "fused_moe_tuning_search_space",
    REPO_ROOT / "benchmark" / "kernels" / "fused_moe_triton" / "search_space.py",
)


def test_estimate_matmul_smem_bytes() -> None:
    config = {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 5,
    }

    assert search_space.estimate_matmul_smem_bytes(config, 2) == 1310720


def test_estimate_matmul_smem_bytes_supports_asymmetric_operand_dtypes() -> None:
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
    }

    assert search_space.estimate_matmul_smem_bytes(config, 2, 1) == 196608


def test_filter_configs_by_smem_keeps_boundary_config() -> None:
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
    }
    smem_bytes = search_space.estimate_matmul_smem_bytes(config, 2)

    assert search_space.filter_configs_by_smem([config], smem_bytes, 2) == [config]
    assert search_space.filter_configs_by_smem([config], smem_bytes - 1, 2) == []


def test_filter_configs_by_smem_reduces_default_cuda_search_space() -> None:
    configs = [
        {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": group_size,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        for num_stages in [2, 3, 4, 5]
        for block_m in [16, 32, 64, 128, 256]
        for block_k in [64, 128, 256]
        for block_n in [32, 64, 128, 256]
        for num_warps in [4, 8]
        for group_size in [1, 16, 32, 64]
    ]

    filtered = search_space.filter_configs_by_smem(configs, 164 * 1024, 2)

    assert len(configs) == 1920
    assert len(filtered) == 1040
