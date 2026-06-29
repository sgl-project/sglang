"""Config selection utilities for TileLang FP8 GEMM."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

KERNEL_TYPES = ("base", "swapAB", "splitK", "splitK_swapAB")
SPLIT_K_KERNEL_TYPES = {"splitK", "splitK_swapAB"}
SWAP_AB_KERNEL_TYPES = {"swapAB", "splitK_swapAB"}
AUTOTUNE_SEARCH_POLICIES = ("full", "family_pruned", "fast_sm90")
SCHEMA_VERSION = 1
_BLOCK_M_VALUES = (64, 128)
_BLOCK_N_VALUES = (16, 32, 64, 128)
_BLOCK_K_VALUES = (128,)
_THREAD_VALUES = (128, 256)
_MATMUL_STAGE_VALUES = (1, 2, 3, 4)
_SPLIT_K_STAGE_VALUES = (0, 1, 2)
_SPLIT_K_VALUES = (2, 4, 8)
_SWIZZLE_PANEL = 8

DEFAULT_M_VALUES = [
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    256,
    512,
    1024,
    1536,
    2048,
    3072,
    4096,
]

_BASE_CONFIG = {
    "kernel_type": "base",
    "block_M": 128,
    "block_N": 128,
    "block_K": 128,
    "num_stages": 2,
    "threads": 128,
    "split_k": 1,
    "out_dtype": "bfloat16",
    "accum_dtype": "float32",
    "c_scale_local": False,
    "a_scale_shm": False,
    "b_scale_shm": False,
    "swizzle_panel": 0,
    "swizzle_order": "row",
}


def normalize_config(config: dict, M: int, N: int, K: int) -> dict:
    normalized = dict(_BASE_CONFIG)
    normalized.update(config)
    normalized.update({"M": int(M), "N": int(N), "K": int(K)})

    for key in ("block_M", "block_N", "block_K", "num_stages", "threads", "split_k"):
        normalized[key] = int(normalized[key])
    for key in ("c_scale_local", "a_scale_shm", "b_scale_shm"):
        normalized[key] = bool(normalized[key])
    normalized["swizzle_panel"] = int(normalized["swizzle_panel"])
    normalized["swizzle_order"] = str(normalized["swizzle_order"])

    return normalized


def default_config(M: int, N: int, K: int) -> dict:
    # Keep the no-config path conservative until the non-base kernels are
    # validated on the SM89/SM90 matrix.
    return normalize_config(_BASE_CONFIG, M, N, K)


def _matmul_configs(M: int, N: int, K: int) -> List[dict]:
    del N, K
    tiles_m = [64] if M < 128 else [64, 128]
    tiles_n = _BLOCK_N_VALUES
    tiles_k = _BLOCK_K_VALUES
    stages = _MATMUL_STAGE_VALUES
    threads = _THREAD_VALUES
    configs = []
    for block_m in tiles_m:
        for block_n in tiles_n:
            for block_k in tiles_k:
                for num_stages in stages:
                    for num_threads in threads:
                        if (block_m * block_k + block_n * block_k) * num_stages >= (
                            256 * 1024
                        ):
                            continue
                        configs.append(
                            {
                                "block_M": block_m,
                                "block_N": block_n,
                                "block_K": block_k,
                                "num_stages": num_stages,
                                "threads": num_threads,
                                "split_k": 1,
                            }
                        )
    return configs


def _matmul_splitk_configs(M: int, N: int, K: int) -> List[dict]:
    del N
    tiles_m = [64] if M < 128 else [64, 128]
    tiles_n = _BLOCK_N_VALUES
    tiles_k = _BLOCK_K_VALUES
    stages = _SPLIT_K_STAGE_VALUES
    threads = _THREAD_VALUES
    split_ks = _SPLIT_K_VALUES
    configs = []
    for block_m in tiles_m:
        for block_n in tiles_n:
            for block_k in tiles_k:
                for num_stages in stages:
                    for num_threads in threads:
                        for split_k in split_ks:
                            if K % split_k != 0:
                                continue
                            k_per_split = K // split_k
                            if k_per_split % block_k != 0:
                                continue
                            if (block_m * block_k + block_n * block_k) * max(
                                num_stages, 1
                            ) >= (256 * 1024):
                                continue
                            configs.append(
                                {
                                    "block_M": block_m,
                                    "block_N": block_n,
                                    "block_K": block_k,
                                    "num_stages": num_stages,
                                    "threads": num_threads,
                                    "split_k": split_k,
                                }
                            )
    return configs


def validate_search_policy(search_policy: str) -> str:
    if search_policy not in AUTOTUNE_SEARCH_POLICIES:
        raise ValueError(
            "TileLang FP8 GEMM autotune search_policy must be one of "
            f"{AUTOTUNE_SEARCH_POLICIES}, got {search_policy}."
        )
    return search_policy


def _pruned_kernel_types(M: int, N: int, K: int) -> Tuple[str, ...]:
    """Return a shape-specialized family shortlist for SM89/SM90 autotuning."""

    if M >= 128:
        return ("base",)
    if M <= 32:
        if N <= 2048 and K >= 2048:
            return ("splitK_swapAB",)
        if K <= 2048:
            return ("swapAB",)
        return ("swapAB", "splitK_swapAB")
    return ("base", "splitK")


def _select_kernel_types(
    M: int,
    N: int,
    K: int,
    search_policy: str,
    kernel_types: Optional[Iterable[str]],
) -> Tuple[str, ...]:
    requested = tuple(kernel_types) if kernel_types is not None else None
    if requested is not None:
        return requested
    if search_policy == "full":
        return KERNEL_TYPES
    return _pruned_kernel_types(M, N, K)


def kernel_type_m_compatibility_error(kernel_type: str, M: int) -> Optional[str]:
    if kernel_type not in KERNEL_TYPES:
        return f"unknown kernel_type={kernel_type}; expected one of {KERNEL_TYPES}"
    if M > 32 and kernel_type in SWAP_AB_KERNEL_TYPES:
        return f"{kernel_type} is only supported for M <= 32; got M={M}"
    if M > 128 and kernel_type in SPLIT_K_KERNEL_TYPES:
        return f"{kernel_type} is only supported for M <= 128; got M={M}"
    return None


def config_compatibility_error(config: dict, M: int, N: int, K: int) -> Optional[str]:
    if M <= 0 or N <= 0 or K <= 0:
        return f"shape dimensions must be positive; got M={M}, N={N}, K={K}"
    if N % 128 != 0 or K % 128 != 0:
        return f"N and K must be multiples of 128; got N={N}, K={K}"

    config = normalize_config(config, M, N, K)
    kernel_type = config["kernel_type"]
    error = kernel_type_m_compatibility_error(kernel_type, M)
    if error is not None:
        return error

    if config["block_M"] not in _BLOCK_M_VALUES:
        return f"block_M must be one of {_BLOCK_M_VALUES}; got {config['block_M']}"
    if config["block_N"] not in _BLOCK_N_VALUES:
        return f"block_N must be one of {_BLOCK_N_VALUES}; got {config['block_N']}"
    if config["block_K"] not in _BLOCK_K_VALUES:
        return f"block_K must be one of {_BLOCK_K_VALUES}; got {config['block_K']}"
    if config["threads"] not in _THREAD_VALUES:
        return f"threads must be one of {_THREAD_VALUES}; got {config['threads']}"
    if config["out_dtype"] != "bfloat16":
        return f"out_dtype must be bfloat16; got {config['out_dtype']}"
    if config["accum_dtype"] != "float32":
        return f"accum_dtype must be float32; got {config['accum_dtype']}"
    if config["swizzle_panel"] < 0:
        return f"swizzle_panel must be non-negative; got {config['swizzle_panel']}"
    if config["swizzle_order"] not in ("row", "column"):
        return (
            "swizzle_order must be one of ('row', 'column'); "
            f"got {config['swizzle_order']}"
        )
    if config["swizzle_panel"] > 0 and kernel_type != "base":
        return "swizzle_panel is currently supported only by the base kernel"

    split_k = config["split_k"]
    if kernel_type in SPLIT_K_KERNEL_TYPES:
        if config["num_stages"] not in _SPLIT_K_STAGE_VALUES:
            return (
                f"{kernel_type} num_stages must be one of {_SPLIT_K_STAGE_VALUES}; "
                f"got {config['num_stages']}"
            )
        if split_k not in _SPLIT_K_VALUES:
            return (
                f"{kernel_type} split_k must be one of {_SPLIT_K_VALUES}; got {split_k}"
            )
        if K % split_k != 0 or (K // split_k) % 128 != 0:
            return (
                f"{kernel_type} requires K/split_k to be divisible by 128; "
                f"got K={K}, split_k={split_k}"
            )
    elif config["num_stages"] not in _MATMUL_STAGE_VALUES:
        return (
            f"{kernel_type} num_stages must be one of {_MATMUL_STAGE_VALUES}; "
            f"got {config['num_stages']}"
        )
    elif split_k != 1:
        return f"{kernel_type} does not use split_k; got split_k={split_k}"

    if kernel_type in SWAP_AB_KERNEL_TYPES:
        if config["a_scale_shm"]:
            return f"{kernel_type} uses b_scale_shm; a_scale_shm must be False"
    elif config["b_scale_shm"]:
        return f"{kernel_type} uses a_scale_shm; b_scale_shm must be False"

    stage_for_smem = max(config["num_stages"], 1)
    if (
        config["block_M"] * config["block_K"] + config["block_N"] * config["block_K"]
    ) * stage_for_smem >= 256 * 1024:
        return (
            "shared-memory tile footprint is too large: "
            f"block_M={config['block_M']}, block_N={config['block_N']}, "
            f"block_K={config['block_K']}, num_stages={config['num_stages']}"
        )

    return None


def is_config_compatible_with_shape(config: dict, M: int, N: int, K: int) -> bool:
    return config_compatibility_error(config, M, N, K) is None


def _is_large_base_shape(M: int, N: int, K: int) -> bool:
    return M >= 48 and (N >= 4096 or K >= 3584)


def _is_large_n_k5120_base_shape(M: int, N: int, K: int) -> bool:
    return M >= 256 and N >= 6144 and K == 5120


def _uses_stage1_large_k_candidate(M: int, N: int, K: int) -> bool:
    if K == 9216 and N <= 2560 and M >= 2048:
        return True
    return K >= 16384 and N <= 5120 and M <= 1024


def _base_swizzle_options(
    M: int,
    N: int,
    K: int,
    base: dict,
    c_scale_local: bool,
    scale_shm: bool,
) -> Tuple[Tuple[int, str], ...]:
    options = [(0, "row")]
    if (
        _is_large_n_k5120_base_shape(M, N, K)
        and base["block_M"] == 64
        and base["block_N"] == 64
        and base["num_stages"] == 2
        and base["threads"] == 128
        and c_scale_local
        and not scale_shm
    ):
        options.append((_SWIZZLE_PANEL, "column"))
    return tuple(options)


def _fast_sm90_candidate_filter(config: dict, M: int, N: int, K: int) -> bool:
    """Aggressive SM90 shortlist learned from H20 TileLang tuning runs."""

    kernel_type = config["kernel_type"]
    if config["block_K"] != 128:
        return False
    if config["threads"] != 128:
        return False

    if kernel_type == "base":
        large_k_stage1 = (
            config["block_M"] == 64
            and config["block_N"] == 32
            and config["num_stages"] == 1
            and not config["c_scale_local"]
            and not config["a_scale_shm"]
            and config["swizzle_panel"] == 0
        )
        if _uses_stage1_large_k_candidate(M, N, K):
            return large_k_stage1
        if not config["c_scale_local"]:
            return False
        if _is_large_base_shape(M, N, K):
            return (
                not config["a_scale_shm"]
                and config["block_M"] in (64, 128)
                and config["block_N"] in (32, 64, 128)
                and config["num_stages"] in (2, 3)
            )
        if config["block_M"] != 64:
            return False
        return config["block_N"] in (16, 32, 64) and config["num_stages"] in (2, 4)

    if config["block_M"] != 64:
        return False
    if not config["c_scale_local"]:
        return False
    if kernel_type == "swapAB":
        return config["block_N"] == 16 and config["num_stages"] in (2, 4)
    if kernel_type == "splitK_swapAB":
        return (
            config["block_N"] == 16
            and config["num_stages"] == 2
            and config["split_k"] in (4, 8)
        )
    if kernel_type == "splitK":
        return (
            config["block_N"] in (16, 32)
            and config["num_stages"] in (1, 2)
            and config["split_k"] in (2, 4, 8)
        )
    return False


def generate_candidate_configs(
    M: int,
    N: int,
    K: int,
    kernel_types: Optional[Iterable[str]] = None,
    search_policy: str = "full",
) -> List[dict]:
    """Generate the legal TileLang FP8 GEMM config space for one shape."""

    search_policy = validate_search_policy(search_policy)
    kernel_types = _select_kernel_types(M, N, K, search_policy, kernel_types)

    configs = []
    for kernel_type in kernel_types:
        if kernel_type not in KERNEL_TYPES:
            raise ValueError(
                f"Unknown TileLang FP8 GEMM kernel type {kernel_type}; "
                f"expected one of {KERNEL_TYPES}."
            )
        if kernel_type_m_compatibility_error(kernel_type, M) is not None:
            continue

        base_configs = (
            _matmul_splitk_configs(M, N, K)
            if kernel_type in SPLIT_K_KERNEL_TYPES
            else _matmul_configs(M, N, K)
        )
        scale_key = (
            "b_scale_shm" if kernel_type in SWAP_AB_KERNEL_TYPES else "a_scale_shm"
        )

        for base in base_configs:
            for c_scale_local in (False, True):
                for scale_shm in (False, True):
                    swizzle_options = (
                        _base_swizzle_options(M, N, K, base, c_scale_local, scale_shm)
                        if kernel_type == "base"
                        else ((0, "row"),)
                    )
                    for swizzle_panel, swizzle_order in swizzle_options:
                        candidate = {
                            **base,
                            "kernel_type": kernel_type,
                            "c_scale_local": c_scale_local,
                            scale_key: scale_shm,
                            "out_dtype": "bfloat16",
                            "accum_dtype": "float32",
                            "swizzle_panel": swizzle_panel,
                            "swizzle_order": swizzle_order,
                        }
                        configs.append(normalize_config(candidate, M, N, K))

    if search_policy == "fast_sm90":
        configs = [
            config for config in configs if _fast_sm90_candidate_filter(config, M, N, K)
        ]

    return configs


@dataclass
class SelectedConfigStore:
    configs_by_nk: Dict[Tuple[int, int], Dict[int, dict]] = field(default_factory=dict)

    def add(self, config: dict) -> None:
        normalized = normalize_config(
            config, int(config["M"]), int(config["N"]), int(config["K"])
        )
        nk = (normalized["N"], normalized["K"])
        self.configs_by_nk.setdefault(nk, {})[normalized["M"]] = normalized

    def update(self, other: SelectedConfigStore) -> None:
        for config in other.as_list():
            self.add(config)

    def get_exact(self, M: int, N: int, K: int) -> Optional[dict]:
        configs = self.configs_by_nk.get((N, K))
        if not configs or M not in configs:
            return None
        return dict(configs[M])

    def get_exact_compatible(self, M: int, N: int, K: int) -> Optional[dict]:
        config = self.get_exact(M, N, K)
        if config is None:
            return None
        if not is_config_compatible_with_shape(config, M, N, K):
            return None
        return config

    def select(self, M: int, N: int, K: int) -> dict:
        configs = self.configs_by_nk.get((N, K))
        if not configs:
            return default_config(M, N, K)

        compatible_configs = {
            candidate_m: config
            for candidate_m, config in configs.items()
            if is_config_compatible_with_shape(config, M, N, K)
        }
        if not compatible_configs:
            return default_config(M, N, K)

        tuned_m = min(compatible_configs, key=lambda candidate_m: abs(candidate_m - M))
        selected = dict(compatible_configs[tuned_m])
        selected.update({"M": M, "N": N, "K": K, "tuned_M": tuned_m})
        return normalize_config(selected, M, N, K)

    def as_list(self) -> List[dict]:
        configs = []
        for _, by_m in sorted(self.configs_by_nk.items()):
            configs.extend(by_m[m] for m in sorted(by_m))
        return configs

    @classmethod
    def from_file(cls, path: str) -> SelectedConfigStore:
        with open(path) as fin:
            payload = json.load(fin)

        store = cls()
        if isinstance(payload, dict) and isinstance(payload.get("configs"), list):
            for config in payload["configs"]:
                store.add(config)
        elif isinstance(payload, list):
            for config in payload:
                store.add(config)
        elif isinstance(payload, dict) and all(str(key).isdigit() for key in payload):
            match = re.search(r"N=(\d+),K=(\d+),", path)
            if match is None:
                raise ValueError(
                    "Legacy TileLang config files must include N=<n>,K=<k> in "
                    "the filename so shape metadata can be recovered."
                )
            N, K = int(match.group(1)), int(match.group(2))
            for M, config in payload.items():
                store.add({**config, "M": int(M), "N": N, "K": K})
        else:
            raise ValueError(
                "TileLang selected config file must contain a `configs` list "
                "or be a list of config objects."
            )
        return store

    @classmethod
    def from_path(cls, path: str) -> SelectedConfigStore:
        if os.path.isdir(path):
            store = cls()
            for filename in sorted(os.listdir(path)):
                if not filename.endswith(".json"):
                    continue
                store.update(cls.from_file(os.path.join(path, filename)))
            return store
        return cls.from_file(path)


def selected_config_payload(
    configs: Iterable[dict],
    metadata: Optional[dict] = None,
) -> dict:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "backend": "tilelang_fp8_gemm",
        "configs": list(configs),
    }
    if metadata:
        payload["metadata"] = metadata
    return payload


def write_selected_config_file(
    path: str,
    configs: Iterable[dict],
    metadata: Optional[dict] = None,
) -> None:
    """Atomically write selected configs in the current export schema."""

    payload = selected_config_payload(configs, metadata)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp_path = os.path.join(directory, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    with open(tmp_path, "w") as fout:
        json.dump(payload, fout, indent=2)
        fout.write("\n")
    os.replace(tmp_path, path)
