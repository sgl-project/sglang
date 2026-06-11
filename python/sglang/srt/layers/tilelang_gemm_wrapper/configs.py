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
}


def normalize_config(config: dict, M: int, N: int, K: int) -> dict:
    normalized = dict(_BASE_CONFIG)
    normalized.update(config)
    normalized.update({"M": int(M), "N": int(N), "K": int(K)})

    for key in ("block_M", "block_N", "block_K", "num_stages", "threads", "split_k"):
        normalized[key] = int(normalized[key])
    for key in ("c_scale_local", "a_scale_shm", "b_scale_shm"):
        normalized[key] = bool(normalized[key])

    return normalized


def default_config(M: int, N: int, K: int) -> dict:
    # Keep the no-config path conservative until the non-base kernels are
    # validated on the SM89/SM90 matrix.
    return normalize_config(_BASE_CONFIG, M, N, K)


def _matmul_configs(M: int, N: int, K: int) -> List[dict]:
    del N, K
    tiles_m = [64] if M < 128 else [64, 128]
    tiles_n = [16, 32, 64, 128]
    tiles_k = [128]
    stages = [1, 2, 3, 4]
    threads = [128, 256]
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
    tiles_n = [16, 32, 64, 128]
    tiles_k = [128]
    stages = [0, 1, 2]
    threads = [128, 256]
    split_ks = [2, 4, 8]
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


def _validate_search_policy(search_policy: str) -> str:
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


def _fast_sm90_candidate_filter(config: dict) -> bool:
    """Aggressive SM90 shortlist learned from H20 TileLang 0.1.9 tuning runs."""

    kernel_type = config["kernel_type"]
    if config["block_M"] != 64 or config["block_K"] != 128:
        return False
    if config["threads"] != 128 or not config["c_scale_local"]:
        return False

    if kernel_type == "base":
        return config["block_N"] in (16, 32, 64) and config["num_stages"] in (2, 4)
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

    search_policy = _validate_search_policy(search_policy)
    kernel_types = _select_kernel_types(M, N, K, search_policy, kernel_types)

    configs = []
    for kernel_type in kernel_types:
        if kernel_type not in KERNEL_TYPES:
            raise ValueError(
                f"Unknown TileLang FP8 GEMM kernel type {kernel_type}; "
                f"expected one of {KERNEL_TYPES}."
            )
        if M > 32 and kernel_type in SWAP_AB_KERNEL_TYPES:
            continue
        if M > 128 and kernel_type in SPLIT_K_KERNEL_TYPES:
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
                    candidate = {
                        **base,
                        "kernel_type": kernel_type,
                        "c_scale_local": c_scale_local,
                        scale_key: scale_shm,
                        "out_dtype": "bfloat16",
                        "accum_dtype": "float32",
                    }
                    configs.append(normalize_config(candidate, M, N, K))

    if search_policy == "fast_sm90":
        configs = [config for config in configs if _fast_sm90_candidate_filter(config)]

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

    def update(self, other: "SelectedConfigStore") -> None:
        for config in other.as_list():
            self.add(config)

    def get_exact(self, M: int, N: int, K: int) -> Optional[dict]:
        configs = self.configs_by_nk.get((N, K))
        if not configs or M not in configs:
            return None
        return dict(configs[M])

    def select(self, M: int, N: int, K: int) -> dict:
        configs = self.configs_by_nk.get((N, K))
        if not configs:
            return default_config(M, N, K)

        tuned_m = min(configs, key=lambda candidate_m: abs(candidate_m - M))
        selected = dict(configs[tuned_m])
        selected.update({"M": M, "N": N, "K": K, "tuned_M": tuned_m})
        return normalize_config(selected, M, N, K)

    def as_list(self) -> List[dict]:
        configs = []
        for _, by_m in sorted(self.configs_by_nk.items()):
            configs.extend(by_m[m] for m in sorted(by_m))
        return configs

    @classmethod
    def from_file(cls, path: str) -> "SelectedConfigStore":
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
    def from_path(cls, path: str) -> "SelectedConfigStore":
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
