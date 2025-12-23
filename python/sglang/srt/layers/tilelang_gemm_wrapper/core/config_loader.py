"""Config loading and M-value lookup for TileLang GEMM."""
import json
import os
from typing import Dict, List, Tuple

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


class ConfigLoader:
    """Config loader for GEMM configurations.

    Config file format (N={N}_K={K}.json):
    {
        "1": {"kernel_type": "splitK_swapAB", "block_M": 64, ...},
        "128": {...},
        ...
    }
    """

    def __init__(self, config_dir: str):
        self.config_dir = config_dir
        self._config_cache: Dict[Tuple[int, int], Dict[int, dict]] = {}

    def _get_config_path(self, N: int, K: int) -> str:
        return os.path.join(self.config_dir, f"N={N}_K={K}.json")

    def config_exists(self, N: int, K: int) -> bool:
        return os.path.exists(self._get_config_path(N, K))

    def load_config(self, N: int, K: int) -> Dict[int, dict]:
        """Load config for (N, K)."""
        cache_key = (N, K)
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]

        config_path = self._get_config_path(N, K)
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config not found: {config_path}. "
                f"Please run tuner first for N={N}, K={K}"
            )

        with open(config_path, "r") as f:
            raw_config = json.load(f)

        config = {int(k): v for k, v in raw_config.items()}
        self._config_cache[cache_key] = config
        return config

    def find_config(self, M: int, N: int, K: int) -> dict:
        """Find closest config for M."""
        configs = self.load_config(N, K)

        if not configs:
            raise ValueError(f"Empty config for N={N}, K={K}")

        closest_M = min(configs.keys(), key=lambda x: abs(x - M))
        return configs[closest_M]

    def get_tuned_M(self, M: int, N: int, K: int) -> int:
        """Get closest tuned M value for kernel compilation."""
        configs = self.load_config(N, K)
        return min(configs.keys(), key=lambda x: abs(x - M))

    def save_config(self, N: int, K: int, config: Dict[int, dict]) -> str:
        """Save config to file."""
        os.makedirs(self.config_dir, exist_ok=True)
        config_path = self._get_config_path(N, K)

        sorted_config = {str(k): v for k, v in sorted(config.items())}

        with open(config_path, "w") as f:
            json.dump(sorted_config, f, indent=4)

        self._config_cache[(N, K)] = config
        return config_path

    def clear_cache(self) -> None:
        self._config_cache.clear()

    def list_available_configs(self) -> List[Tuple[int, int]]:
        """List all available (N, K) configs."""
        configs = []
        if not os.path.exists(self.config_dir):
            return configs

        for filename in os.listdir(self.config_dir):
            if filename.startswith("N=") and filename.endswith(".json"):
                try:
                    parts = filename[:-5].split("_")
                    N = int(parts[0].split("=")[1])
                    K = int(parts[1].split("=")[1])
                    configs.append((N, K))
                except (IndexError, ValueError):
                    continue

        return sorted(configs)

    def get_available_M_values(self, N: int, K: int) -> List[int]:
        """Get available M values for (N, K) config."""
        configs = self.load_config(N, K)
        return sorted(configs.keys())


def get_default_m_values() -> List[int]:
    return DEFAULT_M_VALUES.copy()
