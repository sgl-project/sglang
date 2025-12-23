"""Config loading and M-value lookup for TileLang GEMM."""
import json
import os
import re
from typing import Dict, List, Optional, Tuple

from sglang.srt.utils import get_device_name

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

# Fixed values for TileLang FP8 GEMM
DEFAULT_DTYPE = "fp8_w8a8"
DEFAULT_BLOCK_SHAPE = (128, 128)


def _get_current_device_name() -> str:
    """Get current device name, replacing spaces with underscores."""
    return get_device_name().replace(" ", "_")


class ConfigLoader:
    """Config loader for GEMM configurations.

    Config file format:
    N={N},K={K},device_name={device_name},dtype=fp8_w8a8,block_shape=[128, 128].json
    
    Content:
    {
        "1": {"kernel_type": "splitK_swapAB", "block_M": 64, ...},
        "128": {...},
        ...
    }
    """

    def __init__(self, config_dir: str, device_name: Optional[str] = None):
        self.config_dir = config_dir
        self._config_cache: Dict[Tuple[int, int], Dict[int, dict]] = {}
        self._device_name = device_name or _get_current_device_name()
        self._dtype = DEFAULT_DTYPE
        self._block_shape = DEFAULT_BLOCK_SHAPE

    def _get_config_filename(self, N: int, K: int) -> str:
        """Generate config filename."""
        block_n, block_k = self._block_shape
        return f"N={N},K={K},device_name={self._device_name},dtype={self._dtype},block_shape=[{block_n}, {block_k}].json"

    def _get_config_path(self, N: int, K: int) -> str:
        return os.path.join(self.config_dir, self._get_config_filename(N, K))

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
        """List all available (N, K) configs for current device."""
        configs = []
        if not os.path.exists(self.config_dir):
            return configs

        # Pattern to match config files for current device
        pattern = re.compile(
            rf"N=(\d+),K=(\d+),device_name={re.escape(self._device_name)},"
            rf"dtype={re.escape(self._dtype)},block_shape=\[\d+, \d+\]\.json"
        )

        for filename in os.listdir(self.config_dir):
            match = pattern.match(filename)
            if match:
                N = int(match.group(1))
                K = int(match.group(2))
                configs.append((N, K))

        return sorted(configs)

    def get_available_M_values(self, N: int, K: int) -> List[int]:
        """Get available M values for (N, K) config."""
        configs = self.load_config(N, K)
        return sorted(configs.keys())


def get_default_m_values() -> List[int]:
    return DEFAULT_M_VALUES.copy()
