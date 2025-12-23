"""TileLang GEMM Wrapper - unified calling interface."""
import logging
import os
from typing import Dict, List, Tuple

import torch

from sglang.srt.layers.tilelang_gemm_wrapper.core.config_loader import ConfigLoader
from sglang.srt.layers.tilelang_gemm_wrapper.core.kernel_registry import (
    get_kernel_factory,
    is_registry_available,
)

logger = logging.getLogger(__name__)


class TileLangGEMMWrapper:
    """TileLang GEMM unified calling interface."""

    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = os.path.join(os.path.dirname(__file__), "config")

        self.config_loader = ConfigLoader(config_dir)
        self._kernel_cache: Dict[Tuple, Tuple] = {}
        self._partial_buffer_cache: Dict[Tuple, torch.Tensor] = {}

        if not is_registry_available():
            logger.warning(
                "TileLang kernel registry is not available. "
                "GEMM operations will fail until kernels are properly configured."
            )

    def _build_compile_config(
        self,
        config: dict,
        M: int,
        N: int,
        K: int
    ) -> dict:
        """Build compile config from tuned config."""
        kernel_type = config["kernel_type"]
        info = get_kernel_factory(kernel_type)
        is_swap_ab = info["is_swap_ab"]
        scale_shm_key = info["scale_shm_key"]

        compile_config = {
            "M": N if is_swap_ab else M,
            "N": M if is_swap_ab else N,
            "K": K,
            "block_M": config["block_M"],
            "block_N": config["block_N"],
            "block_K": config["block_K"],
            "num_stages": config["num_stages"],
            "threads": config["threads"],
            "out_dtype": config.get("out_dtype", "bfloat16"),
            "accum_dtype": config.get("accum_dtype", "float32"),
            "c_scale_local": config.get("c_scale_local", False),
        }

        if info["has_split_k"]:
            compile_config["split_k"] = config["split_k"]

        compile_config[scale_shm_key] = config.get(scale_shm_key, False)

        return compile_config

    def _get_kernel(self, M: int, N: int, K: int):
        """Get or compile kernel for given dimensions."""
        config = self.config_loader.find_config(M, N, K)
        tuned_M = self.config_loader.get_tuned_M(M, N, K)
        kernel_type = config["kernel_type"]

        cache_key = (tuned_M, N, K, kernel_type)
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]

        info = get_kernel_factory(kernel_type)
        factory = info["factory"]
        compile_config = self._build_compile_config(config, tuned_M, N, K)

        logger.debug(f"Compiling kernel: M={M}, N={N}, K={K}, type={kernel_type}")
        kernel = factory.par_compile([compile_config], num_workers=16)[0]

        result = (kernel, config, info)
        self._kernel_cache[cache_key] = result
        return result

    def _get_partial_buffer(
        self,
        split_k: int,
        M: int,
        N: int,
        device: torch.device
    ) -> torch.Tensor:
        """Get or create C_partial buffer for splitK."""
        cache_key = (split_k, M, N)
        if cache_key not in self._partial_buffer_cache:
            self._partial_buffer_cache[cache_key] = torch.zeros(
                split_k, M, N, device=device, dtype=torch.float32
            )
        return self._partial_buffer_cache[cache_key]

    def gemm(
        self,
        A_fp8: torch.Tensor,
        B_fp8: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        C: torch.Tensor,
    ) -> None:
        """Execute GEMM: C = A @ B^T with blockwise scaling.

        Args:
            A_fp8: (M, K), float8_e4m3, activation
            B_fp8: (N, K), float8_e4m3, weight
            A_scale: scale tensor for A
            B_scale: scale tensor for B
            C: (M, N), output buffer, bfloat16
        """
        M, K = A_fp8.shape
        N = B_fp8.shape[0]

        kernel, config, info = self._get_kernel(M, N, K)
        is_swap_ab = info["is_swap_ab"]
        has_split_k = info["has_split_k"]

        if has_split_k:
            split_k = config["split_k"]
            C_partial = self._get_partial_buffer(split_k, M, N, A_fp8.device)

            if is_swap_ab:
                kernel(B_fp8, A_fp8, C_partial, C, B_scale, A_scale)
            else:
                kernel(A_fp8, B_fp8, C_partial, C, A_scale, B_scale)
        else:
            if is_swap_ab:
                kernel(B_fp8, A_fp8, C, B_scale, A_scale)
            else:
                kernel(A_fp8, B_fp8, C, A_scale, B_scale)

    def warmup(self, shapes: List[Tuple[int, int, int]]) -> None:
        """Pre-compile kernels for specified shapes."""
        for M, N, K in shapes:
            try:
                self._get_kernel(M, N, K)
                logger.debug(f"Warmed up kernel for M={M}, N={N}, K={K}")
            except FileNotFoundError as e:
                logger.warning(f"Warmup skipped for M={M}, N={N}, K={K}: {e}")
            except Exception as e:
                logger.warning(f"Warmup failed for M={M}, N={N}, K={K}: {e}")

    def get_kernel_info(self, M: int, N: int, K: int) -> dict:
        """Get kernel info for given shape (for debugging)."""
        config = self.config_loader.find_config(M, N, K)
        tuned_M = self.config_loader.get_tuned_M(M, N, K)

        return {
            "kernel_type": config["kernel_type"],
            "tuned_M": tuned_M,
            "actual_M": M,
            "block_M": config["block_M"],
            "block_N": config["block_N"],
            "block_K": config["block_K"],
            "num_stages": config["num_stages"],
            "threads": config["threads"],
            "split_k": config.get("split_k"),
            "latency_ms": config.get("latency_ms"),
            "tflops": config.get("tflops"),
        }

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._kernel_cache.clear()
        self._partial_buffer_cache.clear()
        self.config_loader.clear_cache()

    def list_available_configs(self) -> List[Tuple[int, int]]:
        """List all available (N, K) configurations."""
        return self.config_loader.list_available_configs()
