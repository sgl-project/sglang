"""TileLang GEMM Wrapper - unified calling interface."""
import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.tilelang_gemm_wrapper.core.config_loader import ConfigLoader
from sglang.srt.layers.tilelang_gemm_wrapper.core.kernel_registry import (
    get_kernel_factory,
    is_registry_available,
)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# Global config (similar to DeepGEMM's _BUILTIN_M_LIST)
_M_MAX = 1024 * 16  # Default m_max
_DO_COMPILE_ALL = True
_INITIALIZATION_DICT: Dict[Tuple[int, int], bool] = dict()


def update_tilelang_config(gpu_id: int, server_args: "ServerArgs"):
    """Update TileLang config based on server args (similar to DeepGEMM).
    
    Args:
        gpu_id: Current GPU ID
        server_args: Server arguments
    """
    global _M_MAX, _DO_COMPILE_ALL
    
    # Generate m_max (same logic as DeepGEMM)
    m_max = 1024 * 16
    if server_args.chunked_prefill_size < 1:
        m_max = 1024 * 64
    elif server_args.chunked_prefill_size > 8192:
        m_max = server_args.chunked_prefill_size * 2
    _M_MAX = min(1024 * 128, m_max)
    
    # Only first rank on node does compilation
    _DO_COMPILE_ALL = server_args.base_gpu_id == gpu_id
    
    logger.info(f"TileLang config updated: m_max={_M_MAX}, do_compile_all={_DO_COMPILE_ALL}")


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
        """Get or compile kernel for given dimensions.
        
        Finds the closest tuned M config and uses that kernel.
        Since M is dynamic (tvm.te.var), kernels compiled with tuned_M
        can handle any actual M value.
        """
        config = self.config_loader.find_config(M, N, K)
        tuned_M = self.config_loader.get_tuned_M(M, N, K)
        kernel_type = config["kernel_type"]

        # Cache key uses tuned_M (kernel is compiled with dynamic M)
        cache_key = (tuned_M, N, K, kernel_type)
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]

        info = get_kernel_factory(kernel_type)
        factory = info["factory"]
        compile_config = self._build_compile_config(config, tuned_M, N, K)

        logger.debug(f"Compiling kernel: tuned_M={tuned_M}, N={N}, K={K}, type={kernel_type}")
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

    def warmup_all_m(self, N: int, K: int, m_max: Optional[int] = None) -> None:
        """Pre-compile all tuned M kernels for specified (N, K).
        
        Since M is dynamic (tvm.te.var), we only need to compile kernels for
        the tuned M values in config. At runtime, any actual M will use the
        closest tuned_M kernel.
        Uses parallel compilation for better performance.
        
        Args:
            N: Weight N dimension
            K: Weight K dimension  
            m_max: Maximum M value to warmup. If specified, only compile
                   tuned M values <= m_max.
        """
        # Check if config exists for this (N, K)
        if not self.config_loader.config_exists(N, K):
            logger.warning(f"No config found for N={N}, K={K}, skipping warmup")
            return
        
        # Get tuned M values from config (not all M from 1 to m_max)
        tuned_m_values = self.config_loader.get_available_M_values(N, K)
        
        # Filter by m_max if specified
        if m_max is not None:
            tuned_m_values = [m for m in tuned_m_values if m <= m_max]
        
        # Filter out already cached kernels and group by kernel_type for parallel compilation
        # Structure: {kernel_type: [(cache_key, compile_config, config, info), ...]}
        to_compile: Dict[str, List[Tuple]] = {}
        
        for tuned_M in tuned_m_values:
            config = self.config_loader.find_config(tuned_M, N, K)
            kernel_type = config["kernel_type"]
            
            # Cache key uses tuned_M (kernel handles dynamic M at runtime)
            cache_key = (tuned_M, N, K, kernel_type)
            if cache_key in self._kernel_cache:
                continue
            
            info = get_kernel_factory(kernel_type)
            compile_config = self._build_compile_config(config, tuned_M, N, K)
            
            if kernel_type not in to_compile:
                to_compile[kernel_type] = []
            to_compile[kernel_type].append((cache_key, compile_config, config, info))
        
        if not to_compile:
            logger.info(f"TileLang warmup: N={N}, K={K}, all kernels already cached")
            return
        
        total_kernels = sum(len(v) for v in to_compile.values())
        logger.info(f"TileLang warmup: N={N}, K={K}, compiling {total_kernels} kernels in parallel...")
        
        # Parallel compile for each kernel_type
        for kernel_type, items in to_compile.items():
            if not items:
                continue
            
            info = items[0][3]  # All items have same info for same kernel_type
            factory = info["factory"]
            compile_configs = [item[1] for item in items]
            
            try:
                kernels = factory.par_compile(compile_configs, num_workers=128)
                
                # Store compiled kernels in cache
                for (cache_key, _, config, info), kernel in zip(items, kernels):
                    self._kernel_cache[cache_key] = (kernel, config, info)
            except Exception as e:
                logger.warning(f"Parallel compile failed for kernel_type={kernel_type}: {e}")
        
        logger.info(f"TileLang warmup: N={N}, K={K}, compilation complete")

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


# Global wrapper instance for warmup hook
_global_wrapper: Optional[TileLangGEMMWrapper] = None
_global_wrapper_config_dir: Optional[str] = None


def set_global_wrapper_config(config_dir: Optional[str] = None) -> None:
    """Set config directory for global wrapper before initialization."""
    global _global_wrapper_config_dir
    _global_wrapper_config_dir = config_dir


def get_global_wrapper() -> TileLangGEMMWrapper:
    """Get or create global TileLangGEMMWrapper instance."""
    global _global_wrapper
    if _global_wrapper is None:
        _global_wrapper = TileLangGEMMWrapper(config_dir=_global_wrapper_config_dir)
    return _global_wrapper


def _maybe_compile_tilelang_all(n: int, k: int) -> None:
    """Compile all M kernels for (N, K) if not already done.
    
    Similar to DeepGEMM's _maybe_compile_deep_gemm_one_type_all.
    """
    global _INITIALIZATION_DICT
    
    cache_key = (n, k)
    if cache_key in _INITIALIZATION_DICT:
        return
    
    if not _DO_COMPILE_ALL:
        _INITIALIZATION_DICT[cache_key] = True
        return
    
    wrapper = get_global_wrapper()
    wrapper.warmup_all_m(n, k, m_max=_M_MAX)
    _INITIALIZATION_DICT[cache_key] = True


@contextmanager
def tilelang_execution_hook(n: int, k: int):
    """Pre-compile all M kernels for (N, K) before execution.
    
    Similar to DeepGEMM's deep_gemm_execution_hook, this ensures all
    kernels are compiled before the actual execution begins.
    
    Args:
        n: Weight N dimension
        k: Weight K dimension
    
    Usage:
        with tilelang_execution_hook(N, K):
            # All M kernels for (N, K) are now compiled
            ...
    """
    _maybe_compile_tilelang_all(n, k)
    yield
