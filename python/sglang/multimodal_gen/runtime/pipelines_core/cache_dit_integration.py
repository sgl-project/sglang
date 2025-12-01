# SPDX-License-Identifier: Apache-2.0
"""
cache-dit integration module for SGLang DiT pipelines.

This module provides helper functions to enable cache-dit acceleration
on transformer modules in SGLang's modular pipeline architecture.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Lazy import cache-dit to avoid hard dependency
CACHE_DIT_AVAILABLE = False
try:
    import cache_dit
    from cache_dit import DBCacheConfig, TaylorSeerCalibratorConfig
    from cache_dit.caching.block_adapters import BlockAdapterRegister

    CACHE_DIT_AVAILABLE = True
except ImportError:
    cache_dit = None
    DBCacheConfig = None
    TaylorSeerCalibratorConfig = None
    BlockAdapterRegister = None


def is_cache_dit_available() -> bool:
    """Check if cache-dit is installed and available."""
    return CACHE_DIT_AVAILABLE


@dataclass
class CacheDitConfig:
    """Configuration for cache-dit integration.

    Attributes:
        enabled: Whether to enable cache-dit acceleration.
        Fn_compute_blocks: Number of first blocks to always compute (DBCache F).
        Bn_compute_blocks: Number of last blocks to always compute (DBCache B).
        max_warmup_steps: Number of warmup steps before caching starts (DBCache W).
        residual_diff_threshold: Threshold for residual difference (DBCache R).
        max_continuous_cached_steps: Maximum consecutive cached steps (DBCache MC).
        enable_taylorseer: Whether to enable TaylorSeer calibrator.
        taylorseer_order: Order of Taylor expansion (1 or 2).
        num_inference_steps: Total number of inference steps (required for transformer-only mode).
    """

    enabled: bool = False
    Fn_compute_blocks: int = 1
    Bn_compute_blocks: int = 0
    max_warmup_steps: int = 8
    residual_diff_threshold: float = 0.35
    max_continuous_cached_steps: int = 3
    enable_taylorseer: bool = True
    taylorseer_order: int = 1
    num_inference_steps: Optional[int] = None


def enable_cache_on_transformer(
    transformer: torch.nn.Module,
    config: CacheDitConfig,
    model_name: str = "transformer",
) -> torch.nn.Module:
    """Enable cache-dit on a transformer module.

    This function enables cache-dit acceleration using the BlockAdapterRegister
    for pre-registered models. Only officially supported models can use this
    function directly.

    Args:
        transformer: The transformer module to enable caching on.
        config: CacheDitConfig with caching parameters.
        model_name: Name of the model for logging purposes.

    Returns:
        The transformer module with cache-dit enabled.

    Raises:
        ImportError: If cache-dit is not installed.
        ValueError: If num_inference_steps is not provided or model is not supported.
    """
    if not config.enabled:
        return transformer

    if not CACHE_DIT_AVAILABLE:
        raise ImportError(
            "cache-dit is not installed. Please install it with: pip install cache-dit"
        )

    if config.num_inference_steps is None:
        raise ValueError(
            "num_inference_steps is required for transformer-only mode. "
            "Please provide it in CacheDitConfig."
        )

    # Check if the transformer is pre-registered in cache-dit
    if not BlockAdapterRegister.is_supported(transformer):
        transformer_cls_name = transformer.__class__.__name__
        raise ValueError(
            f"{transformer_cls_name} is not officially supported by cache-dit. "
            "Supported cache-dit DiT families include Flux, QwenImage, HunyuanDiT, "
            "HunyuanVideo, Wan, CogVideoX, Mochi, and others. "
            "Please ensure your transformer belongs to one of these families or "
            "define a custom BlockAdapter."
        )

    # Build cache config
    cache_config = DBCacheConfig(
        num_inference_steps=config.num_inference_steps,
        Fn_compute_blocks=config.Fn_compute_blocks,
        Bn_compute_blocks=config.Bn_compute_blocks,
        max_warmup_steps=config.max_warmup_steps,
        residual_diff_threshold=config.residual_diff_threshold,
        max_continuous_cached_steps=config.max_continuous_cached_steps,
    )

    # Build calibrator config if TaylorSeer is enabled
    calibrator_config = None
    if config.enable_taylorseer:
        calibrator_config = TaylorSeerCalibratorConfig(
            taylorseer_order=config.taylorseer_order,
        )

    # Enable cache-dit on the transformer
    logger.info(
        "Enabling cache-dit on %s with config: Fn=%d, Bn=%d, W=%d, R=%.2f, MC=%d, "
        "TaylorSeer=%s (order=%d), steps=%d",
        model_name,
        config.Fn_compute_blocks,
        config.Bn_compute_blocks,
        config.max_warmup_steps,
        config.residual_diff_threshold,
        config.max_continuous_cached_steps,
        config.enable_taylorseer,
        config.taylorseer_order,
        config.num_inference_steps,
    )

    cache_dit.enable_cache(
        transformer,
        cache_config=cache_config,
        calibrator_config=calibrator_config,
    )

    return transformer


def get_cache_summary(transformer: torch.nn.Module) -> dict:
    """Get cache statistics from a cache-dit enabled transformer.

    Args:
        transformer: The transformer module with cache-dit enabled.

    Returns:
        A dictionary containing cache statistics, or empty dict if not available.
    """
    if not CACHE_DIT_AVAILABLE:
        return {}

    try:
        stats_list = cache_dit.summary(transformer)
        if not stats_list:
            return {}

        # Handle both single stats and list of stats
        stats = stats_list[0] if isinstance(stats_list, list) else stats_list

        return {
            "cache_options": getattr(stats, "cache_options", None),
            "cached_steps": getattr(stats, "cached_steps", None),
            "pruned_ratio": getattr(stats, "pruned_ratio", None),
        }
    except Exception as e:
        logger.warning("Failed to get cache-dit summary: %s", e)
        return {}


def set_compile_configs_for_cache_dit():
    """Set torch.compile configurations compatible with cache-dit.

    Call this before torch.compile if cache-dit is enabled.
    """
    if CACHE_DIT_AVAILABLE:
        cache_dit.set_compile_configs()
