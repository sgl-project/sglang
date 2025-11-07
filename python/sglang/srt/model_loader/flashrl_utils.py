"""
FlashRL integration for SGLang QuantizedRLModelLoader.

This module provides direct imports from FlashRL's quantization functions.
FlashRL must be installed: pip install flash-rl

Usage:
    from sglang.srt.model_loader.flashrl_utils import get_quantize_fn, load_profile
    
    profile = load_profile("/root/profile.7b.pt")
    quantize_fn = get_quantize_fn("int8")
    
    load_config = LoadConfig(
        load_format="quantized_rl",
        quantize_fn=quantize_fn,
        quant_profile=profile
    )
"""

import logging
import torch
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Import FlashRL quantization functions
try:
    from flash_rl.flash_quantization import get_quantize_fn
except ImportError as e:
    raise ImportError(
        "FlashRL is required for QuantizedRLModelLoader. "
        "Install with: pip install flash-rl"
    ) from e


def load_profile(profile_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load FlashRL quantization profile from file.
    
    Args:
        profile_path: Path to profile.pt file (e.g., "/root/profile.7b.pt")
        
    Returns:
        Profile dictionary
        
    Example:
        >>> profile = load_profile("/root/profile.7b.pt")
    """
    import os
    
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Profile not found: {profile_path}")
    
    logger.info(f"[FlashRL] Loading profile from {profile_path}")
    profile = torch.load(profile_path)
    
    if not isinstance(profile, dict):
        raise RuntimeError(f"Invalid profile format: expected dict, got {type(profile)}")
    
    logger.info(f"[FlashRL] Loaded profile with {len(profile)} parameters")
    return profile
