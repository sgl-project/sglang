# SPDX-License-Identifier: Apache-2.0
"""
Nunchaku quantized model loader for multimodal_gen.

This module provides utilities to load Nunchaku (SVDQuant) quantized models
and integrate them into the multimodal_gen pipeline.
"""

import os
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.layers.quantization.nunchaku_config import (
    NunchakuConfig,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _check_nunchaku_available() -> bool:
    """Check if nunchaku is installed."""
    try:
        import nunchaku
        return True
    except ImportError:
        return False


def _get_nunchaku_model_class(model_type: str):
    """
    Get the appropriate Nunchaku model class based on model type.
    
    Args:
        model_type: Type of model (e.g., "flux", "sana", "qwen_image")
        
    Returns:
        Nunchaku model class
        
    Raises:
        ImportError: If nunchaku is not installed
        ValueError: If model type is not supported
    """
    if not _check_nunchaku_available():
        raise ImportError(
            "nunchaku is not installed. Install it with: "
            "pip install nunchaku"
        )
    
    try:
        from nunchaku import (
            NunchakuFluxTransformer2dModel,
            NunchakuFluxTransformer2DModelV2,
            NunchakuQwenImageTransformer2DModel,
            NunchakuSanaTransformer2DModel,
            NunchakuT5EncoderModel,
        )
    except ImportError as e:
        raise ImportError(
            f"Failed to import nunchaku models: {e}. "
            "Please upgrade nunchaku: pip install --upgrade nunchaku"
        )
    
    model_type_lower = model_type.lower()
    
    # Map model types to Nunchaku classes
    model_class_map = {
        "flux": NunchakuFluxTransformer2dModel,
        "flux_v2": NunchakuFluxTransformer2DModelV2,
        "qwen_image": NunchakuQwenImageTransformer2DModel,
        "qwen-image": NunchakuQwenImageTransformer2DModel,
        "sana": NunchakuSanaTransformer2DModel,
        "t5": NunchakuT5EncoderModel,
    }
    
    if model_type_lower not in model_class_map:
        raise ValueError(
            f"Model type '{model_type}' is not supported by Nunchaku. "
            f"Supported types: {list(model_class_map.keys())}"
        )
    
    return model_class_map[model_type_lower]


def load_nunchaku_model(
    model_type: str,
    quantization_config: NunchakuConfig,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: Union[str, torch.device] = "cuda",
) -> nn.Module:
    """
    Load a Nunchaku quantized model.
    
    Args:
        model_type: Type of model (e.g., "flux", "sana", "qwen_image")
        quantization_config: Nunchaku quantization configuration
        torch_dtype: Data type for non-quantized parameters
        device: Device to load model on
        
    Returns:
        Loaded quantized model
        
    Raises:
        ValueError: If quantized model path is not provided or invalid
        ImportError: If nunchaku is not installed
        FileNotFoundError: If quantized model file doesn't exist
    """
    logger.info(f"Loading Nunchaku quantized model: {model_type}")
    logger.info(f"Quantization config: {quantization_config}")
    
    # Validate quantized model path
    if not quantization_config.quantized_model_path:
        raise ValueError(
            "quantized_model_path must be provided in NunchakuConfig. "
            "You can download pre-quantized models from: "
            "https://huggingface.co/nunchaku-tech"
        )
    
    model_path = quantization_config.quantized_model_path
    
    # Check if it's a local file
    if os.path.exists(model_path):
        logger.info(f"Loading from local path: {model_path}")
    else:
        # Assume it's a HuggingFace Hub path
        logger.info(f"Loading from HuggingFace Hub: {model_path}")
    
    # Get the appropriate model class
    model_class = _get_nunchaku_model_class(model_type)
    
    # Get Nunchaku-specific kwargs for model initialization.
    # We only rely on quantization parameters here; attention processor
    # selection is handled separately at the model level.
    nunchaku_kwargs = {
        "precision": quantization_config.precision,
        "rank": quantization_config.rank,
        "torch_dtype": torch_dtype,
    }
    
    logger.info(f"Loading {model_class.__name__} with kwargs: {nunchaku_kwargs}")
    
    try:
        # Load the quantized model
        # Nunchaku models use from_pretrained to load .safetensors files
        model = model_class.from_pretrained(
            model_path,
            **nunchaku_kwargs
        )
        
        # Move to device
        model = model.to(device)
        
        # Set to eval mode
        model.eval()
        
        logger.info(
            f"Successfully loaded Nunchaku quantized model. "
            f"Precision: {quantization_config.precision}, "
            f"Rank: {quantization_config.rank}"
        )
        
        # Log memory usage if on CUDA
        if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
        
        return model
        
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Quantized model not found at: {model_path}. "
            f"Please download from https://huggingface.co/nunchaku-tech or "
            f"quantize your model using DeepCompressor: "
            f"https://github.com/nunchaku-tech/deepcompressor"
        ) from e
    except Exception as e:
        logger.error(f"Failed to load Nunchaku model: {e}")
        raise


def load_nunchaku_text_encoder(
    quantization_config: NunchakuConfig,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: Union[str, torch.device] = "cuda",
) -> nn.Module:
    """
    Load a Nunchaku quantized text encoder (T5).
    
    Args:
        quantization_config: Nunchaku quantization configuration
        torch_dtype: Data type for non-quantized parameters
        device: Device to load model on
        
    Returns:
        Loaded quantized text encoder
    """
    return load_nunchaku_model(
        model_type="t5",
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device=device,
    )


def should_use_nunchaku(
    server_args,
    component_type: str,
) -> bool:
    """
    Determine if Nunchaku quantization should be used for a component.
    
    Args:
        server_args: Server configuration arguments
        component_type: Type of component (e.g., "transformer", "text_encoder")
        
    Returns:
        True if Nunchaku should be used, False otherwise
    """
    # Check if quantization is enabled
    if not getattr(server_args, "enable_quantization", False):
        return False
    
    # Check if nunchaku is available
    if not _check_nunchaku_available():
        logger.warning(
            "Quantization enabled but nunchaku is not installed. "
            "Falling back to standard loading. "
            "Install with: pip install nunchaku"
        )
        return False
    
    # Check if quantized model path is provided
    quantized_path = getattr(server_args, "quantized_model_path", None)
    if not quantized_path:
        logger.warning(
            "Quantization enabled but no quantized_model_path provided. "
            "Falling back to standard loading."
        )
        return False
    
    return True


def create_nunchaku_config_from_server_args(server_args) -> NunchakuConfig:
    """
    Create NunchakuConfig from server arguments.
    
    Args:
        server_args: Server configuration arguments
        
    Returns:
        NunchakuConfig instance
    """
    return NunchakuConfig(
        precision=getattr(server_args, "quantization_precision", "int4"),
        rank=getattr(server_args, "quantization_rank", 32),
        act_unsigned=getattr(server_args, "quantization_act_unsigned", False),
        quantized_model_path=getattr(server_args, "quantized_model_path", None),
        enable_offloading=getattr(server_args, "quantization_enable_offloading", False),
    )

