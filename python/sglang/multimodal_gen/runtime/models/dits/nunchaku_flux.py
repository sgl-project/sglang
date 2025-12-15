# SPDX-License-Identifier: Apache-2.0
"""
Nunchaku quantized FLUX model wrapper for multimodal_gen.

This module provides a wrapper around Nunchaku's quantized FLUX models
to make them compatible with the multimodal_gen pipeline interface.
"""

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class NunchakuFluxWrapper(nn.Module):
    """
    Wrapper for Nunchaku quantized FLUX models to provide compatible interface.
    
    This wrapper makes Nunchaku's quantized models compatible with the
    multimodal_gen pipeline while preserving quantization benefits.
    
    Args:
        nunchaku_model: The loaded Nunchaku quantized model
        config: Model configuration
    """
    
    def __init__(
        self,
        nunchaku_model: nn.Module,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.model = nunchaku_model
        self.config = config or {}
        
        # Copy attributes from nunchaku model for compatibility
        if hasattr(nunchaku_model, "config"):
            self.config.update(nunchaku_model.config)
        
        # Expose common attributes
        for attr in ["in_channels", "out_channels", "num_layers", "attention_head_dim"]:
            if hasattr(nunchaku_model, attr):
                setattr(self, attr, getattr(nunchaku_model, attr))
        
        logger.info(
            f"Initialized NunchakuFluxWrapper with quantized model. "
            f"Memory footprint reduced by ~3.6x"
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        img_ids: Optional[torch.Tensor] = None,
        txt_ids: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        """
        Forward pass through the quantized model.
        
        Args:
            hidden_states: Latent input tensor
            encoder_hidden_states: Text encoder output
            pooled_projections: Pooled text embeddings
            timestep: Diffusion timestep
            img_ids: Image position IDs
            txt_ids: Text position IDs
            guidance: Guidance scale
            joint_attention_kwargs: Additional attention arguments
            return_dict: Whether to return a dict or tensor
            **kwargs: Additional arguments
            
        Returns:
            Model output (Transformer2DModelOutput or tensor)
        """
        # Prepare arguments for nunchaku model
        # Nunchaku models may have slightly different argument names
        forward_kwargs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
        }
        
        # Add optional arguments
        if img_ids is not None:
            forward_kwargs["img_ids"] = img_ids
        if txt_ids is not None:
            forward_kwargs["txt_ids"] = txt_ids
        if guidance is not None:
            forward_kwargs["guidance"] = guidance
        if joint_attention_kwargs is not None:
            forward_kwargs["joint_attention_kwargs"] = joint_attention_kwargs
        
        # Add any additional kwargs
        forward_kwargs.update(kwargs)
        
        # Call the quantized model
        output = self.model(**forward_kwargs)
        
        # Ensure output format is compatible
        if return_dict:
            if not hasattr(output, "sample"):
                # Wrap tensor output in compatible format
                from diffusers.models.modeling_outputs import Transformer2DModelOutput
                return Transformer2DModelOutput(sample=output)
            return output
        else:
            if hasattr(output, "sample"):
                return output.sample
            return output
    
    def set_attention_processor(self, processor: str):
        """
        Set the attention processor for the quantized model.
        
        Args:
            processor: Processor name ("flashattn2" or "nunchaku-fp16")
        """
        if hasattr(self.model, "set_processor"):
            logger.info(f"Setting Nunchaku attention processor: {processor}")
            self.model.set_processor(processor)
        else:
            logger.warning(
                "Nunchaku model does not support set_processor. "
                "Using default attention processor."
            )
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing (if supported)."""
        if hasattr(self.model, "enable_gradient_checkpointing"):
            self.model.enable_gradient_checkpointing()
        else:
            logger.warning(
                "Gradient checkpointing not supported for quantized models"
            )
    
    def enable_offloading(self):
        """Enable CPU offloading for memory efficiency."""
        if hasattr(self.model, "enable_offloading"):
            logger.info("Enabling CPU offloading for quantized model")
            self.model.enable_offloading()
        else:
            logger.warning("CPU offloading not supported by this Nunchaku model")
    
    @property
    def device(self):
        """Get the device of the model."""
        return next(self.model.parameters()).device
    
    @property
    def dtype(self):
        """Get the dtype of the model."""
        return next(self.model.parameters()).dtype
    
    def to(self, *args, **kwargs):
        """Move model to device/dtype."""
        self.model = self.model.to(*args, **kwargs)
        return self
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self
    
    def train(self, mode: bool = True):
        """Set model to training mode (usually not used for quantized models)."""
        if mode:
            logger.warning(
                "Training mode for quantized models is not recommended. "
                "Quantized models should be used for inference only."
            )
        self.model.train(mode)
        return self


def create_nunchaku_flux_model(
    nunchaku_model: nn.Module,
    config: Optional[Dict[str, Any]] = None,
    enable_fp16_attention: bool = False,
    enable_offloading: bool = False,
) -> NunchakuFluxWrapper:
    """
    Create a wrapped Nunchaku FLUX model.
    
    Args:
        nunchaku_model: Loaded Nunchaku quantized model
        config: Model configuration
        enable_fp16_attention: Use FP16 attention for additional speedup
        enable_offloading: Enable CPU offloading
        
    Returns:
        Wrapped quantized model
    """
    wrapper = NunchakuFluxWrapper(nunchaku_model, config)
    
    # Configure attention processor
    if enable_fp16_attention:
        wrapper.set_attention_processor("nunchaku-fp16")
    
    # Enable offloading if requested
    if enable_offloading:
        wrapper.enable_offloading()
    
    return wrapper

