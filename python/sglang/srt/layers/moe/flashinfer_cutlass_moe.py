# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

import logging
from typing import Optional

import torch

# Initialize logger for the module
logger = logging.getLogger(__name__)

# FlashInfer imports
from typing import TYPE_CHECKING
try:
    from flashinfer.fused_moe import cutlass_fused_moe as cutlass_fused_moe
    from flashinfer import fp4_quantize as fp4_quantize
except ImportError:
    if not TYPE_CHECKING:
        cutlass_fused_moe = None
        fp4_quantize = None

has_flashinfer_cutlass_fused_moe = cutlass_fused_moe is not None


def get_device_capability() -> Optional[int]:
    """Get CUDA device capability."""
    try:
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            return device_props.major * 10 + device_props.minor
    except Exception:
        pass
    return None


def is_b200_or_later() -> bool:
    """Check if the current device is B200 (SM 100) or later."""
    capability = get_device_capability()
    return capability is not None and capability >= 100


def _valid_flashinfer_fused_moe(hidden_states: torch.Tensor, w1: torch.Tensor,
                     w2: torch.Tensor) -> bool:
    """
    Check if the given problem size is supported by the FlashInfer CUTLASS 
    fused MoE kernel.
    """
    if not has_flashinfer_cutlass_fused_moe:
        logger.debug("FlashInferExperts disabled: flashinfer_cutlass_fused_moe not available.")
        return False
    return True


def moe_kernel_quantize_input_fp4(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform fp4 quantization on the inputs using flashinfer.
    """
    if fp4_quantize is None:
        raise ImportError("flashinfer is required for fp4 quantization")
    return fp4_quantize(A, A_scale)


class FlashInferCutlassKernels:
    """
    Simplified FlashInfer CUTLASS MoE Kernels implementation for NVFP4 quantization.
    
    This class handles NVFP4 quantized MoE computation using FlashInfer's 
    CUTLASS kernels. It is specifically designed for B200+ GPUs with NVFP4 models.
    """
    
    def __init__(self):
        self.has_nvfp4 = True  # FlashInfer CUTLASS is specifically for NVFP4
        if not has_flashinfer_cutlass_fused_moe:
            raise ImportError("FlashInfer CUTLASS fused MoE is not available")
        if not is_b200_or_later():
            raise RuntimeError("FlashInfer CUTLASS requires B200+ GPU (SM 100+)")
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        a1_scale: torch.Tensor,
        a2_scale: torch.Tensor,
        g1_alphas: torch.Tensor,
        g2_alphas: torch.Tensor,
        inplace: bool = False,
        activation: str = "silu",
        global_num_experts: int = -1,
        ep_rank: Optional[int] = 0,
        ep_size: Optional[int] = 1,
        tp_rank: Optional[int] = 0,
        tp_size: Optional[int] = 1,
        apply_router_weight_on_input: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for FlashInfer CUTLASS MoE with NVFP4 quantization.
        
        Args:
            hidden_states: Input hidden states [batch_size, hidden_dim]
            w1: First expert weight tensor (NVFP4 quantized) [num_experts, intermediate_size, hidden_dim//2]
            w2: Second expert weight tensor (NVFP4 quantized) [num_experts, hidden_dim, intermediate_size//2]
            topk_ids: Top-k expert indices [batch_size, top_k]
            topk_weights: Top-k routing weights [batch_size, top_k]
            w1_scale: Scale for w1 weights [num_experts, intermediate_size, hidden_dim//group_size]
            w2_scale: Scale for w2 weights [num_experts, hidden_dim, intermediate_size//group_size]
            a1_scale: Scale for first activation [num_experts, 2]
            a2_scale: Scale for second activation [num_experts]
            g1_alphas: Alpha parameters for first gate [num_experts, 2]
            g2_alphas: Alpha parameters for second gate [num_experts]
            inplace: Whether to perform computation in-place
            activation: Activation function to use
            global_num_experts: Total number of experts
            ep_rank: Expert parallel rank
            ep_size: Expert parallel size
            tp_rank: Tensor parallel rank
            tp_size: Tensor parallel size
            apply_router_weight_on_input: Whether to apply router weights on input
            
        Returns:
            Output tensor after MoE computation [batch_size, hidden_dim]
        """
        assert activation == "silu", "Only SiLU activation is supported"
        assert self.has_nvfp4, "FlashInfer CUTLASS only supports NVFP4 quantization"
        
        # Check that weights are NVFP4 quantized (uint8)
        assert w1.dtype == torch.uint8, f"w1 must be NVFP4 quantized (torch.uint8), got {w1.dtype}"
        assert w2.dtype == torch.uint8, f"w2 must be NVFP4 quantized (torch.uint8), got {w2.dtype}"
        
        # Input preprocessing - apply router weights if needed
        a1 = hidden_states
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, \
                "apply_router_weight_on_input is only implemented for topk=1"
            a1 = a1 * topk_weights.to(a1.dtype)

        # Quantize input activations to FP4
        a1q, a1q_scale = moe_kernel_quantize_input_fp4(a1, a1_scale)
        
        # Prepare output tensor
        output = a1 if inplace else torch.zeros_like(a1)
        
        # FlashInfer CUTLASS kernel takes scalar global scales
        a1_gs = torch.min(a1_scale)
        a2_gs = torch.min(a2_scale)
        w1_blockscale = w1_scale
        w2_blockscale = w2_scale
        
        quant_scales = [
            a1_gs,
            w1_blockscale.view(torch.int32),
            g1_alphas,
            a2_gs,
            w2_blockscale.view(torch.int32),
            g2_alphas,
        ]
        
        # Call FlashInfer CUTLASS fused MoE kernel
        # This handles both the gate/up projection, SiLU activation, and down projection
        output = cutlass_fused_moe(
            a1q,  # Quantized input activations
            topk_ids.to(torch.int),
            topk_weights,
            w1.view(torch.long),  # NVFP4 quantized gate/up weights
            w2.view(torch.long),  # NVFP4 quantized down weights
            output.dtype,  # Output dtype
            quant_scales=quant_scales,
            input_sf=a1q_scale,
            ep_size=ep_size,
            ep_rank=ep_rank,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )[0]
        
        return output


def flashinfer_cutlass_fused_moe_nvfp4(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor,
    a2_scale: torch.Tensor,
    g1_alphas: torch.Tensor,
    g2_alphas: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    global_num_experts: int = -1,
    ep_rank: Optional[int] = 0,
    ep_size: Optional[int] = 1,
    tp_rank: Optional[int] = 0,
    tp_size: Optional[int] = 1,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """
    Simplified FlashInfer CUTLASS fused MoE with NVFP4 quantization.
    
    This is the main entry point for FlashInfer CUTLASS MoE computation.
    All preprocessing and MoE computation is handled by a single kernel call.
    
    Args:
        hidden_states: Input hidden states
        topk_weights: Top-k routing weights
        topk_ids: Top-k expert indices
        w1: First expert weight tensor (NVFP4 quantized, must be torch.uint8)
        w2: Second expert weight tensor (NVFP4 quantized, must be torch.uint8)
        w1_scale: Scale for w1 weights
        w2_scale: Scale for w2 weights
        a1_scale: Scale for first activation
        a2_scale: Scale for second activation
        g1_alphas: Alpha parameters for first gate
        g2_alphas: Alpha parameters for second gate
        inplace: Whether to perform computation in-place
        activation: Activation function to use
        global_num_experts: Total number of experts
        ep_rank: Expert parallel rank
        ep_size: Expert parallel size
        tp_rank: Tensor parallel rank
        tp_size: Tensor parallel size
        apply_router_weight_on_input: Whether to apply router weights on input
        
    Returns:
        Output tensor after MoE computation
    """
    kernel = FlashInferCutlassKernels()
    
    return kernel(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        inplace=inplace,
        activation=activation,
        global_num_experts=global_num_experts,
        ep_rank=ep_rank,
        ep_size=ep_size,
        tp_rank=tp_rank,
        tp_size=tp_size,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
