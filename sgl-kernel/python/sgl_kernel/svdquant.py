"""
Python wrappers for SVDQuant kernels.

These kernels are ported from the Nunchaku project for quantized diffusion model inference.
"""

import math
from typing import List, Optional, Tuple

import torch


def ceil_divide(a: int, b: int) -> int:
    """Compute ceiling division."""
    return (a + b - 1) // b


def svdq_gemv_awq(
    in_feats: torch.Tensor,
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
    m: int,
    n: int,
    k: int,
    group_size: int = 64,
) -> torch.Tensor:
    """
    Performs quantized GEMV using the AWQ W4A16 format.

    Parameters
    ----------
    in_feats : torch.Tensor, shape (k,) or (m, k), dtype float16 or bfloat16
        Input feature vector or batch of vectors.
    kernel : torch.Tensor, shape (n // 4, k // 2), dtype int32
        Packed quantized weight matrix.
    scaling_factors : torch.Tensor, shape (k // group_size, n), dtype float16 or bfloat16
        Per-group scaling factors.
    zeros : torch.Tensor, shape (k // group_size, n), dtype float16 or bfloat16
        Per-group zero points.
    m : int
        Batch size (number of input vectors).
    n : int
        Output feature dimension.
    k : int
        Input feature dimension.
    group_size : int, optional
        Number of input channels per quantization group. Default is 64.

    Returns
    -------
    torch.Tensor, shape (m, n), dtype float16 or bfloat16
        Output tensor.

    Notes
    -----
    Notations:

    - m: batch size
    - n: output features
    - k: input features
    - group_size: quantization group size
    """
    return torch.ops.sgl_kernel.svdq_gemv_awq.default(
        in_feats, kernel, scaling_factors, zeros, m, n, k, group_size
    )


def svdq_gemm_w4a4(
    act: torch.Tensor,
    wgt: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    qout: Optional[torch.Tensor] = None,
    ascales: Optional[torch.Tensor] = None,
    wscales: Optional[torch.Tensor] = None,
    oscales: Optional[torch.Tensor] = None,
    poolout: Optional[torch.Tensor] = None,
    lora_act_in: Optional[torch.Tensor] = None,
    lora_up: Optional[torch.Tensor] = None,
    lora_down: Optional[torch.Tensor] = None,
    lora_act_out: Optional[torch.Tensor] = None,
    norm_q: Optional[torch.Tensor] = None,
    norm_k: Optional[torch.Tensor] = None,
    rotary_emb: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    smooth_factor: Optional[torch.Tensor] = None,
    act_unsigned: bool = False,
    lora_scales: Optional[List[float]] = None,
    fuse_silu: bool = False,
    fp4: bool = False,
    alpha: Optional[float] = 1.0,
    wcscales: Optional[torch.Tensor] = None,
    out_q: Optional[torch.Tensor] = None,
    out_k: Optional[torch.Tensor] = None,
    out_v: Optional[torch.Tensor] = None,
    attn_tokens: int = 0,
) -> None:
    """
    Quantized GEMM using SVDQuant W4A4 CUDA kernel, with support for LoRA, rotary embeddings,
    normalization, and fused activations.

    Parameters
    ----------
    act : torch.Tensor, shape (M, K // 2), dtype int8
        Packed input activations.
    wgt : torch.Tensor, shape (N, K // 2), dtype int8
        Packed quantized weights.
    out : torch.Tensor or None, shape (M, N), dtype float16 or bfloat16, optional
        Output tensor for the linear layer.
    qout : torch.Tensor or None, shape (M, N // 2), dtype int8, optional
        Packed quantized input for the next layer.
    ascales : torch.Tensor or None, shape (K // G, M), optional
        Activation scales.
    wscales : torch.Tensor or None, shape (K // G, N), optional
        Weight scales.
    oscales : torch.Tensor or None, shape (N // G, M), optional
        Output scales.
    poolout : torch.Tensor or None, optional
        Reserved for future use.
    lora_act_in : torch.Tensor or None, shape (M, R), dtype float32, optional
        LoRA down-projection activations.
    lora_up : torch.Tensor or None, shape (N, R), dtype float16 or bfloat16, optional
        Packed LoRA up-projection weights.
    lora_down : torch.Tensor or None, shape (N, R), dtype float16 or bfloat16, optional
        Packed LoRA down-projection weights for the next layer.
    lora_act_out : torch.Tensor or None, shape (M, R), dtype float32, optional
        Output for LoRA down-projection in the next layer.
    norm_q : torch.Tensor or None, shape (HEAD_DIM,), optional
        Query RMS normalization.
    norm_k : torch.Tensor or None, shape (HEAD_DIM,), optional
        Key RMS normalization.
    rotary_emb : torch.Tensor or None, shape (M, HEAD_DIM // 2, 2, 2), dtype float32, optional
        Packed rotary embeddings.
    bias : torch.Tensor or None, shape (N,), optional
        Bias tensor.
    smooth_factor : torch.Tensor or None, shape (N,), optional
        Smoothing factor for quantization in the next layer.
    act_unsigned : bool, default=False
        If True, activations are unsigned.
    lora_scales : list of float or None, optional
        Per-group LoRA scaling factors (16 channels per group). Defaults to 1.0 per group.
    fuse_silu : bool, default=False
        If True, fuse SiLU activation.
    fp4 : bool, default=False
        If True, use 4-bit floating point quantization (NVFP4).
    alpha : float or None, default=1.0
        Per-tensor scaling factor for NVFP4.
    wcscales : torch.Tensor or None, shape (N,), dtype float8_e4m3fn, optional
        Per-channel scaling for NVFP4.
    out_q : torch.Tensor or None, optional
        Packed quantized Q for attention.
    out_k : torch.Tensor or None, optional
        Packed quantized K for attention.
    out_v : torch.Tensor or None, optional
        Packed quantized V for attention.
    attn_tokens : int, default=0
        Number of attention tokens.

    Returns
    -------
    None
        Results are written in-place to the provided output tensors.
    """
    if lora_scales is None and lora_up is not None:
        rank = lora_up.shape[1]
        lora_scales = [1.0] * math.ceil(rank / 16)
    elif lora_scales is None:
        lora_scales = []

    if alpha is None:
        alpha = 1.0

    torch.ops.sgl_kernel.svdq_gemm_w4a4.default(
        act,
        wgt,
        out,
        qout,
        ascales,
        wscales,
        oscales,
        poolout,
        lora_act_in,
        lora_up,
        lora_down,
        lora_act_out,
        norm_q,
        norm_k,
        rotary_emb,
        bias,
        smooth_factor,
        act_unsigned,
        lora_scales,
        fuse_silu,
        fp4,
        alpha,
        wcscales,
        out_q,
        out_k,
        out_v,
        attn_tokens,
    )


def svdq_quantize_w4a4_act_fuse_lora(
    input: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    oscales: Optional[torch.Tensor] = None,
    lora_down: Optional[torch.Tensor] = None,
    lora_act_out: Optional[torch.Tensor] = None,
    smooth: Optional[torch.Tensor] = None,
    fuse_glu: bool = False,
    fp4: bool = False,
    pad_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantizes activations and computes LoRA down-projection using SVDQuant W4A4 CUDA kernel.

    Parameters
    ----------
    input : torch.Tensor, shape (M, K), dtype bfloat16/float16
        Input activations.
    output : torch.Tensor or None, shape (M_pad, K // 2), dtype uint8, optional
        Packed output tensor for quantized activations. Allocated if None.
    oscales : torch.Tensor or None, optional
        Output scales tensor. Allocated if None.
    lora_down : torch.Tensor or None, shape (K, R), dtype bfloat16/float16, optional
        Packed LoRA down-projection weights.
    lora_act_out : torch.Tensor or None, shape (M_pad, R), dtype float32, optional
        Packed output tensor for LoRA activations. Allocated if None.
    smooth : torch.Tensor or None, optional
        Smoothing factor for quantization.
    fuse_glu : bool, default=False
        If True, fuse GLU activation.
    fp4 : bool, default=False
        If True, use NVFP4 quantization; else INT4.
    pad_size : int, default=256
        Pad batch size to a multiple of this value for efficient CUDA execution.

    Returns
    -------
    output : torch.Tensor, shape (M_pad, K // 2), dtype uint8
        Packed quantized activations.
    oscales : torch.Tensor
        Output scales.
    lora_act_out : torch.Tensor, shape (M_pad, R), dtype float32
        Packed LoRA activation output.

    Notes
    -----
    Notations:

    - M: batch size
    - K: input channels
    - R: LoRA rank
    - G: group size (64 for INT4, 16 for NVFP4)
    - M_pad: padded batch size = ceil(M / pad_size) * pad_size
    """
    return torch.ops.sgl_kernel.svdq_quantize_w4a4_act_fuse_lora.default(
        input, output, oscales, lora_down, lora_act_out, smooth, fuse_glu, fp4, pad_size
    )
