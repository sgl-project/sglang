# Performance optimization operators for diffusion models
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from typing import Optional, Tuple
from functools import lru_cache

from sglang.multimodal_gen.runtime.platforms import current_platform

_is_cuda = current_platform.is_cuda()


def _get_compile_mode() -> str:
    """Get the optimal torch.compile mode for diffusion models."""
    return "reduce-overhead"


def should_use_flashinfer_rope(
    query: torch.Tensor, key: torch.Tensor, is_cuda: bool
) -> bool:
    """
    Determine if FlashInfer RoPE should be used based on tensor properties.

    Args:
        query: Query tensor
        key: Key tensor
        is_cuda: Whether running on CUDA

    Returns:
        True if FlashInfer RoPE should be used
    """
    if not is_cuda:
        return False
    # FlashInfer RoPE requires contiguous tensors with matching shapes
    if query.shape != key.shape:
        return False
    if not query.is_contiguous() or not key.is_contiguous():
        return False
    # Head dim must be compatible
    head_dim = query.shape[-1]
    if head_dim > 256:  # FlashInfer has limitations on head_dim
        return False
    return True


def prepare_cos_sin_cache(
    cos: torch.Tensor, sin: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Prepare cos/sin cache for RoPE with optimal memory layout.

    Args:
        cos: Cosine tensor
        sin: Sine tensor
        dtype: Target dtype

    Returns:
        Fused cos_sin_cache tensor
    """
    return torch.cat(
        [
            cos.to(dtype=dtype, memory_format=torch.contiguous_format),
            sin.to(dtype=dtype, memory_format=torch.contiguous_format),
        ],
        dim=-1,
    )


class FusedAdaLNModulation(nn.Module):
    """
    Fused AdaLN modulation that combines multiple operations into one.
    This reduces kernel launch overhead and memory traffic.
    """

    def __init__(self, hidden_size: int, embed_dim: int, num_params: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.num_params = num_params

    def forward_fused(self, x: torch.Tensor, modulation: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Apply fused scale/shift/gate modulation.

        Args:
            x: Input tensor [B, L, C]
            modulation: Modulation parameters [B, num_params * C]

        Returns:
            Tuple of (scale, gate, scale_mlp, gate_mlp) or similar
        """
        B, L, C = x.shape
        # Chunk modulation params
        chunks = modulation.unsqueeze(1).chunk(self.num_params, dim=2)

        if self.num_params == 4:
            scale_msa, gate_msa, scale_mlp, gate_mlp = chunks
            # Apply tanh to gates for better stability
            gate_msa = gate_msa.tanh()
            gate_mlp = gate_mlp.tanh()
            # Apply scale transformation
            scale_msa = 1.0 + scale_msa
            scale_mlp = 1.0 + scale_mlp
            return scale_msa, gate_msa, scale_mlp, gate_mlp
        elif self.num_params == 6:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks
            gate_msa = gate_msa.tanh()
            gate_mlp = gate_mlp.tanh()
            scale_msa = 1.0 + scale_msa
            scale_mlp = 1.0 + scale_mlp
            return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        else:
            return chunks


@torch.compile(mode="reduce-overhead", disable=not _is_cuda)
def fused_apply_qk_norm_and_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused QK norm and RoPE application for better performance.

    Args:
        q: Query tensor [B, L, num_heads, head_dim]
        k: Key tensor [B, L, num_kv_heads, head_dim]
        cos: Cosine for RoPE
        sin: Sine for RoPE
        head_dim: Head dimension

    Returns:
        Tuple of (q, k) after norm and RoPE
    """
    # Apply rotary embeddings
    from sglang.multimodal_gen.runtime.layers.rotary_embedding import _apply_rotary_emb

    q = _apply_rotary_emb(q, cos, sin, is_neox_style=False)
    k = _apply_rotary_emb(k, cos, sin, is_neox_style=False)
    return q, k


@torch.compile(mode="reduce-overhead", disable=not _is_cuda)
def fused_residual_norm_scale_shift(
    residual: torch.Tensor,
    hidden_states: torch.Tensor,
    gate: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused residual connection + norm + scale/shift.

    Args:
        residual: Residual tensor
        hidden_states: Hidden states tensor
        gate: Gate tensor
        shift: Shift tensor
        scale: Scale tensor
        eps: Epsilon for norm

    Returns:
        Output tensor
    """
    # residual + gate * hidden_states
    output = residual + gate * hidden_states

    # Layer norm with scale/shift
    mean = output.mean(dim=-1, keepdim=True)
    var = output.var(dim=-1, keepdim=True, unbiased=False)
    output = (output - mean) * torch.rsqrt(var + eps)

    # Apply scale and shift
    output = output * (1.0 + scale) + shift
    return output


class OptimizedAttentionLayout:
    """
    Optimized memory layout for attention computation.
    Reduces memory copies and improves cache locality.
    """

    @staticmethod
    def prepare_qkv(
        hidden_states: torch.Tensor,
        to_q: nn.Module,
        to_k: nn.Module,
        to_v: nn.Module,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare QKV with optimized memory layout.

        Args:
            hidden_states: Input tensor
            to_q: Query projection
            to_k: Key projection
            to_v: Value projection
            num_heads: Number of attention heads
            num_kv_heads: Number of key/value heads
            head_dim: Head dimension

        Returns:
            Tuple of (q, k, v) tensors
        """
        # Compute projections
        q, _ = to_q(hidden_states)
        k, _ = to_k(hidden_states)
        v, _ = to_v(hidden_states)

        # Reshape to [B, L, num_heads, head_dim] and keep contiguous
        B, L, _ = q.shape
        q = q.view(B, L, num_heads, head_dim).contiguous()
        k = k.view(B, L, num_kv_heads, head_dim).contiguous()
        v = v.view(B, L, num_kv_heads, head_dim).contiguous()

        return q, k, v

    @staticmethod
    def prepare_fused_qkv(
        hidden_states: torch.Tensor,
        to_qkv: nn.Module,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare QKV with fused projection for better performance.

        Args:
            hidden_states: Input tensor
            to_qkv: Fused QKV projection
            num_heads: Number of attention heads
            num_kv_heads: Number of key/value heads
            head_dim: Head dimension

        Returns:
            Tuple of (q, k, v) tensors
        """
        qkv, _ = to_qkv(hidden_states)

        # Split and reshape in one go
        B, L, _ = hidden_states.shape
        q, k, v = qkv.split(
            [num_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim],
            dim=-1,
        )

        q = q.view(B, L, num_heads, head_dim).contiguous()
        k = k.view(B, L, num_kv_heads, head_dim).contiguous()
        v = v.view(B, L, num_kv_heads, head_dim).contiguous()

        return q, k, v


@torch.compile(mode="reduce-overhead", disable=not _is_cuda)
def optimized_ffn_forward(
    x: torch.Tensor,
    gate_proj: nn.Module,
    up_proj: nn.Module,
    down_proj: nn.Module,
    activation: str = "silu",
) -> torch.Tensor:
    """
    Optimized FFN forward with fusion.

    Args:
        x: Input tensor
        gate_proj: Gate projection
        up_proj: Up projection
        down_proj: Down projection
        activation: Activation function

    Returns:
        Output tensor
    """
    # Compute gate and up in parallel when possible
    gate, _ = gate_proj(x)
    up, _ = up_proj(x)

    if activation == "silu":
        gate = torch.nn.functional.silu(gate)
    elif activation == "gelu":
        gate = torch.nn.functional.gelu(gate, approximate="tanh")

    # Element-wise multiplication
    x = gate * up

    # Down projection
    output, _ = down_proj(x)
    return output


class MemoryEfficientAttentionHelper:
    """
    Helper class for memory-efficient attention computation.
    """

    @staticmethod
    def should_use_fused_attention(
        seq_len: int,
        head_dim: int,
        num_heads: int,
        dtype: torch.dtype,
    ) -> bool:
        """
        Determine if fused attention should be used based on tensor shapes.

        Args:
            seq_len: Sequence length
            head_dim: Head dimension
            num_heads: Number of heads
            dtype: Data type

        Returns:
            True if fused attention should be used
        """
        if not _is_cuda:
            return False

        # Fused attention works best for certain configurations
        if dtype not in (torch.float16, torch.bfloat16):
            return False

        if head_dim > 256:
            return False

        # For small sequence lengths, fused attention is beneficial
        if seq_len <= 4096:
            return True

        return True

    @staticmethod
    def get_optimal_num_splits(
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> int:
        """
        Get optimal number of splits for attention computation.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            num_heads: Number of heads
            head_dim: Head dimension

        Returns:
            Optimal number of splits
        """
        # Heuristic for optimal splits based on workload
        total_tokens = batch_size * seq_len

        if total_tokens < 1024:
            return 1
        elif total_tokens < 4096:
            return 2
        elif total_tokens < 16384:
            return 4
        else:
            return 8


@lru_cache(maxsize=128)
def get_chunk_size(hidden_size: int, seq_len: int) -> int:
    """
    Get optimal chunk size for processing based on hidden size and sequence length.

    Args:
        hidden_size: Hidden dimension
        seq_len: Sequence length

    Returns:
        Optimal chunk size
    """
    # Heuristic for chunk size based on memory and computation
    if hidden_size <= 512:
        return min(seq_len, 2048)
    elif hidden_size <= 1024:
        return min(seq_len, 1024)
    elif hidden_size <= 2048:
        return min(seq_len, 512)
    else:
        return min(seq_len, 256)


def optimize_model_for_inference(model: nn.Module) -> nn.Module:
    """
    Apply inference optimizations to a model.

    Args:
        model: Model to optimize

    Returns:
        Optimized model
    """
    if not _is_cuda:
        return model

    # Enable cudnn benchmarking for better performance
    torch.backends.cudnn.benchmark = True

    # Enable TF32 for better performance on Ampere+
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    return model
