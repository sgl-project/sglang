from typing import TYPE_CHECKING

import torch
import torch.distributed._functional_collectives as ft_c
from packaging.version import parse

from sgl_diffusion.runtime.distributed.parallel_state import (
    get_sp_group,
    get_ulysses_parallel_world_size,
)

if TYPE_CHECKING:
    from sgl_diffusion.runtime.layers.attention.backends.attention_backend import (
        AttentionImpl,
    )


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """
    When tracing the code, the result tensor is not an AsyncCollectiveTensor,
    so we cannot call ``wait()``.
    """
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


def _usp_all_to_all_single(x: torch.Tensor) -> torch.Tensor:
    ulysses_pg = get_sp_group().ulysses_group
    assert ulysses_pg is not None, "Ulysses process group is not initialized."
    x_shape = x.shape
    x = x.flatten()
    x = ft_c.all_to_all_single(
        x, output_split_sizes=None, input_split_sizes=None, group=ulysses_pg
    )
    x = _maybe_wait(x)
    x = x.reshape(x_shape)
    return x


def _usp_input_all_to_all(x: torch.Tensor) -> torch.Tensor:
    """
    [b, h, s // world_size, d] -> [b, h // world_size, s, d]
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, f"x must have 4 dimensions, got {x.ndim}"
    b, h, s, d = x.shape
    assert (
        h % world_size == 0
    ), f"h ({h}) must be divisible by world_size ({world_size})"

    # [b, h, s, d] -> [h, b, s, d]
    x = x.permute(1, 0, 2, 3).contiguous()

    # [h, b, s, d]
    x = _usp_all_to_all_single(x)

    # -> [b, h, s, d]
    x = (
        x.reshape(world_size, h // world_size, b, -1, d)
        .permute(2, 1, 0, 3, 4)
        .reshape(b, h // world_size, -1, d)
    )
    return x


def _usp_output_all_to_all(x: torch.Tensor) -> torch.Tensor:
    """
    [b, h // world_size, s, d] -> [b, h, s // world_size, d]
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, f"x must have 4 dimensions, got {x.ndim}"
    b, h, s, d = x.shape
    assert (
        s % world_size == 0
    ), f"s ({s}) must be divisible by world_size ({world_size})"

    x = x.permute(2, 0, 1, 3).contiguous()
    x = _usp_all_to_all_single(x)
    x = (
        x.reshape(world_size, s // world_size, b, -1, d)
        .permute(2, 0, 3, 1, 4)
        .reshape(b, -1, s // world_size, d)
    )
    return x


def ring_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_impl: "AttentionImpl",
    is_causal: bool = False,
    dropout_p: float = 0.0,
):
    """
    Ring Attention implementation.

    This function implements Ring Attention, a strategy for distributed attention
    computation that reduces peak memory usage. It accepts a generic attention
    implementation (`attn_impl`) which is called by the underlying PyTorch
    distributed attention primitive.

    Args:
        query, key, value: The input tensors for attention.
        attn_impl: An instance of an attention implementation backend
                   (e.g., FlashAttentionImpl) whose `forward` method will be
                   used as the computational kernel.
        is_causal: Whether to apply causal masking.
        dropout_p: Dropout probability.
    """
    # torch.distributed.tensor.experimental._attention is not a public API,
    # but it's what's used in official examples and xDiT.
    from torch.distributed.tensor.experimental._attention import (
        _templated_ring_attention,
    )

    ring_pg = get_sp_group().ring_group
    assert ring_pg is not None, "Ring process group is not initialized."

    # Ring attention primitives expect tensors in [B, H, S, D] layout.
    # The `attn_impl` backends (like FlashAttention) also expect this layout.
    # We permute the inputs here.
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()

    # Create an adapter function that matches the signature expected by
    # _templated_ring_attention. The `attn_impl` already has dropout and
    # causal settings configured during its initialization.
    def attn_callable_adapter(q, k, v, *args, **kwargs):
        # We ignore the dropout_p and is_causal passed by _templated_ring_attention
        # and rely on the pre-configured attn_impl.
        # The `attn_metadata` is not available here, so we pass None.
        # This is a limitation we must accept when using this experimental API.
        output = attn_impl.forward(q, k, v, attn_metadata=None)
        # _templated_ring_attention requires logsumexp as a second return value.
        # We return a dummy tensor as it's not used in the inference forward pass.
        logsumexp = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
        return output, logsumexp

    # Starting from torch 2.6.0, _templated_ring_attention expects an integer
    # segment_id for the attention function.
    use_segment_id = parse(torch.__version__).release >= parse("2.6.0").release

    attn_kwargs = dict(
        mesh=ring_pg,
        op=attn_callable_adapter,
        dropout_p=dropout_p,
        is_causal=is_causal,
        query=query,
        key=key,
        value=value,
    )

    if use_segment_id:
        # For torch >= 2.6, segment_id is required. The value '1' is a placeholder
        # as we are not using complex segmentation features.
        out, *_ = _templated_ring_attention(
            seq_dim=1,  # segment_id
            **attn_kwargs,
        )
    else:
        out, *_ = _templated_ring_attention(
            **attn_kwargs,
        )

    # Permute the output back to [B, S, H, D] layout.
    output = torch.permute(out, [0, 2, 1, 3])
    return output
