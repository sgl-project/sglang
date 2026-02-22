# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import logging
from typing import TYPE_CHECKING

import torch
import torch.distributed._functional_collectives as ft_c
from torch.distributed.tensor.experimental._attention import _cp_options

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
    get_ulysses_parallel_rank,
    get_ulysses_parallel_world_size,
)
from sglang.srt.utils.common import torch_release

_cp_options.enable_load_balance = False

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
        AttentionImpl,
    )

logger = logging.getLogger(__name__)


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


def _usp_split_for_ulysses(x: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
    """
    Split tensor along dimension for UAA using tensor_split.

    Unlike chunk(), tensor_split correctly handles non-divisible sizes.
    Returns the shard for the current rank.
    """
    rank = get_ulysses_parallel_rank()
    if x.shape[dim] % world_size == 0:
        # Evenly divisible - use simple indexing
        chunk_size = x.shape[dim] // world_size
        start_idx = rank * chunk_size
        return x.narrow(dim, start_idx, chunk_size)
    # Unevenly divisible - use tensor_split
    chunks = torch.tensor_split(x, world_size, dim=dim)
    return chunks[rank]


def _usp_gather_for_ulysses(x: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
    """
    Gather tensors from all ranks along dimension.

    Handles varying sizes per rank (UAA capability).
    Uses pad-then-slice strategy to work with standard all_gather
    which requires equal-sized tensors across ranks.
    """
    ulysses_pg = get_sp_group().ulysses_group
    assert ulysses_pg is not None, "Ulysses process group is not initialized."

    # Ensure input is contiguous - tensor_split/narrow may return non-contiguous views
    x = x.contiguous()

    local_size = x.shape[dim]

    # Collect the actual sizes from each rank
    local_size_tensor = torch.tensor([local_size], dtype=torch.long, device=x.device)
    all_sizes_list = [
        torch.zeros(1, dtype=torch.long, device=x.device) for _ in range(world_size)
    ]
    torch.distributed.all_gather(all_sizes_list, local_size_tensor, group=ulysses_pg)
    all_sizes = [int(s.item()) for s in all_sizes_list]
    max_size = max(all_sizes)

    if all(s == max_size for s in all_sizes):
        # All ranks have the same size - use simple all_gather
        gathered = [torch.zeros_like(x) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, x, group=ulysses_pg)
        return torch.cat(gathered, dim=dim)

    # Pad local tensor to max_size along dim so all ranks have equal size
    pad_size = max_size - local_size
    if pad_size > 0:
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_size
        padding = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        x_padded = torch.cat([x, padding], dim=dim)
    else:
        x_padded = x

    # All ranks now have the same size along dim - safe to all_gather
    gathered = [torch.zeros_like(x_padded) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, x_padded, group=ulysses_pg)

    # Slice each gathered tensor to its actual size and concatenate
    sliced = []
    for i, g in enumerate(gathered):
        sliced.append(g.narrow(dim, 0, all_sizes[i]))

    return torch.cat(sliced, dim=dim)


def _usp_input_all_to_all(
    x: torch.Tensor, head_dim: int = 1, enable_uaa: bool = False
) -> torch.Tensor:
    """
    Perform Ulysses-style input all-to-all over the head dimension.

    Default layout expects heads at dim=1 and sequence at dim=2:
        [b, h, s_local, d] -> [b, h_local, s_global, d]

    If heads are at dim=2 (input is [b, s_local, h, d]), set head_dim=2, and the
    function returns [b, s_global, h_local, d], preserving the original
    head/sequence dim ordering.

    Args:
        x: A 4D tensor with layout [b, *, *, d] where '*' are sequence and heads
        head_dim: Which dimension index corresponds to heads (1 or 2)
        enable_uaa: Enable UAA (Ulysses Anything Attention) for arbitrary head/sequence sizes

    Returns:
        Tensor with the same dim order as input, with heads sharded and sequence gathered.
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, f"x must have 4 dimensions, got {x.ndim}"
    assert head_dim in (1, 2), f"head_dim must be 1 or 2, got {head_dim}"
    seq_dim = 1 if head_dim == 2 else 2

    # Bring to canonical [b, h, s, d]
    if head_dim == 1 and seq_dim == 2:
        x_c = x
    else:
        x_c = x.permute(0, head_dim, seq_dim, 3).contiguous()

    b, h, s, d = x_c.shape

    if not enable_uaa:
        # Original path with strict divisibility
        assert (
            h % world_size == 0
        ), f"h ({h}) must be divisible by world_size ({world_size})"

        # [b, h, s_local, d] -> [h, b, s_local, d]
        x_c = x_c.permute(1, 0, 2, 3).contiguous()
        # all-to-all along h
        x_c = _usp_all_to_all_single(x_c)
        # -> [b, h_local, s, d]
        x_c = (
            x_c.reshape(world_size, h // world_size, b, -1, d)
            .permute(2, 1, 0, 3, 4)
            .reshape(b, h // world_size, -1, d)
        )
    else:
        # UAA path: Split heads locally, then gather sequences
        # [b, h, s_local, d] -> [b, h_local, s_local, d]
        x_c = _usp_split_for_ulysses(x_c, dim=1, world_size=world_size)

        # Now gather sequences from all ranks
        # [b, h_local, s_local, d] -> [b, h_local, s_global, d]
        x_c = _usp_gather_for_ulysses(x_c, dim=2, world_size=world_size)

    if head_dim == 1 and seq_dim == 2:
        return x_c

    # Map back to original ordering, preserving head/seq positions
    new_order = [0, None, None, 3]
    new_order[head_dim] = 1
    new_order[seq_dim] = 2
    return x_c.permute(tuple(new_order)).contiguous()


def _usp_output_all_to_all(
    x: torch.Tensor, head_dim: int = 1, enable_uaa: bool = False
) -> torch.Tensor:
    """
    Perform Ulysses-style output all-to-all over the head dimension (inverse of input).

    Default layout expects heads at dim=1 and sequence at dim=2:
        [b, h_local, s, d] -> [b, h, s_local, d]

    If heads are at dim=2 (input is [b, s_global, h // world_size, d]), set head_dim=2,
    and the function returns [b, s_local, h, d], preserving the original head/sequence
    dim ordering.

    Args:
        x: A 4D tensor with layout [b, *, *, d] where '*' are sequence and heads
        head_dim: Which dimension index corresponds to heads (1 or 2)
        enable_uaa: Enable UAA (Ulysses Anything Attention) for arbitrary head/sequence sizes

    Returns:
        Tensor with the same dim order as input, with heads gathered and sequence sharded.
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, f"x must have 4 dimensions, got {x.ndim}"
    assert head_dim in (1, 2), f"head_dim must be 1 or 2, got {head_dim}"
    seq_dim = 1 if head_dim == 2 else 2

    # Bring to canonical [b, h, s, d]
    if head_dim == 1 and seq_dim == 2:
        x_c = x
    else:
        x_c = x.permute(0, head_dim, seq_dim, 3).contiguous()

    b, h, s, d = x_c.shape

    if not enable_uaa:
        # Original path
        assert (
            s % world_size == 0
        ), f"s ({s}) must be divisible by world_size ({world_size})"

        # [b, h_local, s, d] -> [s, b, h_local, d]
        x_c = x_c.permute(2, 0, 1, 3).contiguous()
        x_c = _usp_all_to_all_single(x_c)
        # -> [b, h, s_local, d]
        x_c = (
            x_c.reshape(world_size, s // world_size, b, -1, d)
            .permute(2, 0, 3, 1, 4)
            .reshape(b, -1, s // world_size, d)
        )
    else:
        # UAA path: Split sequences locally, then gather heads
        # [b, h_local, s_global, d] -> [b, h_local, s_local, d]
        x_c = _usp_split_for_ulysses(x_c, dim=2, world_size=world_size)

        # Now gather heads from all ranks
        # [b, h_local, s_local, d] -> [b, h, s_local, d]
        x_c = _usp_gather_for_ulysses(x_c, dim=1, world_size=world_size)

    if head_dim == 1 and seq_dim == 2:
        return x_c

    # Map back to original ordering, preserving head/seq positions
    new_order = [0, None, None, 3]
    new_order[head_dim] = 1
    new_order[seq_dim] = 2
    return x_c.permute(tuple(new_order)).contiguous()


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
    from torch.distributed.tensor.experimental._attention import (
        _templated_ring_attention,
    )

    ring_pg = get_sp_group().ring_group
    assert ring_pg is not None, "Ring process group is not initialized."

    # Ring attention primitives expect tensors in [B, H, S, D] layout.
    # We permute the inputs here.
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()

    # Create an adapter function that matches the signature expected by
    # _templated_ring_attention. The `attn_impl` already has dropout and
    # causal settings configured during its initialization.

    # Note: Please be aware that Attention Backend and Ring Attention may require different QKV tensor shapes.
    # For example, FlashAttention expects the format to be BSHD.
    def attn_callable_adapter(q, k, v, *args, **kwargs):
        # We ignore the dropout_p and is_causal passed by _templated_ring_attention
        # and rely on the pre-configured attn_impl.
        # The `attn_metadata` is not available here, so we pass None.
        # This is a limitation we must accept when using this experimental API.
        q = torch.permute(q, [0, 2, 1, 3])
        k = torch.permute(k, [0, 2, 1, 3])
        v = torch.permute(v, [0, 2, 1, 3])
        # logger.warning(f"Warning: return_s·oftmax_lse is only supported for FlashAttentionImpl")
        output, softmax_lse, *rest = attn_impl.forward(
            q,
            k,
            v,
            attn_metadata=None,
            return_softmax_lse=True,
        )
        output = torch.permute(output, [0, 2, 1, 3])
        return output, softmax_lse, *rest

    # Starting from torch 2.6.0, _templated_ring_attention expects an integer
    # segment_id for the attention function.
    use_segment_id = torch_release >= (2, 6)

    attn_kwargs = dict(
        op=attn_callable_adapter,
        dropout_p=dropout_p,
        is_causal=is_causal,
        query=query,
        key=key,
        value=value,
        group=ring_pg,  # https://github.com/pytorch/pytorch/blob/c907c778f42ba2fdaf25b733dd25baf9779c6a12/torch/distributed/tensor/experimental/_context_parallel/_attention.py#L309
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
