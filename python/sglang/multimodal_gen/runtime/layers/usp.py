# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import logging
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
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
    x = x.flatten().contiguous()
    output = torch.empty_like(x)
    # USP calls this collective many times per denoising step and waits
    # immediately, so avoid the extra wrapper overhead of functional collectives.
    torch.distributed.all_to_all_single(output, x, group=ulysses_pg)
    return output.reshape(x_shape)


def _usp_all_to_all_single_varlen(
    x: torch.Tensor,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
) -> torch.Tensor:
    ulysses_pg = get_sp_group().ulysses_group
    assert ulysses_pg is not None, "Ulysses process group is not initialized."
    x = x.flatten().contiguous()
    output = torch.empty(sum(output_split_sizes), dtype=x.dtype, device=x.device)
    dist.all_to_all_single(
        output,
        x,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=ulysses_pg,
    )
    return output


def _ipc_ready_group():
    """The ulysses group when the 2-rank IPC transport is usable, else None."""
    from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a import (
        ipc_a2a_ready,
    )

    group = get_sp_group().ulysses_group
    return group if ipc_a2a_ready(group) else None


def _ipc_varlen_fast(x, seq_lens, head_dim, direction):
    """2-rank IPC path for the varlen A2A pair; None when unavailable."""
    if head_dim != 2:
        return None
    group = _ipc_ready_group()
    if group is None:
        return None
    from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a import (
        IPC_A2A,
    )

    r = IPC_A2A.rank
    off = [0, seq_lens[0]]
    if direction == "input":
        # [b, s_local, h_global, d] -> [b, sum(seq_lens), h_global/2, d]
        b, s_local, h_global, d = x.shape
        half = h_global // 2
        peer_len = seq_lens[1 - r]
        send = x[:, :, (1 - r) * half : (2 - r) * half].contiguous()
        out = x.new_empty(b, seq_lens[0] + seq_lens[1], half, d)
        out.narrow(1, off[r], s_local).copy_(x[:, :, r * half : (r + 1) * half])
        theirs = IPC_A2A.exchange(group, send, (b, peer_len, half, d))
        out.narrow(1, off[1 - r], peer_len).copy_(theirs)
        return out
    # output: [b, s_global, h_local, d] -> [b, seq_lens[r], 2*h_local, d].
    # The staging buffer IS the gathered result: each rank writes its head
    # half of the peer's slot directly; no intermediate copy.
    b, s_global, h_local, d = x.shape
    my_len = seq_lens[r]
    peer_len = seq_lens[1 - r]
    n_out = b * my_len * 2 * h_local * d
    n_peer_out = b * peer_len * 2 * h_local * d
    local, peer = IPC_A2A.get_staging(n_out, n_peer_out, x.dtype, group)
    slot = IPC_A2A.next_slot()
    pst = peer[slot].narrow(0, 0, n_peer_out).view(b, peer_len, 2 * h_local, d)
    pst[:, :, r * h_local : (r + 1) * h_local].copy_(
        x.narrow(1, off[1 - r], peer_len), non_blocking=True
    )
    IPC_A2A.signal()
    out = local[slot].narrow(0, 0, n_out).view(b, my_len, 2 * h_local, d)
    out[:, :, r * h_local : (r + 1) * h_local].copy_(x.narrow(1, off[r], my_len))
    IPC_A2A.wait()
    return out


def _ipc_input_a2a_qkv(q, k, v):
    """The three input A2As of one attention as a single IPC exchange;
    None when unavailable."""
    if get_ulysses_parallel_world_size() != 2:
        return None
    group = _ipc_ready_group()
    if group is None:
        return None
    from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a import (
        IPC_A2A,
    )

    b, s_local, h_global, d = q.shape
    half = h_global // 2
    r = IPC_A2A.rank
    n = b * s_local * half * d
    local, peer = IPC_A2A.get_staging(3 * n, 3 * n, q.dtype, group)
    slot = IPC_A2A.next_slot()
    outs = []
    for i, t in enumerate((q, k, v)):
        send = t[:, :, (1 - r) * half : (2 - r) * half].contiguous()
        peer[slot].narrow(0, i * n, n).copy_(send.view(-1), non_blocking=True)
        out = t.new_empty(b, 2 * s_local, half, d)
        out.narrow(1, r * s_local, s_local).copy_(t[:, :, r * half : (r + 1) * half])
        outs.append(out)
    IPC_A2A.signal_and_wait()
    for i, out in enumerate(outs):
        theirs = local[slot].narrow(0, i * n, n).view(b, s_local, half, d)
        out.narrow(1, (1 - r) * s_local, s_local).copy_(theirs)
    return tuple(outs)


def _ipc_input_a2a_qkv_segmented(txt_q, img_q, txt_k, img_k, txt_v, img_v, local_pad):
    """Joint-attention input A2A that reads the (txt, img) pair directly in
    join_seqs layout [txt_real | img | txt_pad], skipping the per-projection
    joint cats. The staging buffer IS the gathered q/k/v: each rank writes its
    own sequence span of the peer's slot; returns (q, k, v) staging views or
    None when unavailable."""
    if get_ulysses_parallel_world_size() != 2:
        return None
    group = _ipc_ready_group()
    if group is None:
        return None
    from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a import (
        IPC_A2A,
    )

    b, txt_len, h_global, d = txt_q.shape
    img_len = img_q.shape[1]
    half = h_global // 2
    r = IPC_A2A.rank
    L = txt_len + img_len
    real = txt_len - local_pad
    n = b * 2 * L * half * d
    local, peer = IPC_A2A.get_staging(3 * n, 3 * n, txt_q.dtype, group)
    slot = IPC_A2A.next_slot()
    ph = slice((1 - r) * half, (2 - r) * half)
    lh = slice(r * half, (r + 1) * half)
    base = r * L
    outs = []
    peer_dsts, peer_srcs = [], []
    loc_dsts, loc_srcs = [], []
    for i, (txt, img) in enumerate(((txt_q, img_q), (txt_k, img_k), (txt_v, img_v))):
        pst = peer[slot].narrow(0, i * n, n).view(b, 2 * L, half, d)
        pspan = pst[:, base : base + L]
        peer_dsts += [pspan[:, 0:real], pspan[:, real : real + img_len]]
        peer_srcs += [txt[:, 0:real, ph], img[:, :, ph]]
        if local_pad:
            peer_dsts.append(pspan[:, real + img_len :])
            peer_srcs.append(txt[:, real:, ph])
        out = local[slot].narrow(0, i * n, n).view(b, 2 * L, half, d)
        lspan = out[:, base : base + L]
        loc_dsts += [lspan[:, 0:real], lspan[:, real : real + img_len]]
        loc_srcs += [txt[:, 0:real, lh], img[:, :, lh]]
        if local_pad:
            loc_dsts.append(lspan[:, real + img_len :])
            loc_srcs.append(txt[:, real:, lh])
        outs.append(out)
    torch._foreach_copy_(peer_dsts, peer_srcs)
    # signal as soon as the peer's data is in flight; our local-half writes
    # overlap with the peer's wait
    IPC_A2A.signal()
    torch._foreach_copy_(loc_dsts, loc_srcs)
    IPC_A2A.wait()
    return tuple(outs)


def _usp_input_all_to_all(x: torch.Tensor, head_dim: int = 1) -> torch.Tensor:
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

    Returns:
        Tensor with the same dim order as input, with heads sharded and sequence gathered.
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    if world_size == 2 and head_dim == 2:
        fast = _ipc_varlen_fast(x, [x.shape[1], x.shape[1]], 2, "input")
        if fast is not None:
            return fast

    assert x.ndim == 4, f"x must have 4 dimensions, got {x.ndim}"
    assert head_dim in (1, 2), f"head_dim must be 1 or 2, got {head_dim}"

    # Move the dimension to be split (h_global) to dim 0 for all_to_all_single
    if head_dim == 1:
        b, h_global, s_local, d = x.shape
        # Shape transition: [b, h_global, s_local, d] -> [h_global, b, s_local, d]
        permute_order = (1, 0, 2, 3)
    else:  # head_dim == 2
        b, s_local, h_global, d = x.shape
        # Shape transition: [b, s_local, h_global, d] -> [h_global, b, s_local, d]
        permute_order = (2, 0, 1, 3)

    assert (
        h_global % world_size == 0
    ), f"h_global ({h_global}) must be divisible by world_size ({world_size})"

    h_local, s_global = h_global // world_size, s_local * world_size

    x = x.permute(permute_order).contiguous()
    x = _usp_all_to_all_single(x)
    x = x.reshape(world_size, h_local, b, s_local, d)

    # Reorder dims to place 'world_size' adjacent to 's_local' to merge them into 's_global'
    if head_dim == 1:
        # Shape transition: [world_size, h_local, b, s_local, d] -> [b, h_local, world_size, s_local, d]
        x = x.permute(2, 1, 0, 3, 4).contiguous().reshape(b, h_local, s_global, d)
    else:  # head_dim == 2
        # Shape transition: [world_size, h_local, b, s_local, d] -> [b, world_size, s_local, h_local, d]
        x = x.permute(2, 0, 3, 1, 4).contiguous().reshape(b, s_global, h_local, d)

    return x


def _usp_input_all_to_all_varlen(
    x: torch.Tensor, seq_lens: list[int], head_dim: int = 1
) -> torch.Tensor:
    """
    Perform Ulysses-style input all-to-all over the head dimension with variable
    local sequence lengths.

    Default layout expects heads at dim=1 and sequence at dim=2:
        [b, h, s_local, d] -> [b, h_local, s_global, d]

    If heads are at dim=2 (input is [b, s_local, h, d]), set head_dim=2, and the
    function returns [b, s_global, h_local, d], preserving the original
    head/sequence dim ordering.

    Args:
        x: A 4D tensor with layout [b, *, *, d] where '*' are sequence and heads
        seq_lens: Local sequence lengths for each rank in the Ulysses group
        head_dim: Which dimension index corresponds to heads (1 or 2)

    Returns:
        Tensor with the same dim order as input, with heads sharded and sequence gathered.
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    if world_size == 2:
        fast = _ipc_varlen_fast(x, seq_lens, head_dim, "input")
        if fast is not None:
            return fast

    assert x.ndim == 4, f"x must have 4 dimensions, got {x.ndim}"
    assert head_dim in (1, 2), f"head_dim must be 1 or 2, got {head_dim}"
    assert (
        len(seq_lens) == world_size
    ), f"seq_lens must have length {world_size}, got {len(seq_lens)}"

    rank = get_ulysses_parallel_rank()

    # Move the dimension to be split (h_global) to dim 0 for all_to_all_single
    if head_dim == 1:
        b, h_global, s_local, d = x.shape
        # Shape transition: [b, h_global, s_local, d] -> [h_global, b, s_local, d]
        permute_order = (1, 0, 2, 3)
    else:  # head_dim == 2
        b, s_local, h_global, d = x.shape
        # Shape transition: [b, s_local, h_global, d] -> [h_global, b, s_local, d]
        permute_order = (2, 0, 1, 3)

    assert (
        s_local == seq_lens[rank]
    ), f"s_local ({s_local}) must equal seq_lens[{rank}] ({seq_lens[rank]})"
    assert (
        h_global % world_size == 0
    ), f"h_global ({h_global}) must be divisible by world_size ({world_size})"

    h_local = h_global // world_size

    x = x.permute(permute_order).contiguous()
    x = x.reshape(world_size, h_local, b, s_local, d)
    input_split_sizes = [h_local * b * s_local * d] * world_size
    output_split_sizes = [h_local * b * seq_len * d for seq_len in seq_lens]
    x = _usp_all_to_all_single_varlen(x, output_split_sizes, input_split_sizes)

    chunks = []
    offset = 0
    for seq_len, split_size in zip(seq_lens, output_split_sizes):
        chunk = x[offset : offset + split_size].reshape(h_local, b, seq_len, d)
        chunks.append(chunk)
        offset += split_size
    x = torch.cat(chunks, dim=2)

    if head_dim == 1:
        # Shape transition: [h_local, b, s_global, d] -> [b, h_local, s_global, d]
        x = x.permute(1, 0, 2, 3).contiguous()
    else:  # head_dim == 2
        # Shape transition: [h_local, b, s_global, d] -> [b, s_global, h_local, d]
        x = x.permute(1, 2, 0, 3).contiguous()

    return x


def _usp_output_all_to_all(x: torch.Tensor, head_dim: int = 1) -> torch.Tensor:
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

    Returns:
        Tensor with the same dim order as input, with heads gathered and sequence sharded.
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    if world_size == 2 and head_dim == 2 and x.shape[1] % 2 == 0:
        half_len = x.shape[1] // 2
        fast = _ipc_varlen_fast(x, [half_len, half_len], 2, "output")
        if fast is not None:
            return fast

    assert x.ndim == 4, f"x must have 4 dimensions, got {x.ndim}"
    assert head_dim in (1, 2), f"head_dim must be 1 or 2, got {head_dim}"

    # Move the dimension to be split (s_global) to dim 0 for all_to_all_single
    if head_dim == 1:
        b, h_local, s_global, d = x.shape
        # Shape transition: [b, h_local, s_global, d] -> [s_global, b, h_local, d]
        permute_order = (2, 0, 1, 3)
    else:  # head_dim == 2
        b, s_global, h_local, d = x.shape
        # Shape transition: [b, s_global, h_local, d] -> [s_global, b, h_local, d]
        permute_order = (1, 0, 2, 3)

    assert (
        s_global % world_size == 0
    ), f"s_global ({s_global}) must be divisible by world_size ({world_size})"

    s_local, h_global = s_global // world_size, h_local * world_size

    x = x.permute(permute_order).contiguous()
    x = _usp_all_to_all_single(x)
    x = x.reshape(world_size, s_local, b, h_local, d)

    # Reorder dims to place 'world_size' adjacent to 'h_local' to merge them into 'h_global'
    if head_dim == 1:
        # Shape transition: [world_size, s_local, b, h_local, d] -> [b, world_size, h_local, s_local, d]
        x = x.permute(2, 0, 3, 1, 4).contiguous().reshape(b, h_global, s_local, d)
    else:  # head_dim == 2
        # Shape transition: [world_size, s_local, b, h_local, d] -> [b, s_local, world_size, h_local, d]
        x = x.permute(2, 1, 0, 3, 4).contiguous().reshape(b, s_local, h_global, d)

    return x


def _usp_output_all_to_all_varlen(
    x: torch.Tensor, seq_lens: list[int], head_dim: int = 1
) -> torch.Tensor:
    """
    Perform Ulysses-style output all-to-all over the head dimension (inverse of input)
    with variable local sequence lengths.

    Default layout expects heads at dim=1 and sequence at dim=2:
        [b, h_local, s, d] -> [b, h, s_local, d]

    If heads are at dim=2 (input is [b, s_global, h // world_size, d]), set head_dim=2,
    and the function returns [b, s_local, h, d], preserving the original head/sequence
    dim ordering.

    Args:
        x: A 4D tensor with layout [b, *, *, d] where '*' are sequence and heads
        seq_lens: Local sequence lengths for each rank in the Ulysses group
        head_dim: Which dimension index corresponds to heads (1 or 2)

    Returns:
        Tensor with the same dim order as input, with heads gathered and sequence sharded.
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    if world_size == 2:
        fast = _ipc_varlen_fast(x, seq_lens, head_dim, "output")
        if fast is not None:
            return fast

    assert x.ndim == 4, f"x must have 4 dimensions, got {x.ndim}"
    assert head_dim in (1, 2), f"head_dim must be 1 or 2, got {head_dim}"
    assert (
        len(seq_lens) == world_size
    ), f"seq_lens must have length {world_size}, got {len(seq_lens)}"

    rank = get_ulysses_parallel_rank()

    # Move the sequence dimension to dim 2 for splitting across seq_lens
    if head_dim == 1:
        b, h_local, s_global, d = x.shape
        # Shape transition: [b, h_local, s_global, d] -> [h_local, b, s_global, d]
        permute_order = (1, 0, 2, 3)
    else:  # head_dim == 2
        b, s_global, h_local, d = x.shape
        # Shape transition: [b, s_global, h_local, d] -> [h_local, b, s_global, d]
        permute_order = (2, 0, 1, 3)

    assert s_global == sum(
        seq_lens
    ), f"s_global ({s_global}) must equal sum(seq_lens) ({sum(seq_lens)})"

    s_local = seq_lens[rank]

    x = x.permute(permute_order).contiguous()
    input_chunks = []
    start = 0
    for seq_len in seq_lens:
        end = start + seq_len
        input_chunks.append(x[:, :, start:end, :].contiguous().reshape(-1))
        start = end
    x = torch.cat(input_chunks, dim=0)
    input_split_sizes = [h_local * b * seq_len * d for seq_len in seq_lens]
    output_split_sizes = [h_local * b * s_local * d] * world_size
    x = _usp_all_to_all_single_varlen(x, output_split_sizes, input_split_sizes)

    chunks = []
    offset = 0
    for split_size in output_split_sizes:
        chunk = x[offset : offset + split_size].reshape(h_local, b, s_local, d)
        chunks.append(chunk)
        offset += split_size
    x = torch.cat(chunks, dim=0)

    if head_dim == 1:
        # Shape transition: [h_global, b, s_local, d] -> [b, h_global, s_local, d]
        x = x.permute(1, 0, 2, 3).contiguous()
    else:  # head_dim == 2
        # Shape transition: [h_global, b, s_local, d] -> [b, s_local, h_global, d]
        x = x.permute(1, 2, 0, 3).contiguous()

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
        # logger.warning(f"Warning: return_softmax_lse is only supported for FlashAttentionImpl")
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
