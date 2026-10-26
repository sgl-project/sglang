from typing import TYPE_CHECKING

import torch
import torch_mlu_ops

if TYPE_CHECKING:
    from sglang.kernels.ops.attention.rope import FusedSetKVBufferArg


def _transform_cache(
    cos_sin_cache: torch.Tensor, is_neox_style: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform upstream cos_sin_cache into MLU-format cos/sin views.

    Upstream layout (per position):
      [cos_0, ..., cos_{d/2-1}, sin_0, ..., sin_{d/2-1}]  -- shape [max_pos, rotary_dim]

    MLU neox-style (tile each half):
      cos = [cos_0..cos_{d/2-1}, cos_0..cos_{d/2-1}]  -- shape [max_pos, rotary_dim]
      sin = [sin_0..sin_{d/2-1}, sin_0..sin_{d/2-1}]

    MLU non-neox-style (interleave each element):
      cos = [cos_0,cos_0, cos_1,cos_1, ..., cos_{d/2-1},cos_{d/2-1}]
      sin = [sin_0,sin_0, sin_1,sin_1, ..., sin_{d/2-1},sin_{d/2-1}]
    """
    rotary_dim = cos_sin_cache.shape[-1]
    assert rotary_dim % 2 == 0, f"rotary_dim must be even, got {rotary_dim}"

    if is_neox_style:
        cache = cos_sin_cache.unflatten(dim=-1, sizes=(2, -1))
        cache = torch.tile(cache, (1, 1, 2)).flatten(start_dim=1)
    else:
        cache = cos_sin_cache.repeat_interleave(2, dim=-1)

    cos, sin = cache.chunk(2, dim=-1)
    sin = sin.view(-1, rotary_dim)
    cos = cos.view(-1, rotary_dim)
    return cos, sin


def _get_mlu_cos_sin(rope) -> tuple[torch.Tensor, torch.Tensor]:
    """Lazily compute MLU-format cos/sin from upstream cos_sin_cache.

    Refresh when the source cache grows or moves to another device/dtype.
    """
    cache_ptr = (
        rope.cos_sin_cache.data_ptr(),
        rope.cos_sin_cache.shape,
        rope.cos_sin_cache.dtype,
        rope.cos_sin_cache.device,
    )
    cached_ptr = getattr(rope, "_mlu_cache_id", None)
    if cache_ptr != cached_ptr:
        rope._mlu_cos, rope._mlu_sin = _transform_cache(
            rope.cos_sin_cache, rope.is_neox_style
        )
        rope._mlu_cache_id = cache_ptr
        # Retain the source so allocator pointer reuse cannot hide replacement.
        rope._mlu_cache_source = rope.cos_sin_cache
    return rope._mlu_cos, rope._mlu_sin


def apply_rotary(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: torch.Tensor | None = None,
    fused_set_kv_buffer_arg: "FusedSetKVBufferArg | None" = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MLU RoPE forward via torch_mlu_ops.apply_rotary.

    Uses lazy cache transformation from upstream cos_sin_cache — no
    class replacement or __init__ hook needed.
    """
    assert fused_set_kv_buffer_arg is None, (
        "fused_set_kv_buffer_arg is not supported for mlu implementation"
    )

    # The MLU kernel requires matching input/cache dtypes. Preserve FP32
    # cache precision by using the native path when they differ.
    if (query is not None and self.cos_sin_cache.dtype != query.dtype) or (
        key is not None and self.cos_sin_cache.dtype != key.dtype
    ):
        return self.forward_native(positions, query, key, offsets)

    if offsets is not None:
        positions = positions + offsets
    positions = positions.to(torch.int32)

    cos, sin = _get_mlu_cos_sin(self)

    if query is not None:
        query = _apply_rotary_inplace(self, positions, query, cos, sin)
    if key is not None:
        key = _apply_rotary_inplace(self, positions, key, cos, sin)
    return query, key


def _apply_rotary_inplace(
    self, positions: torch.Tensor, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embedding to x via MLU op, writing in-place into the rotary dims.

    Uses a fake batched shape [1, num_tokens, *, head_size] to bypass
    cu_seqlens requirement of torch_mlu_ops.apply_rotary.
    When rotary_dim < head_size, only the first rotary_dim dims are rotated.
    """
    x_shape = x.shape
    num_tokens = positions.shape[-1]
    x = x.view(num_tokens, -1, self.head_size).unsqueeze(0)
    x_rot = x[..., : self.rotary_dim]

    torch_mlu_ops.apply_rotary(
        input=x_rot,
        sin_cache=sin,
        cos_cache=cos,
        position_ids=positions,
        cu_seqlens=None,
        interleaved=not self.is_neox_style,
        discrete=True,
        dynamic_ntk=False,
        max_seqlen=x_rot.shape[1],
        output=x_rot,
    )

    return x.reshape(x_shape)
