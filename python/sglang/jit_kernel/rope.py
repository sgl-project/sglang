from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import flashinfer
import torch

from sglang.jit_kernel.utils import cache_once, is_arch_support_pdl, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_apply_rope_pos_ids_cos_sin_cache_module() -> Module:
    flashinfer_dir = pathlib.Path(flashinfer.__file__).parent.resolve()
    assert (
        flashinfer_dir / "data" / "include"
    ).exists(), (
        f"flashinfer headers are missing {str(flashinfer_dir / 'data' / 'include')}"
    )
    flashinfer_include_path = (flashinfer_dir / "data" / "include").resolve()
    return load_jit(
        "apply_rope_pos_ids_cos_sin_cache",
        cuda_files=["elementwise/rope.cuh"],
        cuda_wrappers=[
            (
                "apply_rope_pos_ids_cos_sin_cache",
                "ApplyRopePosIdsCosSinCacheKernel::run",
            )
        ],
        extra_include_paths=[str(flashinfer_include_path)],
    )


# Split the ops because k_buffer/v_buffer are mutated only when provided,
# and torch.custom_op cannot express optional mutates_args reliably
@register_custom_op(
    op_name="apply_rope_pos_ids_cos_sin_cache_with_kv_cache",
    mutates_args=["q", "k", "q_rope", "k_rope", "k_buffer", "v_buffer"],
)
def apply_rope_pos_ids_cos_sin_cache_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    v: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    kv_cache_loc: torch.Tensor,
    interleave: bool = False,
    enable_pdl: bool = False,
) -> None:
    """
    Apply RoPE (Rotary Positional Embedding) with position IDs and cos/sin cache.

    Args:
        q: Input Q tensor of shape [nnz, num_qo_heads, head_dim]
        k: Input K tensor of shape [nnz, num_kv_heads, head_dim]
        q_rope: Output Q tensor with RoPE applied, same shape as q
        k_rope: Output K tensor with RoPE applied, same shape as k
        cos_sin_cache: Cos/sin cache of shape [max_seq_len, rotary_dim]
        pos_ids: Position IDs of shape [nnz]
        interleave: Whether to use interleaved RoPE
        enable_pdl: Enable PDL (Programmable Data Layout)
        v: Optional V tensor for KV caching
        k_buffer: Optional K buffer for KV caching
        v_buffer: Optional V buffer for KV caching
        kv_cache_loc: Optional KV cache location tensor
    """
    module = _jit_apply_rope_pos_ids_cos_sin_cache_module()

    module.apply_rope_pos_ids_cos_sin_cache(
        q,
        k,
        q_rope,
        k_rope,
        cos_sin_cache,
        pos_ids,
        interleave,
        enable_pdl,
        v,
        k_buffer,
        v_buffer,
        kv_cache_loc,
    )


@register_custom_op(
    op_name="apply_rope_pos_ids_cos_sin_cache_without_kv_cache",
    mutates_args=["q", "k", "q_rope", "k_rope"],
)
def apply_rope_pos_ids_cos_sin_cache_without_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    interleave: bool = False,
    enable_pdl: bool = False,
) -> None:
    module = _jit_apply_rope_pos_ids_cos_sin_cache_module()

    module.apply_rope_pos_ids_cos_sin_cache(
        q,
        k,
        q_rope,
        k_rope,
        cos_sin_cache,
        pos_ids,
        interleave,
        enable_pdl,
        None,
        None,
        None,
        None,
    )


# Adepted from
@dataclass
class FusedSetKVBufferArg:
    """
    value : Optional[torch.Tensor]
        Value tensor, shape: ``(nnz, num_v_heads * head_size)``.
    k_buffer : Optional[torch.Tensor]
        Buffer for keys, shape: ``(nnz, num_k_heads * head_size)``.
    v_buffer : Optional[torch.Tensor]
        Buffer for values, shape: ``(nnz, num_v_heads * head_size)``.
    k_scale : Optional[float]
        Scale factor for keys.
    v_scale : Optional[float]
        Scale factor for values.
    cache_loc : Optional[torch.Tensor]
        Cache location tensor, used for indexing kv cache.
    """

    value: torch.Tensor
    k_buffer: torch.Tensor
    v_buffer: torch.Tensor
    k_scale: Optional[float]
    v_scale: Optional[float]
    cache_loc: torch.Tensor


def _view_3d(x, head_size):
    return x.view(x.shape[0], -1, head_size)


def apply_rope_with_cos_sin_cache_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
    fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
    enable_pdl: Optional[bool] = None,
) -> None:
    r"""
    Apply rotary embedding to keys and queries with precomputed cos/sin values.
    This is designed to be compatible with the SGL/vLLM implementation.
    The result is inplace applied to the input tensors.

    Parameters
    ----------
    positions : torch.Tensor
        Position indices, shape: ``(nnz)``.
    query : torch.Tensor
        Query tensor, shape: ``(nnz, num_q_heads * head_size)``.
    key : torch.Tensor
        Key tensor, shape: ``(nnz, num_k_heads * head_size)``.
    cos_sin_cache : torch.Tensor
        Cosine and Sine cache tensor, shape: ``(max_seq_len, rotary_dim)``.
        Cosine is the first half and Sine is the second half on rotary_dim.
    is_neox : bool
        Whether to use Neox style RoPE, default: ``True``.

        * If ``True``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

        * If ``False``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.
    fused_set_kv_buffer_arg : FusedSetKVBufferArg
        Fuse the set-kv-buffer operation into this kernel

    Note
    ----
    The rotary dimension is determined by the cosine cache and sine cache.
    """
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache should be float32")

    if enable_pdl is None:
        # the non-fused branch does not yet support PDL, but after we switch to our impl for that branch it will
        enable_pdl = is_arch_support_pdl() and (fused_set_kv_buffer_arg is not None)

    if (a := fused_set_kv_buffer_arg) is not None:
        assert a.k_scale is None, "k_scale is not yet supported"
        assert a.v_scale is None, "v_scale is not yet supported"
        assert a.cache_loc.dtype == torch.int64, f"{a.cache_loc.dtype=}"

    save_kv_cache = fused_set_kv_buffer_arg is not None
    if save_kv_cache:
        apply_rope_pos_ids_cos_sin_cache_with_kv_cache(
            _view_3d(query, head_size),
            _view_3d(key, head_size),
            _view_3d(query, head_size),
            _view_3d(key, head_size),
            cos_sin_cache,
            positions.long(),
            _view_3d(fused_set_kv_buffer_arg.value, head_size),
            _view_3d(fused_set_kv_buffer_arg.k_buffer, head_size),
            _view_3d(fused_set_kv_buffer_arg.v_buffer, head_size),
            (fused_set_kv_buffer_arg.cache_loc),
            (not is_neox),
            enable_pdl,
        )
    else:
        apply_rope_pos_ids_cos_sin_cache_without_kv_cache(
            _view_3d(query, head_size),
            _view_3d(key, head_size),
            _view_3d(query, head_size),
            _view_3d(key, head_size),
            cos_sin_cache,
            positions.long(),
            (not is_neox),
            enable_pdl,
        )
