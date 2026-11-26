from typing import Optional, Tuple

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import is_gfx95_supported, is_hip

_HIP_BACKENDS = ("tilelang", "triton", "aiter_sparse")

# at this many query tokens and above the aiter sparse kernel runs unsplit (batch-invariant prefill)
_AITER_SPARSE_SINGLE_SPLIT_MIN_TOKENS = 1024


def hip_attn_kv_splits() -> int:
    """Split-KV count for the HIP sparse decode kernels. Deterministic inference pins 4
    splits (a fixed combine order keeps the bits identical at every batch size) and
    everything else uses 0: adaptive splits plus the native 16-head attention for small
    TP4 decode batches."""
    try:
        deterministic = get_exec().deterministic.enable_deterministic_inference
    except ValueError:  # no published exec config (kernel tests, offline tools)
        deterministic = False
    return 4 if deterministic else 0


# the eager warmups before each capture fill this, so captured graphs reuse these tensors
_UNIFORM_INDPTR_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}


def _uniform_indptr(num_tokens: int, width: int, device: str) -> torch.Tensor:
    """Row pointers of the aiter sparse decode kernel: token t reads kv_indices[t*w : (t+1)*w].

    Only tensors built outside capture are cached: an arange recorded into a graph runs only
    when that graph replays, and V4.1 captures two verify graphs per batch size."""
    key = (num_tokens, width, device)
    indptr = _UNIFORM_INDPTR_CACHE.get(key)
    if indptr is not None:
        return indptr
    indptr = torch.arange(
        0, (num_tokens + 1) * width, width, dtype=torch.int32, device=device
    )
    if (
        num_tokens < _AITER_SPARSE_SINGLE_SPLIT_MIN_TOKENS
        and not torch.cuda.is_current_stream_capturing()
    ):
        _UNIFORM_INDPTR_CACHE[key] = indptr
    return indptr


def aiter_sparse_decode_fwd(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    inv_rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    **_,
):
    """aiter's gfx950 gluon sparse decode kernel (pa_decode_sparse) behind the
    flash_mla_with_kvcache shapes. Only -1 entries are skipped, so callers fold
    topk_length into the index lists; inv_rope folds the model's inverse RoPE into the output."""
    if k_cache.shape[-1] in (
        KVLayout.V41.bytes_per_token,
        KVLayout.V41_FP4.bytes_per_token,
    ):
        from sglang.kernels.ops.attention.dsv4.compact_attention_hip import (
            compact_attention_hip,
        )

        b, s, h, d = q.shape
        out = compact_attention_hip(
            q.reshape(b * s, h, d),
            k_cache,
            indices,
            topk_length.reshape(-1),
            attn_sink,
            extra_cache=extra_k_cache,
            extra_indices=extra_indices_in_kvcache,
            extra_lengths=(
                extra_topk_length.reshape(-1) if extra_k_cache is not None else None
            ),
            softmax_scale=softmax_scale,
            inv_rope=inv_rope,
            splits=(
                1
                if b * s >= _AITER_SPARSE_SINGLE_SPLIT_MIN_TOKENS
                else hip_attn_kv_splits() or None
            ),
        )
        return out.view(b, s, h, d), None

    from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse

    from sglang.kernels.ops.attention.sparse_decode_reduce_hip import (
        gfx_sparse_split_reduce,
    )

    b, s, h, d = q.shape
    n = b * s
    q3 = q.reshape(n, h, d)
    assert k_cache.dim() == 4 and k_cache.shape[2] == 1, k_cache.shape
    cache = k_cache.view(torch.uint8).squeeze(2)
    idx = indices.reshape(-1)
    assert idx.dtype == torch.int32 and idx.is_contiguous()
    indptr = _uniform_indptr(n, indices.shape[-1], str(q.device))
    extra_kwargs = {}
    if extra_k_cache is not None:
        assert extra_k_cache.dim() == 4 and extra_k_cache.shape[2] == 1
        extra_idx = extra_indices_in_kvcache.reshape(-1)
        assert extra_idx.dtype == torch.int32 and extra_idx.is_contiguous()
        extra_kwargs = dict(
            extra_cache=extra_k_cache.view(torch.uint8).squeeze(2),
            extra_indices=extra_idx,
            extra_indptr=_uniform_indptr(
                n, extra_indices_in_kvcache.shape[-1], str(q.device)
            ),
        )
    if n >= _AITER_SPARSE_SINGLE_SPLIT_MIN_TOKENS:
        extra_kwargs["kv_splits"] = 1
    elif hip_attn_kv_splits() > 0:
        # a pinned split count keeps the combine order, hence the bits, the same at every batch size
        extra_kwargs["kv_splits"] = hip_attn_kv_splits()
    out = pa_decode_sparse(
        q3,
        cache,
        idx,
        indptr,
        attn_sink,
        softmax_scale,
        skip_reduce=True,
        **extra_kwargs,
    )
    if isinstance(out, tuple):
        # Split-KV partials (acc, m, l): combine them here.
        part_acc, part_m, part_l = out
        out = gfx_sparse_split_reduce(
            part_acc, part_m, part_l, attn_sink, q.dtype, inv_rope=inv_rope
        )
    elif inv_rope is not None:
        _apply_inverse_rope(out, inv_rope)
    return out.view(b, s, h, d), None


def _apply_inverse_rope(
    out: torch.Tensor, inv_rope: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """The model's standalone inverse RoPE on out [n, h, d] (last 64 dims of
    every head), for the paths that did not fold it into their combine."""
    from sglang.kernels.ops.attention.dsv4.norm_rope import fused_rope_inplace

    freqs_real, positions = inv_rope
    n = out.shape[0]
    # freqs_real is view_as_real(freqs_cis).flatten(-2); fused_rope_inplace takes the complex table
    freqs_cis = torch.view_as_complex(freqs_real.view(freqs_real.shape[0], -1, 2))
    fused_rope_inplace(
        out.view(n, -1, out.shape[-1])[..., -freqs_real.shape[-1] :],
        None,
        freqs_cis,
        positions.view(-1)[:n],
        inverse=True,
    )


def resolve_hip_flashmla_backend(backend: Optional[str] = None) -> str:
    """The HIP decode attention kernel name; "auto" (the default) is aiter's
    gluon sparse kernel on gfx950 and the tilelang partial + combine elsewhere."""
    if backend is None:
        backend = envs.SGLANG_HACK_FLASHMLA_BACKEND.get()
    if backend == "auto":
        return "aiter_sparse" if is_gfx95_supported() else "tilelang"
    if backend == "aiter_sparse" and not is_gfx95_supported():
        raise ValueError(
            "SGLANG_HACK_FLASHMLA_BACKEND=aiter_sparse needs gfx950 (aiter's gluon sparse "
            "decode kernel); use auto or tilelang"
        )
    return backend


# kernels taking the real per-rank head count; the tilelang kernel builds only for 64-padded widths
_HIP_BACKENDS_ANY_HEAD_COUNT = frozenset({"aiter_sparse", "triton"})


def hip_attention_needs_head_pad() -> bool:
    """Whether the kernel DeepseekV4HipRadixBackend.forward picks needs the per-rank query
    heads padded to 64 (zero q, zero sink)."""
    return resolve_hip_flashmla_backend() not in _HIP_BACKENDS_ANY_HEAD_COUNT


def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):
    if is_hip():
        # a name outside _HIP_BACKENDS resolves from SGLANG_HACK_FLASHMLA_BACKEND
        backend = resolve_hip_flashmla_backend(
            backend if backend in _HIP_BACKENDS else None
        )

    if backend == "tilelang":
        from sglang.kernels.ops.attention.dsv4.tilelang_fp8_attention import (
            dpsk_v4_fp8_attention_fwd,
        )

        return dpsk_v4_fp8_attention_fwd(**kwargs)

    if backend == "triton":
        from sglang.kernels.ops.attention.dsv4.triton_sparse_decode import (
            triton_fp8_attention_fwd,
        )

        return triton_fp8_attention_fwd(**kwargs)

    if backend == "aiter_sparse":
        return aiter_sparse_decode_fwd(**kwargs)

    raise NotImplementedError(f"unknown backend: {backend!r}")
