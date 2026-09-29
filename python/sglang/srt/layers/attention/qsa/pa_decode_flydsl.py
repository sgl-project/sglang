"""Optional gfx942/gfx950 QSA paged decode using AITER's FlyDSL pa_decode_tile.

The post-gather QSA attention currently runs through ``flash_attn_varlen_func``,
which on ROCm is CK's group-mode ``FmhaFwdKernel``. That is a prefill kernel:
its tile shape puts only ``seqlen_q`` on the MFMA M axis, so a decode row (one
query token per sequence) occupies 1 of 128 rows. ``pa_decode_tile`` flattens
``query_length * query_group_size`` onto M instead -- 12 of 16 rows at Qwen3.8's
Hq=24 / Hkv=2 -- and splits KV across CTAs.

It reads a paged cache rather than the packed varlen buffers, which is why
``qwen_sparse_attn_backend`` feeds it from the same page-aligned gather the
FlashInfer path already uses, sharing that path's static arange block table.

The gather is ``_compact_kv_paged``, which writes the vectorized-5D layout the
kernel reads directly rather than a linear ``[token, head, dim]`` region a later
permute has to rearrange. That means the scratch never needs re-laying:
``paged_cache_views`` below only relabels it, at no cost. The permute it replaced
was as expensive as the attention itself (9.3us against 10.8us at one row, 198us
against 71us at 64).
"""

from functools import lru_cache
from importlib.util import find_spec

import torch

from sglang.srt.environ import envs

# 16 bytes per vector lane; 8 elements for bf16. The kernel's cache layout is
# defined in these units, and both head_dim and block_size must divide by it.
_KV_VECTOR_BYTES = 16


@lru_cache(maxsize=1)
def _aiter_pa_decode_available() -> bool:
    # Check only after device/shape guards; optional packages stay lazy.
    try:
        return (
            find_spec("flydsl") is not None
            and find_spec("aiter.ops.flydsl.pa_decode") is not None
        )
    except ModuleNotFoundError:
        return False


def flydsl_qsa_pa_decode_supported(
    q: torch.Tensor, k_buffer: torch.Tensor, page_size: int
) -> bool:
    """Whether the FlyDSL paged decode can serve this QSA call.

    Ordered so the AITER/FlyDSL imports stay off the NVIDIA and disabled paths.
    """
    if not envs.SGLANG_AITER_QSA_PA_DECODE.get():
        return False
    if torch.version.hip is None or not q.is_cuda:
        return False
    if q.dim() != 3 or k_buffer.dim() != 3:
        return False
    # The gather writes scratch in the query dtype, so an FP8 pool still reaches
    # the kernel as bf16; gate on the query, not on k_buffer.
    if q.dtype is not torch.bfloat16:
        return False
    head_dim = k_buffer.shape[2]
    elem = q.element_size()
    vec = _KV_VECTOR_BYTES // elem
    if head_dim % vec or page_size % vec:
        return False
    if not _aiter_pa_decode_available():
        return False

    from aiter.ops.flydsl.pa_decode import flydsl_pa_decode_supported

    # Probe with the cache shape the backend will actually build, not the pool's.
    probe_key = k_buffer.new_empty(
        (1, k_buffer.shape[1], head_dim // vec, page_size, vec)
    )
    return flydsl_pa_decode_supported(q, probe_key, block_size=page_size)


def paged_cache_views(
    scratch_k: torch.Tensor,
    scratch_v: torch.Tensor,
    num_blocks: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Label the gather scratch with the cache shapes the kernel asserts on.

    key   -> ``[num_blocks, num_kv_heads, head_dim // vec, page_size, vec]``
    value -> ``[num_blocks, num_kv_heads, page_size // vec, head_dim, vec]``

    Views, not copies: ``_compact_kv_paged`` has already written the scratch in
    exactly this order, so nothing moves. The kernel addresses the cache flat and
    only reads the shape to validate it.
    """
    vec = _KV_VECTOR_BYTES // scratch_k.element_size()
    key_cache = scratch_k.view(
        num_blocks, num_kv_heads, head_dim // vec, page_size, vec
    )
    value_cache = scratch_v.view(
        num_blocks, num_kv_heads, page_size // vec, head_dim, vec
    )
    return key_cache, value_cache
