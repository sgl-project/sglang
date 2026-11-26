import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.kvcache.mxfp8_dequant_paged import (
    MXFP8_PAGE,
    mxfp8_scale_words,
)

_HEADS = 24
_KV_HEADS = 8
_HEAD_DIM = 128
_REL_EXTENT = 1024
_QLEN = 8
_MAX_SPLITS = 64
_MAX_BATCH = 8
_MIN_B1_PAGES = 8


def _require_sm12x() -> None:
    if not current_arch().is_sm(12):
        raise RuntimeError("inkling2_rel_verify_mxfp8_sm120 requires NVIDIA SM12x")


@cache_once
def _module():
    _require_sm12x()
    return load_jit(
        "inkling2_rel_verify_mxfp8_sm120",
        cuda_files=["inkling2/rel_verify_mxfp8_sm120.cuh"],
        cuda_wrappers=[
            (
                "rel_verify_full_mxfp8",
                "inkling2_rel_verify_mxfp8_sm120::rel_verify_full_mxfp8",
            ),
        ],
    )


def can_use_rel_verify_mxfp8_sm120(
    q,
    k_cache,
    v_cache,
    page_table,
    seqused_k,
    rel_bias,
    *,
    window_size_left,
    k_scale,
    v_scale,
):
    if window_size_left is not None or not current_arch().is_sm(12):
        return False
    if (
        k_scale is None
        or v_scale is None
        or not q.is_cuda
        or q.ndim != 3
        or k_cache.ndim != 4
        or v_cache.ndim != 4
        or page_table.ndim != 2
        or seqused_k.ndim != 1
        or rel_bias.ndim != 3
        or k_scale.ndim != 5
        or v_scale.ndim != 5
        or q.dtype != torch.bfloat16
        or k_cache.dtype != torch.float8_e4m3fn
        or v_cache.dtype != torch.float8_e4m3fn
        or rel_bias.dtype != torch.bfloat16
        or seqused_k.dtype != torch.int32
        or page_table.dtype != torch.int32
    ):
        return False
    dev = q.device
    if (
        k_cache.device != dev
        or v_cache.device != dev
        or k_scale.device != dev
        or v_scale.device != dev
        or page_table.device != dev
        or seqused_k.device != dev
        or rel_bias.device != dev
    ):
        return False
    tokens, heads_q, head_dim = q.shape
    batch = seqused_k.shape[0]
    if batch <= 0 or batch > _MAX_BATCH or tokens != batch * _QLEN:
        return False
    num_pages, page_size, heads_k, kv_dim = k_cache.shape
    scale_tail = (32, MXFP8_PAGE // 32, _HEAD_DIM // 32)
    return bool(
        heads_q == _HEADS
        and heads_k == _KV_HEADS
        and head_dim == _HEAD_DIM
        and kv_dim == _HEAD_DIM
        and page_size == MXFP8_PAGE
        and v_cache.shape == k_cache.shape
        and k_scale.shape == (num_pages, _KV_HEADS, *scale_tail)
        and v_scale.shape == k_scale.shape
        and k_scale.element_size() == 1
        and v_scale.element_size() == 1
        and k_scale.is_contiguous()
        and v_scale.is_contiguous()
        and k_scale.data_ptr() % 4 == 0
        and v_scale.data_ptr() % 4 == 0
        and rel_bias.shape == (tokens, _HEADS, _REL_EXTENT)
        and page_table.shape[0] >= batch
        and page_table.shape[1] > 0
        and (batch > 1 or page_table.shape[1] >= _MIN_B1_PAGES)
        and q.stride(2) == 1
        and q.stride(0) % 4 == 0
        and q.stride(1) % 4 == 0
        and q.data_ptr() % 8 == 0
        and k_cache.stride(3) == 1
        and v_cache.stride(3) == 1
        and k_cache.stride() == v_cache.stride()
        and k_cache.stride(0) % 16 == 0
        and k_cache.stride(1) % 16 == 0
        and k_cache.stride(2) % 16 == 0
        and k_cache.data_ptr() % 16 == 0
        and v_cache.data_ptr() % 16 == 0
        and rel_bias.stride(2) == 1
        and rel_bias.stride(0) % 8 == 0
        and rel_bias.stride(1) % 8 == 0
        and rel_bias.data_ptr() % 16 == 0
        and page_table.stride(1) == 1
        and seqused_k.is_contiguous()
    )


def rel_verify_mxfp8_partials_sm120(
    q,
    k_cache,
    v_cache,
    page_table,
    seqused_k,
    rel_bias,
    softmax_scale,
    *,
    window_size_left,
    k_scale,
    v_scale,
):
    _require_sm12x()
    if window_size_left is not None:
        raise ValueError(
            "rel_verify_mxfp8_partials_sm120 supports full attention only "
            f"(window_size_left=None), got {window_size_left}"
        )
    tokens, heads_q, head_dim = q.shape
    batch = seqused_k.shape[0]
    k_words = mxfp8_scale_words(k_scale, head_dim)
    v_words = mxfp8_scale_words(v_scale, head_dim)
    if k_words.stride() != v_words.stride():
        raise ValueError(
            f"k_words and v_words strides must match: {k_words.stride()} vs {v_words.stride()}"
        )
    span = page_table.shape[1] * MXFP8_PAGE
    sms = current_arch().num_sms
    max_splits = max(1, min((2 * sms) // max(1, batch * _KV_HEADS), _MAX_SPLITS))
    splits = max(1, min((span + 31) // 32, max_splits))
    o_part = torch.empty(
        (splits, tokens, 1, heads_q, head_dim),
        device=q.device,
        dtype=torch.float32,
    )
    m_part = torch.empty(
        (splits, tokens, 1, heads_q),
        device=q.device,
        dtype=torch.float32,
    )
    l_part = torch.empty(
        (splits, tokens, 1, heads_q),
        device=q.device,
        dtype=torch.float32,
    )
    _module().rel_verify_full_mxfp8(
        q,
        k_cache,
        v_cache,
        k_words,
        v_words,
        page_table,
        seqused_k,
        rel_bias,
        o_part,
        m_part,
        l_part,
        float(softmax_scale),
        splits,
    )
    return o_part, m_part, l_part
