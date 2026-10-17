from typing import List, Optional, Tuple

import torch


def build_page_major_qsa_mha_views(
    raw: torch.Tensor,
    *,
    layer_num: int,
    head_num: int,
    head_dim: int,
    v_head_dim: int,
    store_dtype: torch.dtype,
    qsa_index_kv_heads: int,
    qsa_index_head_dim: int,
    qsa_store_dtype: torch.dtype,
    qsa_compress_ratio: int,
    page_size: int,
    num_pages: int,
    anchor_bytes: int = 0,
    page_stride_bytes: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """MHA K/V plus compressed QSA keys in one movable page envelope.

    A page is laid out as ``[L0 K | L0 V | L0 QSA | L1 K | ...]``.  The
    returned QSA tensor is a flat view of the raw allocation; callers translate
    a full-cache page id to the layer-specific QSA page embedded in this layout.
    Keeping the index keys inside the envelope makes Full-page compaction atomic.
    """
    assert page_size % qsa_compress_ratio == 0
    itemsize = store_dtype.itemsize
    qsa_itemsize = qsa_store_dtype.itemsize
    k_row_bytes = head_num * head_dim * itemsize
    v_row_bytes = head_num * v_head_dim * itemsize
    qsa_row_bytes = qsa_index_kv_heads * qsa_index_head_dim * qsa_itemsize
    compressed_page_size = page_size // qsa_compress_ratio
    layer_bytes = (
        page_size * (k_row_bytes + v_row_bytes) + compressed_page_size * qsa_row_bytes
    )
    host_page_bytes = layer_num * layer_bytes
    page_bytes = page_stride_bytes or host_page_bytes
    assert page_bytes >= host_page_bytes
    assert anchor_bytes == 0, (
        "QSA's global compressed view currently requires anchor_bytes=0"
    )
    assert page_bytes % itemsize == 0
    assert page_bytes % qsa_row_bytes == 0

    typed = raw.view(store_dtype)
    stride_page = page_bytes // itemsize
    k_shape = (num_pages, page_size, head_num, head_dim)
    v_shape = (num_pages, page_size, head_num, v_head_dim)
    k_stride = (stride_page, k_row_bytes // itemsize, head_dim, 1)
    v_stride = (stride_page, v_row_bytes // itemsize, v_head_dim, 1)
    k_buffer, v_buffer = [], []
    for layer in range(layer_num):
        k_base = layer * layer_bytes
        v_base = k_base + page_size * k_row_bytes
        k_buffer.append(
            torch.as_strided(
                typed, size=k_shape, stride=k_stride, storage_offset=k_base // itemsize
            )
        )
        v_buffer.append(
            torch.as_strided(
                typed, size=v_shape, stride=v_stride, storage_offset=v_base // itemsize
            )
        )

    # Trim to complete compressed pages so existing QSA kernels can reshape it.
    qsa_page_bytes = compressed_page_size * qsa_row_bytes
    qsa_rows = (raw.numel() * raw.itemsize // qsa_page_bytes) * compressed_page_size
    qsa_flat = (
        raw[: qsa_rows * qsa_row_bytes]
        .view(qsa_store_dtype)
        .view(qsa_rows, qsa_index_kv_heads, qsa_index_head_dim)
    )
    return k_buffer, v_buffer, qsa_flat
