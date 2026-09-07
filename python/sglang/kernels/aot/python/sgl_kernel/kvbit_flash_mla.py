"""DSV4 INT4 decode wrapper isolated from the upstream FlashMLA ABI."""

from typing import Optional, Tuple

import torch
from sgl_kernel.flash_mla import FlashMLASchedMeta

try:
    from sgl_kernel import kvbit_flashmla_ops  # noqa: F401
except Exception as _e:
    _kvbit_flashmla_import_error = _e
else:
    _kvbit_flashmla_import_error = None

_IMPORT_ERROR = ImportError(
    "Failed to load sgl_kernel.kvbit_flashmla_ops extension. "
    "Ensure CUDA Driver >= 12.4."
)


def kvbit_int4_flash_mla_with_kvcache(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    head_dim_v: int,
    sched_meta: FlashMLASchedMeta,
    softmax_scale: float,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    packed_kcache: torch.Tensor,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    extra_packed_kcache: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the 368-byte signed INT4+G64 FP16-step+H256 MODEL1/H64 kernel."""
    if _kvbit_flashmla_import_error is not None:
        raise _IMPORT_ERROR from _kvbit_flashmla_import_error
    if indices is None:
        raise ValueError("KVBit INT4 FlashMLA requires sparse decode indices")

    out, lse, new_metadata, new_num_splits = (
        torch.ops.sgl_kernel.kvbit_int4_sparse_decode_fwd.default(
            q,
            k_cache,
            indices,
            topk_length,
            attn_sink,
            sched_meta.tile_scheduler_metadata,
            sched_meta.num_splits,
            extra_k_cache,
            extra_indices_in_kvcache,
            extra_topk_length,
            head_dim_v,
            softmax_scale,
            packed_kcache,
            extra_packed_kcache,
        )
    )
    sched_meta.tile_scheduler_metadata = new_metadata
    sched_meta.num_splits = new_num_splits
    return out, lse
