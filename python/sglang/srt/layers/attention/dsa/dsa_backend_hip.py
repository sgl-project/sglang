"""Kimi-K3.1 ROCm DSA partials over this rank's fp8_ds_mla shard."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.attention.dsa.sparse_mla_decode_gfx950 import (
    sparse_mla_decode_gfx950,
    supports_sparse_mla_decode_gfx950,
)
from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx95_supported

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsa_backend import (
        DeepseekSparseAttnBackend,
        DSAMetadata,
    )
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@functools.cache
def _gfx950_decode_enabled() -> bool:
    return is_gfx95_supported() and envs.SGLANG_ENABLE_ROCM_DSA_OPUS_DECODE.get()


def forward_triton_dcp_partial(
    backend: DeepseekSparseAttnBackend,
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    layer: RadixAttention,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A decode or target-verify layer over this rank's KV shard: (o [T, H, D],
    natural-log lse [T, H]) for the merge, as _forward_flashmla_kv's
    dcp_partial returns."""
    # the indexer's local slots (dcp_local_topk) and their counts
    topk_length = backend.forward_metadata.dcp_topk_length
    pad = page_table_1.shape[0] - topk_length.shape[0]
    if pad > 0:  # padded rows (DP / attention-TP padding) select nothing
        topk_length = torch.nn.functional.pad(topk_length, (0, pad))
    # Each rank holds about 1/N of the global top-k; bound rows by their share.
    expected_topk = -(-page_table_1.shape[-1] // backend.attn_dcp_size)
    if _gfx950_decode_enabled() and supports_sparse_mla_decode_gfx950(
        q_nope, kv_cache, layer.v_head_dim
    ):
        o, lse = sparse_mla_decode_gfx950(
            q_nope=q_nope,
            q_rope=q_rope,
            kv=kv_cache,
            indices=page_table_1.unsqueeze(1),
            sm_scale=layer.scaling,
            topk_length=topk_length,
            expected_topk=expected_topk,
        )
        return o.squeeze(0), lse

    from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
        triton_sparse_mla_decode_splitk,
    )

    stream_id = int(torch.cuda.current_stream(q_nope.device).cuda_stream)
    workspace = backend._triton_sparse_mla_workspaces.setdefault(stream_id, [])
    o, lse = triton_sparse_mla_decode_splitk(
        q_nope=q_nope,
        q_rope=q_rope,
        kv=kv_cache,
        indices=page_table_1.unsqueeze(1),
        sm_scale=layer.scaling,
        d_v=layer.v_head_dim,
        workspace=workspace,
        return_lse=True,
        topk_length=topk_length,
        expected_topk=expected_topk,
    )
    return o.squeeze(0), lse


def dcp_gather_kv_cache(
    backend: DeepseekSparseAttnBackend,
    kv_cache: torch.Tensor,
    forward_batch: ForwardBatch,
    metadata: DSAMetadata,
) -> torch.Tensor:
    """``DeepseekSparseAttnBackend.dcp_gather_kv_cache``: the send buffer is read
    straight from this rank's cache rows and the output is page-padded in place."""
    from sglang.srt.layers.attention.dsa.dsa_dcp_sharded_hip import (
        dcp_gather_request_rows,
    )

    assert metadata.dcp_local_kv_rows is not None
    rows = kv_cache.view(torch.uint8).view(-1, backend.kv_cache_dim)
    gathered = dcp_gather_request_rows(
        rows,
        forward_batch.seq_lens_cpu,
        row_ids=metadata.dcp_local_kv_rows,
        page_size=backend.physical_page_size,
    )
    return gathered.view(kv_cache.dtype)
