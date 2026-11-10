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
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    from sglang.srt.layers.radix_attention import RadixAttention


def validate_dcp_backend_hip(prefill_impl: str, decode_impl: str) -> None:
    # ROCm merges top-k with its HIP kernels; only the triton backends localize
    # slots and return the DCP partials.
    if prefill_impl != "triton" or decode_impl != "triton":
        raise ValueError(
            "DSA with --dcp-size > 1 on ROCm needs --dsa-prefill-backend and "
            f"--dsa-decode-backend triton; got {prefill_impl} and {decode_impl}."
        )


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
