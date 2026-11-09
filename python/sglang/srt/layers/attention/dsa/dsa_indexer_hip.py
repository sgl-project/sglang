"""ROCm DSA index-K sharding and shared-request target-verify logits."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch

from sglang.kernels.ops.attention.dsa import (
    aiter_paged_mqa_logits_persistent_schedule,
    aiter_paged_mqa_logits_verify,
)
from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.srt.utils import print_info_once


class _PerForwardMemo:
    """Layers run in increasing layer_id order; a repeated or lower ID, or a new
    key, starts a new forward. Graph capture records the first layer's compute."""

    def __init__(self) -> None:
        self._key: Optional[Tuple[Any, ...]] = None
        self._last_layer_id = -1
        self._value: Optional[torch.Tensor] = None

    def get(
        self,
        key: Tuple[Any, ...],
        layer_id: int,
        compute: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        if key != self._key or layer_id <= self._last_layer_id:
            self._key = key
            self._value = compute()
        self._last_layer_id = layer_id
        return self._value


# Index-K write slots and paged-logits lengths of this DCP rank.
_dcp_local_store_loc = _PerForwardMemo()
_dcp_local_lens = _PerForwardMemo()
# Target verify on the shared-index-K path: each request's full length and page-table row.
_verify_request_lens = _PerForwardMemo()
_verify_request_block_tables = _PerForwardMemo()
# Generic paged-logits path: per-row work table balanced over the real lengths.
_paged_logits_schedule = _PerForwardMemo()


def _localize_store_loc(
    out_loc: torch.Tensor, dcp_size: int, dcp_rank: int
) -> torch.Tensor:
    # Widened slot s lives on rank s % N at local row s // N; -1 skips the write.
    owned = (out_loc >= 0) & (out_loc % dcp_size == dcp_rank)
    return torch.where(owned, out_loc // dcp_size, -1)


def dcp_local_store_loc(
    forward_batch, layer_id: int, out_loc: torch.Tensor, dcp_size: int, dcp_rank: int
) -> torch.Tensor:
    """This rank's index-K rows for the widened write locs, -1 where AITER's store skips."""
    return _dcp_local_store_loc.get(
        key=(id(forward_batch), out_loc.data_ptr(), out_loc.shape),
        layer_id=layer_id,
        compute=lambda: _localize_store_loc(out_loc, dcp_size, dcp_rank),
    )


def dcp_local_lens(
    forward_batch, layer_id: int, lens: torch.Tensor, dcp_size: int, dcp_rank: int
) -> torch.Tensor:
    """This rank's share of each length over its index-K shard (CUDA builds it with
    the DeepGEMM schedule)."""
    return _dcp_local_lens.get(
        key=(id(forward_batch), lens.data_ptr(), lens.shape),
        layer_id=layer_id,
        compute=lambda: get_dcp_lens(lens, dcp_size, dcp_rank).to(lens.dtype),
    )


def paged_logits_persistent_schedule(
    forward_batch, layer_id: int, lens: torch.Tensor, max_seq_len: int
) -> torch.Tensor:
    """The aiter persistent schedule for one query row per entry of lens, shared by the
    forward's layers."""
    return _paged_logits_schedule.get(
        key=(id(forward_batch), lens.data_ptr(), lens.shape, max_seq_len),
        layer_id=layer_id,
        compute=lambda: aiter_paged_mqa_logits_persistent_schedule(lens, max_seq_len),
    )


def verify_shares_index_k(
    forward_batch,
    q_fp8: torch.Tensor,
    block_tables: torch.Tensor,
    page_size: int,
    q_offset: int,
    B: int,
    next_n: int,
    dcp_size: int,
) -> bool:
    # Verify rows come next_n per request with a page-table row each, unpadded.
    return (
        dcp_size == 1
        and forward_batch.forward_mode.is_target_verify()
        and next_n >= 2
        and q_fp8.shape[0] == q_offset == B * next_n
        and page_size == 64
        and block_tables.shape[0] >= q_offset
    )


def verify_paged_mqa_logits(
    forward_batch,
    layer_id: int,
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_seq_len: int,
    q_offset: int,
    B: int,
    next_n: int,
    kv_block_size: int,
) -> torch.Tensor:
    """Target-verify logits with each request's draft rows side by side over its
    index-K, where verify_shares_index_k holds."""
    print_info_once(
        "DSA indexer: target-verify logits run each request's "
        f"{next_n} draft rows side by side over its index-K"
    )
    # A request's last row carries its full length; its page table rows repeat.
    key = (id(forward_batch), q_offset, next_n)
    request_lens = _verify_request_lens.get(
        key=key,
        layer_id=layer_id,
        compute=lambda: lens.reshape(-1)[next_n - 1 : q_offset : next_n].contiguous(),
    )
    request_block_tables = _verify_request_block_tables.get(
        key=key,
        layer_id=layer_id,
        compute=lambda: block_tables[:q_offset:next_n].contiguous(),
    )
    return aiter_paged_mqa_logits_verify(
        q_fp8,
        kv_cache_fp8,
        weights,
        request_lens,
        request_block_tables,
        max_seq_len,
        q_offset=q_offset,
        B=B,
        next_n=next_n,
        kv_block_size=kv_block_size,
    )


# ---- the monokernel's K3.1 indexer (--enable-amd-monokernel, gfx950, no rope)


def hist_topk_workspace(indexer, config):
    """Preallocate the gfx950 logits/top-k workspace before pools are sized,
    or return None for unsupported shapes."""
    from sglang.kernels.ops.attention.dsa.hip_gfx950 import loader
    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
    from sglang.srt.layers.attention.dsa.gfx950_fused_indexer import (
        _WORKSPACES,
        HEAD_DIM,
        INDEX_TOPK,
        N_HEADS,
        _workspace_key,
        prealloc_workspace,
        supported_hardware,
    )
    from sglang.srt.runtime_context import get_spec

    if not (
        supported_hardware()
        and indexer.head_dim == HEAD_DIM
        and indexer.n_heads == N_HEADS
        and indexer.index_topk == INDEX_TOPK
        and indexer.num_init_tokens == 0
        and indexer.num_local_tokens == 0
    ):
        return None
    max_ctx = getattr(config, "max_position_embeddings", 0) or 0
    if not max_ctx or loader.modules_or_none() is None:
        return None
    # the draft rows past the context, then 4 more pages of slack
    extra = 4 + (get_spec().speculative_num_draft_tokens or 0)
    max_ctx = (-(-(max_ctx + extra) // 64) + 4) * 64
    device = torch.device("cuda", torch.cuda.current_device())
    prealloc_workspace(device=device, max_cols=max_ctx, fp8_dtype=fp8_dtype)
    workspace = _WORKSPACES.get(_workspace_key(device, fp8_dtype))
    if workspace is not None:
        print_info_once("DSA indexer: gfx950 logits+histogram and one-launch top-k")
    return workspace


def hist_topk_paged(
    indexer, ws, forward_batch, layer_id, q_fp8, weights, seqlens, metadata
):
    """The paged logits and top-k as the gfx950 fused indexer's last two launches
    (physical page_size=1 slots, -1 padded, as metadata.topk_transform), or None
    for the standard path."""
    from sglang.kernels.ops.attention.dsa.hip_gfx950 import loader
    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
    from sglang.srt.layers.attention.dsa.gfx950_fused_indexer import (
        INDEX_TOPK,
        MAX_ROWS,
        PAGE_SIZE,
    )
    from sglang.srt.mem_cache.dsa_cache_layer_split import LayerSplitDSATokenToKVPool
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    mode = forward_batch.forward_mode
    if not (mode.is_decode_or_idle() or mode.is_target_verify()):
        return None
    if metadata.force_unfused_topk:
        return None
    pool = get_token_to_kv_pool()
    if pool.index_page_size != PAGE_SIZE or isinstance(
        pool, LayerSplitDSATokenToKVPool
    ):
        return None
    rows = q_fp8.shape[0]
    page_table_64 = metadata.get_page_table_64()
    seqlens = seqlens.reshape(-1)
    if not (1 <= rows <= MAX_ROWS) or seqlens.shape[0] != rows:
        return None
    if tuple(q_fp8.shape[1:]) != (indexer.n_heads, indexer.head_dim) or tuple(
        weights.shape
    ) != (rows, indexer.n_heads):
        return None  # an indexer TP split of the heads
    if (
        page_table_64 is None
        or page_table_64.shape[0] == 0
        or not page_table_64.is_contiguous()
        or page_table_64.dtype != torch.int32
        or seqlens.dtype != torch.int32
    ):
        return None
    n_pt, blocks = page_table_64.shape
    if blocks * PAGE_SIZE > ws.max_cols:
        return None
    # draft rows share their request's table: broadcast into the workspace
    if n_pt < rows:
        if rows % n_pt or rows * blocks > ws.page_table.numel():
            return None
        dst = ws.page_table[: rows * blocks].view(rows, blocks)
        dst.view(n_pt, rows // n_pt, blocks).copy_(page_table_64.unsqueeze(1))
        page_table_64 = dst
    else:
        page_table_64 = page_table_64[:rows]
    _, _, logits_mod, topk_mod = loader.modules()
    kv_read = indexer._get_index_k_read_buffer(pool, layer_id).view(fp8_dtype)
    logits = ws.logits[:rows, : blocks * PAGE_SIZE]
    out = torch.empty((rows, INDEX_TOPK), dtype=torch.int32, device=q_fp8.device)
    logits_mod.logits_hist(
        q_fp8,
        kv_read,
        weights.contiguous(),
        seqlens,
        page_table_64,
        logits,
        ws.ghist[:rows],
        loader.LOGITS_BLOCKS_PER_ROW,
    )
    topk_mod.topk_transform(
        logits,
        seqlens,
        page_table_64,
        out,
        ws.ghist[:rows],
        ws.cursor[:rows],
        ws.cand_idx[:rows],
        ws.cand_val[:rows],
        loader.TOPK_G,
        PAGE_SIZE,
    )
    return out


def fuse_index_k_norm(indexer, forward_batch, dcp_size: int) -> bool:
    """Whether this call's index-K store also applies k_norm (one launch): aiter's
    preshuffled page-64 ue8m0 cache, a fp32 LayerNorm k_norm, no rope after it,
    decode or verify, DCP 1, no layer-split pool."""
    from sglang.srt.layers.cp.utils import is_cp_active
    from sglang.srt.layers.layernorm import LayerNorm
    from sglang.srt.mem_cache.dsa_cache_layer_split import LayerSplitDSATokenToKVPool
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    kn = indexer.k_norm
    mode = forward_batch.forward_mode
    if not (
        indexer.skip_rope
        and indexer.bf16_head_gate
        and not indexer.use_dsa_indexer_fusion
        and not indexer.use_aiter_fused_fp8_writer
        and indexer.head_dim == 128
        and indexer.block_size == 128
        and indexer.scale_fmt == "ue8m0"
        and isinstance(kn, LayerNorm)
        and kn.use_bias
        and kn.elementwise_affine
        and kn.weight.dtype == torch.float32
        and kn.bias.dtype == torch.float32
        and (mode.is_decode_or_idle() or mode.is_target_verify())
        and not is_cp_active(forward_batch)
        and dcp_size == 1
    ):
        return False
    pool = get_token_to_kv_pool()
    return pool.index_page_size == 64 and not isinstance(
        pool, LayerSplitDSATokenToKVPool
    )


def index_k_norm_store_into(indexer, pool, layer_id, key, out_cache_loc) -> None:
    """k_norm(key) quantized into the preshuffled FP8 index-K cache, one launch."""
    from sglang.kernels.ops.attention.dsa.index_k_norm_store import index_k_norm_store
    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype

    kn = indexer.k_norm
    index_k_norm_store(
        key,
        kn.weight,
        kn.bias,
        kn.variance_epsilon,
        pool.get_index_k_with_scale_buffer(layer_id=layer_id).view(fp8_dtype),
        out_cache_loc.contiguous(),
    )
