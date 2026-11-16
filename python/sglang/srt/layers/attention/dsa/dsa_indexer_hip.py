"""ROCm DSA index-K sharding and shared-request target-verify logits."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch

from sglang.kernels.ops.attention.dsa import (
    aiter_paged_mqa_logits_persistent_schedule,
    aiter_paged_mqa_logits_verify,
)
from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated
from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.srt.layers.layernorm import LayerNorm
from sglang.srt.utils import is_gfx95_supported, print_info_once

_is_gfx95 = is_gfx95_supported()


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
# Target verify on the shared-index-K path: each request's full length and page-table row.
_verify_request_lens = _PerForwardMemo()
_verify_request_block_tables = _PerForwardMemo()
# FP4 index-K logits: each request's page-table row and its first query row
_fp4_request_block_tables = _PerForwardMemo()
_fp4_query_starts = _PerForwardMemo()
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


def store_fp4_index_k(
    forward_batch, pool, layer_id: int, key, dcp_size: int, dcp_rank: int
) -> None:
    """The MXFP4 index-K store: this DCP rank's rows of the batch's keys."""
    from sglang.kernels.ops.attention.dsa import fp4_index_hip

    out_loc = forward_batch.out_cache_loc
    if dcp_size > 1:
        out_loc = dcp_local_store_loc(
            forward_batch, layer_id, out_loc, dcp_size, dcp_rank
        )
    fp4_index_hip.store_fp4_index_k(
        key, *pool.get_index_k_fp4_buffers(layer_id), out_loc.contiguous()
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
    """The monokernel's K3.1 indexer (no rope, its own projections): the paged logits
    with their histogram and the top-k in one launch (gfx950). Preallocate the
    workspace before pools are sized, or return None where it does not run."""
    from sglang.srt.layers.attention.dsa.utils import (
        aiter_can_use_preshuffle_paged_mqa,
    )
    from sglang.srt.runtime_context import get_exec

    if not (
        aiter_can_use_preshuffle_paged_mqa()
        and indexer.skip_rope
        and get_exec().kernel.enable_amd_monokernel
        and not indexer.use_gfx950_fused_indexer
    ):
        return None

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


def init_indexer_hip(indexer, config) -> None:
    """Indexer.__init__'s AITER parts, after the gfx950 fused indexer's gate (which
    hist_topk_workspace reads)."""
    from sglang.srt.layers.attention.dsa.indexer_prologue_hip import (
        make_rocm_nope_prologue,
    )
    from sglang.srt.runtime_context import get_exec

    # MXFP4 index-K (Kimi-K3.1 on gfx950 under DCP); the pool checks the setup
    indexer.fp4_index_k = get_exec().kernel.enable_dsa_fp4_indexer
    # gfx950 decode / verify prologue of the NoPE (Kimi-K3.1) indexer
    if _is_gfx95:
        indexer.rocm_nope_prologue = make_rocm_nope_prologue(indexer)
    indexer._hist_topk = hist_topk_workspace(indexer, config)


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


def fuse_index_k_norm(indexer, forward_batch) -> bool:
    """Whether this call's index-K store also applies k_norm (one launch)."""
    from sglang.srt.layers.attention.dsa.dsa_indexer import _index_k_dcp
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
        and _index_k_dcp()[0] == 1
    ):
        return False
    pool = get_token_to_kv_pool()
    return pool.index_page_size == 64 and not isinstance(
        pool, LayerSplitDSATokenToKVPool
    )


def store_index_k_prenorm(indexer, forward_batch, layer_id: int, key) -> None:
    """Indexer._store_index_k_cache for k_norm's input: k_norm(key) quantized into
    the preshuffled FP8 index-K cache, one launch."""
    from sglang.kernels.ops.attention.dsa.index_k_norm_store import index_k_norm_store
    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    pool = get_token_to_kv_pool()
    if hasattr(pool, "invalidate_index_buffer_for_layer"):
        pool.invalidate_index_buffer_for_layer(layer_id)
    if hasattr(pool, "_is_layer_owned") and not pool._is_layer_owned(layer_id):
        return
    kn = indexer.k_norm
    index_k_norm_store(
        key,
        kn.weight,
        kn.bias,
        kn.variance_epsilon,
        pool.get_index_k_with_scale_buffer(layer_id=layer_id).view(fp8_dtype),
        forward_batch.out_cache_loc.contiguous(),
    )


def fp4_request_rows(
    forward_batch, layer_id: int, block_tables: torch.Tensor, q_offset: int, next_n: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Each request's page-table row (dense, as AITER's row-group logits index it) and
    the start of its ``next_n`` query rows."""
    key = (id(forward_batch), q_offset, next_n)
    tables = _fp4_request_block_tables.get(
        key=key,
        layer_id=layer_id,
        compute=lambda: block_tables[:q_offset:next_n].contiguous(),
    )
    starts = _fp4_query_starts.get(
        key=key,
        layer_id=layer_id,
        compute=lambda: torch.arange(
            0, q_offset + 1, next_n, dtype=torch.int32, device=block_tables.device
        ),
    )
    return tables, starts


def fp4_topk_paged(
    indexer,
    forward_batch,
    layer_id: int,
    query,
    weights: torch.Tensor,
    metadata,
) -> torch.Tensor:
    """Decode / verify top-k over the MXFP4 index-K: this rank's local KV slots, with
    their counts in ``metadata.attn_metadata.dcp_topk_length`` (dcp_local_topk)."""
    from sglang.kernels.ops.attention.dsa import fp4_index_hip
    from sglang.srt.layers.attention.dsa.dsa_dcp_sharded import dcp_local_topk
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    # AITER's row-group logits score each row up to its own rank-local length,
    # the only columns the DCP top-k reads
    assert isinstance(query, fp4_index_hip.Fp4IndexQuery), (
        "FP4 index-K needs the FP4 prologue"
    )
    pool = get_token_to_kv_pool()
    page_size = pool.index_page_size
    block_tables = metadata.get_page_table_64()
    mode = forward_batch.forward_mode
    if mode.is_target_verify() or mode.is_draft_extend_v2():
        seqlens_32 = metadata.get_seqlens_expanded()
    else:
        seqlens_32 = metadata.get_seqlens_int32()
    q_offset = sum(metadata.get_dsa_extend_len_cpu())
    B = metadata.get_seqlens_int32().shape[0]
    next_n = max(q_offset // B if B > 0 else 1, 1)
    # this rank's share of each row's length, built with the metadata
    local_lens = metadata.paged_mqa_ctx_lens_2d.view(-1)[:q_offset]
    request_tables, query_starts = fp4_request_rows(
        forward_batch, layer_id, block_tables, q_offset, next_n
    )
    logits = fp4_index_hip.fp4_paged_mqa_logits(
        query[:q_offset],
        weights[:q_offset].squeeze(-1),
        *pool.get_index_k_fp4_buffers(layer_id),
        request_tables,
        query_starts,
        local_lens,
        block_tables.shape[1] * page_size,
    )
    # attention reads local KV slots and their counts
    return dcp_local_topk(
        logits,
        local_lens,
        seqlens_32.view(-1)[:q_offset],
        block_tables,
        indexer.index_topk,
        page_size,
        metadata.attn_metadata.topk_v2_plan,
        metadata.attn_metadata.dcp_topk_length,
    )


def fp4_dcp_gather_index_k(
    layer_id: int, metadata, block_tables: torch.Tensor, seq_lens_cpu: torch.Tensor
):
    """Prefill under DCP: every request's MXFP4 index-K from all ranks, gathered as
    68-byte rows and staged back into pages (fp4_index_hip.Fp4IndexKeys)."""
    from sglang.kernels.ops.attention.dsa import fp4_index_hip
    from sglang.srt.layers.attention.dsa.dsa_dcp_sharded_hip import (
        dcp_gather_request_rows,
    )
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    dcp_size, dcp_rank = parallel.attn_dcp_size, parallel.attn_dcp_rank
    local_lens_cpu = get_dcp_lens(seq_lens_cpu, dcp_size, dcp_rank)
    rows = fp4_index_hip.gather_fp4_index_k_rows(
        *get_token_to_kv_pool().get_index_k_fp4_buffers(layer_id),
        block_tables.contiguous(),
        get_dcp_lens(metadata.get_indexer_seq_len(), dcp_size, dcp_rank),
        int(local_lens_cpu.sum()),
        max(int(local_lens_cpu.max()), 1),
    )
    return fp4_index_hip.stage_fp4_index_k_rows(
        dcp_gather_request_rows(rows, seq_lens_cpu)
    )


def fp4_logits_plans(metadata) -> Optional[dict]:
    """The FP4 prefill logits schedules of an eager batch: every layer scores the same
    row chunks over the same key table, so the first keeps them (and their host read
    of ks) for the rest, on the batch's DSAMetadata outside its fields (a replace()
    copy starts empty). None under graph capture."""
    from sglang.srt.layers.attention.dsa_backend import DSAMetadata
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
        is_in_breakable_cuda_graph,
    )

    if torch.cuda.is_current_stream_capturing() or is_in_breakable_cuda_graph():
        return None
    attn_metadata = metadata.attn_metadata
    if not isinstance(attn_metadata, DSAMetadata):
        return None
    plans = attn_metadata.__dict__.get("_fp4_logits_plans")
    if plans is None:
        plans = {}
        object.__setattr__(attn_metadata, "_fp4_logits_plans", plans)  # frozen
    return plans


def fp4_prefill_q(indexer, query: torch.Tensor, x):
    """FP4 queries and the bf16 head gate the FP4 prefill logits read."""
    from sglang.kernels.ops.attention.dsa import fp4_index_hip

    return fp4_index_hip.indexer_q_fp4_head_gate(
        query, indexer._project_head_gates(x), 1.0, torch.bfloat16
    )


def fp4_prefill(
    indexer,
    x,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
    enable_dual_stream: bool,
    forward_batch,
    layer_id: int,
    act_quant,
):
    """FP4 index-K prefill outside the graph split op: store the keys, return
    fp4_prefill_q's queries and gate."""
    assert not isinstance(x, tuple) or len(x) == 3, "the gate needs bf16 input"
    query, key, _ = indexer._get_q_k_bf16(
        q_lora, x, positions, enable_dual_stream, forward_batch=forward_batch
    )
    indexer._store_index_k_cache(
        forward_batch=forward_batch,
        layer_id=layer_id,
        key=key,
        act_quant=act_quant,
    )
    return fp4_prefill_q(indexer, query, x)


class Fp32LayerNormHip(LayerNorm):
    """An fp32-weight LayerNorm (Kimi-K3.1's index k_norm) whose bf16 / fp16 input
    runs on gfx950 as fp32 math in one launch instead of upcast + layer_norm +
    downcast: the fused norm without a gate is a LayerNorm."""

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        if (
            _is_gfx95
            and x.dtype in (torch.bfloat16, torch.float16)
            and self.dtype == torch.float32
            and self.elementwise_affine
        ):
            return rms_norm_gated(
                x=x,
                weight=self.weight,
                bias=self.bias if self.use_bias else None,
                eps=self.variance_epsilon,
                is_rms_norm=False,
            )
        return super().forward_hip(x)
