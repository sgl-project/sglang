"""Opt-in, decode-only context partitioning of MiniMax's replicated index K.

This first implementation gathers the existing post-RoPE index queries rather
than changing checkpoint loading or fused projections. Both gathers go through
SGLang's communicator so AITER registration and graph capture remain owned by
the runtime. Communication cost must be included in performance comparisons.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def unsupported_reason(
    *,
    gfx950,
    tp_size,
    attn_tp_size,
    attn_cp_size,
    attn_dp_size,
    index_heads,
    kv_heads,
    head_dim,
    block_size,
    topk,
    score_type,
    max_context_len,
    radix_topk,
    draft_is_chain,
    tbo,
    hisparse,
    fp8_query,
    dense_sparse_decode,
):
    """Run-level gates: identical on every rank, with no device data reads."""
    if not gfx950:
        return "requires AMD gfx950"
    if (tp_size, attn_tp_size, attn_cp_size, attn_dp_size) != (4, 4, 1, 1):
        return "requires TP4 with attention CP/DP size 1"
    if (index_heads, kv_heads, head_dim, block_size, topk) != (4, 4, 128, 128, 16):
        return "requires four index/KV heads, dimension 128, block 128, top-k 16"
    if score_type != "max" or not radix_topk:
        return "requires max scores and the ROCm radix top-k tie ordering"
    if not 0 < max_context_len <= 16384 * 128:
        return "context length exceeds the ROCm radix selector contract"
    if not draft_is_chain:
        return "verify rows must be chain drafts: no speculation, or EAGLE with top-k 1"
    if tbo or hisparse or fp8_query or dense_sparse_decode:
        return "TBO, HiSparse, FP8 queries, and dense sparse decode are unsupported"
    return None


def draft_is_chain_layout(algorithm, eagle_topk):
    """Whether verify rows reach the indexer as independent chain rows.

    Allowlisted, not tree-denylisted: an unrecognized algorithm (DSPARK's ragged
    verify, NGRAM's tree-in-mask) must disable CP rather than silently mis-score.
    """
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    algo = SpeculativeAlgorithm.from_string(algorithm)
    return algo.is_none() or (algo.is_eagle() and (eagle_topk or 1) == 1)


def make_indexer_cp(backend, runner, sparse_cfg):
    from sglang.srt.environ import envs

    if not envs.SGLANG_MINIMAX_M3_INDEXER_CP.get():
        return None
    from sglang.srt.layers.moe.utils import is_tbo_enabled
    from sglang.srt.runtime_context import get_parallel, get_spec
    from sglang.srt.utils import is_hip

    parallel = get_parallel()
    spec = get_spec()
    arch = (
        torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
        if is_hip()
        else ""
    )
    reason = unsupported_reason(
        gfx950=arch.split(":")[0] == "gfx950",
        tp_size=parallel.tp_size,
        attn_tp_size=parallel.attn_tp_size,
        attn_cp_size=parallel.attn_cp_size,
        attn_dp_size=parallel.attn_dp_size,
        index_heads=sparse_cfg["sparse_num_index_heads"],
        kv_heads=runner.model_config.get_total_num_kv_heads(),
        head_dim=backend.idx_head_dim,
        block_size=backend.block_size_k,
        topk=backend.topk_blocks,
        score_type=backend.score_type,
        max_context_len=backend.max_context_len,
        radix_topk=envs.SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX.get(),
        draft_is_chain=draft_is_chain_layout(
            spec.speculative_algorithm, spec.speculative_eagle_topk
        ),
        tbo=is_tbo_enabled(),
        hisparse=backend.hisparse_coordinator is not None,
        fp8_query=backend.fp8_attn_gemm,
        dense_sparse_decode=backend.use_dense_sparse_decode,
    )
    if reason is not None:
        logger.warning("MiniMax indexer CP disabled: %s", reason)
        return None
    cp = MiniMaxIndexerCP(parallel.attn_tp_group)
    cp.warmup()
    logger.info(
        "MiniMax indexer CP enabled: TP4, decode-only, replicated index cache, "
        "query gather + candidate gather, top-k 16; index-value layers use TP"
    )
    return cp


class MiniMaxIndexerCP:
    def __init__(self, group):
        if group.world_size != 4:
            raise ValueError("MiniMax indexer CP requires a four-rank group")
        self.group = group
        self.rank = group.rank_in_group

    def warmup(self):
        """Initialize both runtime and PyNCCL gather paths before graph capture."""
        probe = torch.zeros((1, 128), dtype=torch.bfloat16, device=self.group.device)
        output = torch.empty((4, 128), dtype=probe.dtype, device=probe.device)
        self.group.all_gather_into_tensor(output, probe)
        comm = self.group.pynccl_comm
        if comm is not None and comm.available:
            # The ordinary eager fallback can use torch.distributed, whereas
            # graph replay uses PyNCCL. Warm the latter explicitly as well.
            with comm.change_state(enable=True):
                comm.all_gather(output, probe)
        torch.cuda.synchronize(probe.device)

    def supports(self, q, k_cache, max_seqlen, req_to_token):
        # Shape/dtype/config predicates only. A rank-local exception must never
        # choose a different collective sequence from its peers.
        return (
            q.ndim == 3
            and q.shape[1:] == (1, 128)
            and q.dtype in (torch.bfloat16, torch.float16)
            and k_cache.ndim == 3
            and k_cache.shape[1:] == (1, 128)
            and k_cache.dtype
            in (q.dtype, torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2)
            and req_to_token.stride(1) == 1
            and 0 < max_seqlen <= min(req_to_token.shape[1], 16384 * 128)
        )

    def __call__(
        self,
        q,
        k_cache,
        req_to_token,
        slot_ids,
        seq_lens,
        max_seqlen,
        init_blocks,
        local_blocks,
        sm_scale=None,
        q_scale=None,
        k_scale=None,
    ):
        from sglang.kernels.ops.attention.minimax_sparse.decode.indexer_cp import (
            merge_candidates,
            score_local_blocks,
            select_local_candidates,
        )

        batch = q.shape[0]
        if batch == 0:
            return torch.empty((1, 0, 16), dtype=torch.int32, device=q.device)
        local_q = q[:, 0].contiguous()
        gathered_q = torch.empty((4 * batch, 128), dtype=q.dtype, device=q.device)
        self.group.all_gather_into_tensor(gathered_q, local_q)
        sm_scale = 128**-0.5 if sm_scale is None else sm_scale
        sm_scale *= 1.0 if q_scale is None else q_scale
        scores = score_local_blocks(
            gathered_q.view(4, batch, 128),
            k_cache,
            req_to_token,
            slot_ids,
            seq_lens,
            max_seqlen,
            self.rank,
            init_blocks,
            local_blocks,
            sm_scale,
            1.0 if k_scale is None else k_scale,
        )
        keys = select_local_candidates(scores, seq_lens, self.rank, max_seqlen)
        received = torch.empty((4, 4, batch, 16), dtype=torch.int64, device=q.device)
        # Pure byte copies: numeric conversion would destroy the score/ID keys.
        # Float32 views also qualify for the runtime's AITER custom gather.
        self.group.all_gather_into_tensor(
            received.view(torch.float32).view(16 * batch, 32),
            keys.view(torch.float32).view(4 * batch, 32),
        )
        return merge_candidates(received, self.rank)
