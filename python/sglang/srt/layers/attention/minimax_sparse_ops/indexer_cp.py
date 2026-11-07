"""Opt-in, decode-only context partitioning of MiniMax's replicated index K.

This first implementation gathers the existing post-RoPE index queries rather
than changing checkpoint loading or fused projections. On NVLink hosts with
multicast, the query and candidate gathers are symmetric-memory multimem stores.
Otherwise the query gather goes through SGLang's communicator so AITER registration
and graph capture remain owned by the runtime, and the candidates return through a
second gather, or on the NVFP4 index cache through an IPC exchange. Communication
cost must be included in performance comparisons.
"""

import logging

import torch

logger = logging.getLogger(__name__)

# NVFP4 verify batches below this many requests keep the native indexer: there the Q
# gather and candidate exchange cost more than the index-K reads CP saves (TP4, graphs)
CP_MIN_VERIFY_REQS = 12

# Arbitrary cap on decode rows for the multimem buffers (2 MB + 8 MB at W=8);
# larger batches gather through the communicator.
MULTIMEM_MAX_ROWS = 1024


def unsupported_reason(
    *,
    gfx950,
    sm90,
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
    if gfx950:
        if (tp_size, attn_tp_size, attn_cp_size, attn_dp_size) != (4, 4, 1, 1):
            return "requires TP4 with attention CP/DP size 1"
    elif sm90:
        # At TP8 each rank still serves one index head; head pairs are replicated.
        if attn_tp_size not in (4, 8) or (tp_size, attn_cp_size, attn_dp_size) != (
            attn_tp_size,
            1,
            1,
        ):
            return "requires TP4 or TP8 with attention CP/DP size 1"
    else:
        return "requires AMD gfx950 or NVIDIA SM90"
    if (index_heads, kv_heads, head_dim, block_size, topk) != (4, 4, 128, 128, 16):
        return "requires four index/KV heads, dimension 128, block 128, top-k 16"
    if score_type != "max" or not radix_topk:
        return "requires max scores and the radix decode top-k"
    if not 0 < max_context_len <= 16384 * 128:
        return "context length exceeds the radix selector contract"
    if not draft_is_chain:
        return "verify rows must be chain drafts: no speculation, or EAGLE with top-k 1"
    if tbo or hisparse or fp8_query or dense_sparse_decode:
        return "TBO, HiSparse, FP8 queries, and dense sparse decode are unsupported"
    return None


def draft_is_chain_layout(algorithm, eagle_topk):
    """Whether verify rows reach the indexer as independent chain rows.

    Allowlisted, not tree-denylisted: an unrecognized algorithm (NGRAM's
    tree-in-mask) must disable CP rather than silently mis-score. DSPARK
    qualifies only under the static (verify-all) schedule, where every request
    verifies its whole draft block; ragged verify trims rows per request.
    """
    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    algo = SpeculativeAlgorithm.from_string(algorithm)
    if algo.is_dspark():
        return read_ragged_verify_mode() is RaggedVerifyMode.STATIC
    return algo.is_none() or (algo.is_eagle() and (eagle_topk or 1) == 1)


def make_indexer_cp(backend, runner, sparse_cfg):
    from sglang.srt.environ import envs

    if not envs.SGLANG_MINIMAX_M3_INDEXER_CP.get():
        return None
    from sglang.srt.layers.moe.utils import is_tbo_enabled
    from sglang.srt.runtime_context import get_parallel, get_spec
    from sglang.srt.utils import is_hip, is_sm90_supported

    parallel = get_parallel()
    spec = get_spec()
    arch = (
        torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
        if is_hip()
        else ""
    )
    reason = unsupported_reason(
        gfx950=arch.split(":")[0] == "gfx950",
        sm90=not is_hip() and is_sm90_supported(),
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
    # only the NVFP4 index cache reads the exchange; it is a collective on every rank
    if backend.kv_is_nvfp4:
        from sglang.kernels.ops.attention.minimax_sparse.decode.verify_topk_gfx950 import (
            CpExchange,
        )

        cp.exchange = CpExchange(cp.group, cp.group.device)
    cp.warmup()
    logger.info(
        "MiniMax indexer CP enabled: TP%d, decode-only, replicated index cache, "
        "query gather + %s, top-k 16; index-value layers use TP",
        cp.world,
        "IPC candidate exchange" if cp.exchange is not None else "candidate gather",
    )
    return cp


class MiniMaxIndexerCP:
    def __init__(self, group):
        if group.world_size not in (4, 8):
            raise ValueError("MiniMax indexer CP requires a four- or eight-rank group")
        self.group = group
        self.world = group.world_size
        self.rank = group.rank_in_group
        self.exchange = None
        # (query, candidate) multimem states; None keeps both gathers on the communicator.
        self.multimem = None

    def warmup(self):
        """Initialize the runtime, PyNCCL and multimem gather paths before graph capture."""
        probe = torch.zeros((1, 128), dtype=torch.bfloat16, device=self.group.device)
        output = torch.empty((self.world, 128), dtype=probe.dtype, device=probe.device)
        self.group.all_gather_into_tensor(output, probe)
        comm = self.group.get_available_pynccl_comm()
        if comm is not None:
            # The ordinary eager fallback can use torch.distributed, whereas
            # graph replay uses PyNCCL. Warm the latter explicitly as well.
            with comm.change_state(enable=True):
                comm.all_gather(output, probe)
        self.multimem = self._build_multimem()
        torch.cuda.synchronize(probe.device)

    def _build_multimem(self):
        """Collective, with inputs identical on every rank, so all ranks agree."""
        from sglang.srt.runtime_context import get_parallel
        from sglang.srt.utils import is_hip

        if is_hip():
            return None
        if get_parallel().nnodes > 1:
            from sglang.srt.distributed.parallel_state import in_the_same_node_as

            if not all(in_the_same_node_as(self.group.cpu_group, source_rank=0)):
                return None
        from sglang.srt.distributed.device_communicators import triton_symm_mem_ag

        try:
            states = tuple(
                triton_symm_mem_ag.create_state(
                    group=self.group.device_group,
                    rank_in_group=self.rank,
                    max_tokens=1,
                    hidden_size=self.world * MULTIMEM_MAX_ROWS * row_width,
                    device=self.group.device,
                )
                # bf16 per decode row: a 128-dim query, and world x 16 int64 candidates.
                for row_width in (128, self.world * 64)
            )
        except Exception as e:  # rendezvous refused: every rank sees the same failure
            logger.warning("MiniMax indexer CP multimem setup failed: %s", e)
            return None
        if any(state.symm_mem_hdl.multicast_ptr == 0 for state in states):
            logger.warning("MiniMax indexer CP multimem disabled: no multicast mapping")
            return None
        logger.info("MiniMax indexer CP gathers use multimem stores")
        return states

    def supports(
        self, q, k_cache, max_seqlen, req_to_token, packed_queries=1, local_blocks=1
    ):
        # Shape/dtype/config predicates only. A rank-local exception must never
        # choose a different collective sequence from its peers.
        from sglang.kernels.ops.attention.minimax_sparse.common.nvfp4 import (
            Nvfp4Cache,
        )

        common = (
            q.ndim == 3
            and q.shape[1:] == (1, 128)
            and req_to_token.stride(1) == 1
            and 0 < max_seqlen <= min(req_to_token.shape[1], 16384 * 128)
        )
        if isinstance(k_cache, Nvfp4Cache):
            from sglang.kernels.ops.attention.minimax_sparse.decode.verify_topk_gfx950 import (
                CP_MAX_ROWS,
                can_fuse_verify_topk,
            )

            return (
                common
                and packed_queries > 1
                and q.shape[0] % packed_queries == 0
                and q.shape[0] // packed_queries >= CP_MIN_VERIFY_REQS
                # the fused verify indexer scores all four heads' rows in one 32-row tile
                and packed_queries * 4 <= 32
                # the exchange's receive buffers hold CP_MAX_ROWS rows per head
                and q.shape[0] <= CP_MAX_ROWS
                and can_fuse_verify_topk(
                    q, k_cache, packed_queries, 16, local_blocks, k_cache.page_size, 128
                )
            )
        return (
            common
            and q.dtype in (torch.bfloat16, torch.float16)
            and k_cache.ndim == 3
            and k_cache.shape[1:] == (1, 128)
            and k_cache.dtype
            in (q.dtype, torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2)
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
        packed_queries=1,
    ):
        from sglang.kernels.ops.attention.minimax_sparse.common.nvfp4 import (
            Nvfp4Cache,
        )
        from sglang.kernels.ops.attention.minimax_sparse.common.utils import unit_scale
        from sglang.kernels.ops.attention.minimax_sparse.decode.indexer_cp import (
            merge_candidates,
            score_local_blocks,
            select_local_candidates,
        )
        from sglang.kernels.ops.attention.minimax_sparse.decode.verify_topk_gfx950 import (
            cp_verify_exchange,
        )

        batch = q.shape[0]
        if batch == 0:
            return torch.empty((1, 0, 16), dtype=torch.int32, device=q.device)
        local_q = q[:, 0].contiguous()
        world = self.world
        multimem = self.multimem is not None and batch <= MULTIMEM_MAX_ROWS
        # The offset of q in its allocation is shape-determined, so this agrees across ranks.
        if multimem and local_q.data_ptr() % 16 == 0:
            gathered_q = _multimem_all_gather(self.multimem[0], local_q)
        else:
            gathered_q = torch.empty(
                (world * batch, 128), dtype=q.dtype, device=q.device
            )
            self.group.all_gather_into_tensor(gathered_q, local_q)
        q_heads = gathered_q.view(world, batch, 128)
        sm_scale = 128**-0.5 if sm_scale is None else sm_scale
        if isinstance(k_cache, Nvfp4Cache):
            # sm_scale folds q and k scales as verify_topk_idx does, so the ids match it
            return cp_verify_exchange(
                q_heads=q_heads,
                k_cache=k_cache,
                req_to_token=req_to_token,
                seq_lens=seq_lens,
                max_seqlen=max_seqlen,
                slot_ids=slot_ids,
                num_draft=packed_queries,
                block_size=128,
                topk=16,
                init_blocks=init_blocks,
                local_blocks=local_blocks,
                sm_scale=sm_scale * unit_scale(q_scale) * unit_scale(k_scale),
                exchange=self.exchange,
            )
        sm_scale *= 1.0 if q_scale is None else q_scale
        scores = score_local_blocks(
            q_heads,
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
            packed_queries=packed_queries,
        )
        keys = select_local_candidates(scores, seq_lens, self.rank, max_seqlen)
        if multimem:
            received = _multimem_all_gather(self.multimem[1], keys)
        else:
            received = torch.empty(
                (world, world, batch, 16), dtype=torch.int64, device=q.device
            )
            # Pure byte copies: numeric conversion would destroy the score/ID keys.
            # Float32 views also qualify for the runtime's AITER custom gather.
            self.group.all_gather_into_tensor(
                received.view(torch.float32).view(world * world * batch, 32),
                keys.view(torch.float32).view(world * batch, 32),
            )
        return merge_candidates(received.view(world, world, batch, 16), self.rank)


def _multimem_all_gather(state, local):
    """Every rank's ``local`` bytes in rank order, flat; a view of the symmetric
    buffer, which the next gather overwrites only after every rank enters it."""
    from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
        all_gather_inner,
    )

    flat = local.view(torch.bfloat16).view(1, -1)
    out = all_gather_inner(state, flat, state.world_size * flat.shape[1], safe=False)
    return out.view(local.dtype)
