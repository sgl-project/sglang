from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.configs.model_config import (
    get_minimax_sparse_attention_config,
    get_minimax_sparse_disable_value_layer_ids,
    get_minimax_sparse_layer_ids,
    get_minimax_sparse_score_type,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.minimax_sparse_ops.common.index import (
    topk_index_reduce,
)
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_npu

if not is_npu():
    from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
        minimax_sparse_decode,
        minimax_sparse_prefill,
    )


def _npu_use_triton_sparse() -> bool:
    """Whether the NPU sparse path should use the fused triton kernels.

    The fused Triton path is the default on NPU. Set SGLANG_MINIMAX_NPU_TRITON=0
    to fall back to the non-Triton sparse decode path.
    """
    import os

    return is_npu() and bool(int(os.environ.get("SGLANG_MINIMAX_NPU_TRITON", "1")))


def _disable_fused_merge_topk() -> bool:
    """A/B kill-switch for the fused merge-topk kernel (``_merge_topk_local_bnsd``).

    The fused single-kernel path (one launch replacing the ~35-op Python
    ``_merge_sparse_blocks`` eager chain) is the default. Set
    ``MINIMAX_NPU_DISABLE_FUSED_MERGE_TOPK=1`` to force the PRE-fusion Python
    path (``_merge_sparse_blocks``) even when ``init_blocks == 0`` and
    ``local_blocks == 1``, so the fusion's end-to-end effect can be A/B'd by
    restarting the server with/without the flag (same binary, no code change).
    Read once per call (cheap); not registered in the sglang ``envs`` registry on
    purpose -- it's a local diagnostic toggle matching the
    ``MINIMAX_NPU_TRITON_DISABLE_TRIVIAL_SKIP`` convention.
    """
    import os

    return bool(os.environ.get("MINIMAX_NPU_DISABLE_FUSED_MERGE_TOPK"))


def _npu_use_triton_prefill() -> bool:
    """Whether the NPU sparse PREFILL (extend) path uses the fused triton kernels.

    Off by default: prefill still runs the validated PyTorch masked-full attention
    (``_forward_npu_sparse_prefill``). When on, prefill reuses the block-sparse
    triton decode kernels via per-query flattening
    (``_forward_npu_triton_prefill``) -- attending only to the selected
    topk+init+local blocks instead of computing QK over the full seq_len.
    Respects the master kill-switch ``SGLANG_MINIMAX_NPU_TRITON``.
    """
    from sglang.srt.environ import envs

    return _npu_use_triton_sparse() and envs.SGLANG_MINIMAX_NPU_TRITON_PREFILL.get()


# Adaptive crossover: e2e measured Approach A slower at <=8K and faster at >=16K
# (PyTorch full-dense main is O(seq^2), Approach A sparse main ~O(seq)). 12K is the
# midpoint. Auto-enable Approach A at/above this KV length so long-context prefills
# get the sparse win without regressing the short-context case (which stays PyTorch).
# Override with SGLANG_MINIMAX_NPU_TRITON_PREFILL (forces on for all lengths).
MINIMAX_NPU_TRITON_PREFILL_AUTO_MIN_SEQLEN = 20000

# Adaptive block_size_q thresholds: larger BSQ reduces all_seqblock_q, allowing
# more num_score_chunks under the Ascend program_cap (32768), which shortens the
# per-program serial loop in the prefill score kernel.  Trade-off: coarser top-k
# selection granularity.  Standalone tests validate BSQ {1,4,8,16}.
#
# BSQ only affects the **serial loop** (driven by max_seqlen_k); it does NOT
# change coreDim (driven by total_q, already protected by program_cap).
_BSQ_THRESHOLD_16 = 65536  # max_seqlen_k >= 64K -> BSQ=16 (bench_pinpoint: 17ms/layer @131K vs 28ms@BSQ=8)
_BSQ_THRESHOLD_8 = 32768   # max_seqlen_k >= 32K -> BSQ=8
# BSQ=16 is UB-safe for the prefill indexer: BLOCK_SIZE_H=next_pow2(gqa)=1 (not
# padded to 16 like decode), so Q tile = [BSQ*1, 128] = 4KB at BSQ=16.


def _npu_triton_prefill_auto(seq_lens: torch.Tensor) -> bool:
    """Adaptive: auto-use the triton prefill path once KV length crosses the crossover.

    Returns True iff the batch's max KV length >= ``MINIMAX_NPU_TRITON_PREFILL_AUTO_MIN_SEQLEN``
    and the NPU triton master gate is on. ``seq_lens.max().item()`` is a host sync,
    fine because extend/prefill runs eager (not cuda-graph captured).
    """
    if not _npu_use_triton_sparse():
        return False
    return bool(int(seq_lens.max().item()) >= MINIMAX_NPU_TRITON_PREFILL_AUTO_MIN_SEQLEN)


if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class MiniMaxSparseAttnBackend(AttentionBackend):
    def __init__(self, runner: ModelRunner):
        assert isinstance(runner.token_to_kv_pool, MiniMaxSparseKVPool)
        self.is_npu = is_npu()
        self.kv_pool = runner.token_to_kv_pool
        self.req_to_token = runner.req_to_token_pool.req_to_token
        self.max_context_len = int(runner.model_config.context_len)

        hf_config = runner.model_config.hf_config
        sparse_cfg = get_minimax_sparse_attention_config(hf_config)
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        self.dense_layer_ids, self.sparse_layer_ids = get_minimax_sparse_layer_ids(
            sparse_cfg
        )
        self.disable_value_layer_ids: set[int] = set(
            get_minimax_sparse_disable_value_layer_ids(sparse_cfg)
        )
        self.score_type: str = get_minimax_sparse_score_type(sparse_cfg)

        # max_seqlen for the current forward pass, stored as a plain Python int
        # so that it is safe to use inside CUDA graphs (no .item() at graph time).
        # Populated by init_forward_metadata* before each forward.
        self._max_seqlen_q: int = 1
        self._max_seqlen_k: int = 1

        # Layer-invariant prefill flatten + main block_table, cached across the 57
        # sparse layers of one forward pass (built lazily on the first layer,
        # invalidated by init_forward_metadata_out_graph). See _build_prefill_meta.
        self._prefill_meta: Optional[SimpleNamespace] = None

        self.block_size_q = 1
        self.block_size_k = sparse_cfg["sparse_block_size"]
        if "sparse_init_block" in sparse_cfg:
            self.init_blocks = sparse_cfg["sparse_init_block"]
        else:
            init_tokens = sparse_cfg["sparse_init_tokens"]
            self.init_blocks = (
                init_tokens + self.block_size_k - 1
            ) // self.block_size_k
        if "sparse_local_block" in sparse_cfg:
            self.local_blocks = sparse_cfg["sparse_local_block"]
        else:
            local_tokens = sparse_cfg["sparse_local_tokens"]
            self.local_blocks = (
                local_tokens + self.block_size_k - 1
            ) // self.block_size_k + 1
        self.topk_blocks = sparse_cfg["sparse_topk_blocks"]

        # NVIDIA Blackwell (SM100): use MiniMax's MSA kernel (fmha_sm100) only
        # for the main sparse-attention step when the kernel constraints hold.
        # The lightning indexer remains unchanged; missing fmha_sm100 keeps the
        # existing Triton path.
        from sglang.srt.environ import envs

        # MSA (fmha_sm100) is bf16/fp16-only. With an fp8 main KV cache
        # (--kv-cache-dtype fp8_*) keep the sparse path on Triton (it dequants fp8 on
        # load) rather than feeding fp8 bytes to the bf16 kernel; mirrors vLLM's
        # select_main_impl_cls (fp8 KV -> Triton, never MSA).
        if self.is_npu:
            self.use_msa = False
        else:
            from sglang.srt.layers.attention.minimax_sparse_ops.msa import (
                msa_available,
            )

            _main_kv_is_fp8 = self.kv_pool.main_pool.dtype in (
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            )
            self.use_msa = (
                not envs.SGLANG_DISABLE_MSA.get()
                and msa_available()
                and self.block_size_k == 128
                and self.kv_pool.page_size == self.block_size_k
                and self.topk_blocks in (4, 8, 16, 32)
                and not _main_kv_is_fp8
            )
        # Per-forward MSA decode metadata (page table + fmha plan), shared by every
        # sparse layer of a forward; (re)built in init_forward_metadata_out_graph.
        self._msa_dec_meta = None
        if self.use_msa:
            from sglang.srt.layers.dp_attention import get_attention_tp_size

            # Per-rank head counts for the decode plan (== runtime q.shape[1] /
            # k_cache.shape[1]); needed in out_graph where q/k_cache aren't available.
            self.num_q_heads = (
                runner.model_config.num_attention_heads // get_attention_tp_size()
            )
            # KV head count lives on the main sub-pool (== runtime k_cache.shape[1]).
            self.num_kv_heads = self.kv_pool.main_pool.head_num
            # CUDA-graph decode: one persistent plan + page-table buffer per batch
            # size, refreshed in place each step (worklist is length-independent).
            self._msa_nb_max = (
                self.max_context_len + self.block_size_k - 1
            ) // self.block_size_k
            self._msa_cg: dict[int, tuple] = {}

        self.page_size = self.kv_pool.page_size
        self.use_dense_sparse_decode = (
            (not self.is_npu)
            and envs.SGLANG_OPT_USE_MINIMAX_DENSE_SPARSE_DECODE.get()
            and self.block_size_k % self.page_size == 0
        )
        # MSA fmha_sm100 decode is NOT cuda-graph-safe: captured & replayed it returns
        # wrong results that compound across replays (silent ~14% GSM8K loss on B200;
        # masked early by radix-cache prefix reuse, then cliffs under sustained load).
        # Use the MSA decode kernel only when decode does NOT run under cuda graph;
        # otherwise route the decode step through the cuda-graph-safe Triton sparse path.
        # MSA still serves prefill (run eager — prefill cuda graph is disabled), where
        # its long-context speedup matters.
        #
        # Decide from the resolved cuda_graph_config — the same source
        # init_decode_cuda_graph uses to decide capture — not the legacy disable_*
        # server_args flags: the two can disagree under config-native flags, and a
        # mismatch could capture the unsafe MSA decode kernel into a graph.
        from sglang.srt.model_executor.cuda_graph_config import (
            Backend,
            Phase,
            check_cuda_graph_backend,
        )

        _sa = getattr(runner, "server_args", None)
        self.speculative_num_draft_tokens = getattr(
            _sa, "speculative_num_draft_tokens", None
        )
        _decode_cuda_graph = not check_cuda_graph_backend(
            Phase.DECODE, Backend.DISABLED
        )
        self._use_msa_decode = self.use_msa and not _decode_cuda_graph

        # MSA + speculative decode + cuda graph is unsupported: spec verify
        # (TARGET_VERIFY) batches route to forward_extend and are captured into the
        # decode graph, which both dereferences extend metadata absent in the capture
        # batch and would record the MSA prefill kernel into a graph. Fail loudly at
        # startup instead of crashing mid-capture.
        if (
            self.use_msa
            and _decode_cuda_graph
            and getattr(_sa, "speculative_algorithm", None) is not None
        ):
            raise NotImplementedError(
                "MiniMax-M3 MSA attention does not support speculative decoding under "
                "CUDA graph. Use --disable-cuda-graph, set SGLANG_DISABLE_MSA=1, or "
                "disable speculative decoding."
            )
        # MSA owns the main decode step unless dense-sparse-decode does; the dense
        # path only engages when k_cache.shape[1] == 1 (see forward_decode).
        self._msa_owns_decode = self._use_msa_decode and not (
            self.use_dense_sparse_decode and self.kv_pool.main_pool.head_num == 1
        )
        # The page table + effective KV length are allocated and returned by the
        # fused decode top-k kernel each layer, so the backend keeps no metadata.
        self.dense_backend: Optional[AttentionBackend] = None

        logger.info(
            f"[MiniMaxSparse] Backend initialized "
            f"(score_type={self.score_type!r}, "
            f"main_attn={'MSA' if self.use_msa else 'triton'}, "
            f"disable_value_layers={sorted(self.disable_value_layer_ids)})"
        )

    @staticmethod
    def _choose_block_size_q(max_seqlen_k: int) -> int:
        """Pick block_size_q adaptively based on max KV sequence length.

        Larger block_size_q reduces ``all_seqblock_q`` (the number of query blocks),
        which allows more ``num_score_chunks`` under the Ascend program_cap (32768).
        More chunks -> fewer KV blocks scanned per program -> shorter serial loop
        in the prefill score kernel.

        Trade-off: coarser query batching means top-k block selection operates on
        groups of tokens instead of individually, which may slightly reduce
        selection precision (mitigated by init+local forced blocks).

        .. note::
           BSQ only affects the **serial loop** (driven by ``max_seqlen_k``); it
           does NOT change coreDim (driven by ``total_q``, already protected by
           ``program_cap``).
        """
        if max_seqlen_k >= _BSQ_THRESHOLD_16:
            return 16
        if max_seqlen_k >= _BSQ_THRESHOLD_8:
            return 8
        return 1

    @staticmethod
    def _get_safe_block_size_q(
        max_seqlen_k: int,
        extend_lens_cpu: "torch.Tensor | None" = None,
    ) -> int:
        """Adaptive BSQ with cross-request contamination safety guard.

        BSQ>1 batches multiple query tokens into a block. When a request's
        ``extend_len`` is not an exact multiple of BSQ, its last q-block is
        **partial**: the phantom rows (qi beyond the request's real tokens)
        fall into the next request's token range, read that request's Q,
        but write scores using **this request's block_table** (KV).

        The prefill score kernels (_prefill_bnsd_score_kernel,
        _prefill_bnsd_score_attn_kernel) already guard against cross-request
        contamination via ``row_valid = (q_token_flat < q_end)`` where
        ``q_end`` is ``cu_seqlens[r+1]`` (the exclusive upper bound of the
        owning request's token range). Phantom rows' stores are masked, so
        their computed (garbage) scores are never written to the output.

        To prevent phantom rows from wasting memory bandwidth by reading Q
        data from the next request, the kernels also clamp ``q_token_flat``
        to ``max(q_start, q_end - 1)`` before the Q load, confining every
        read to the owning request's token range.

        With these two guards (masked store + clamped read), BSQ>1 is safe
        for ALL batch configurations, including multi-request batches with
        non-aligned extend lengths. The fallback to BSQ=1 is removed.
        """
        bsq = MiniMaxSparseAttnBackend._choose_block_size_q(max_seqlen_k)
        if bsq == 1:
            return 1
        return bsq

    # ------------------------------------------------------------------
    # Delegation helpers
    # ------------------------------------------------------------------

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        # cuda-graph replay views are a SimpleNamespace without extend_seq_lens_cpu,
        # and TARGET_VERIFY sets it to None despite is_extend() — getattr covers both.
        # New forward -> invalidate the cached per-forward MSA decode metadata.
        self._msa_dec_meta = None
        # Invalidate the cached layer-invariant prefill flatten + block_table
        # (see _build_prefill_meta): forces a rebuild on the first sparse layer
        # of this forward pass.
        self._prefill_meta = None
        extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_lens is not None:
            self._max_seqlen_q = int(max(extend_lens))
        else:
            self._max_seqlen_q = 1
        if in_capture and (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
        ):
            # Under cuda-graph capture the dummy batch's seq_lens are tiny, so
            # seq_lens_cpu.max() would under-bound max_blocks and truncate the
            # captured block_table — at replay, real (longer) sequences would
            # miss KV blocks and produce garbage. Decode already used the full
            # context bound for this reason; TARGET_VERIFY (captured into the
            # same decode graph) needs it too.
            self._max_seqlen_k = self.max_context_len
        else:
            self._max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item())

        # Build the MSA decode plan + page table here (eager, outside graph capture)
        # so forward_decode — captured into the graph — only runs device-side ops.
        # Runs at capture, replay, and eager, refreshing the persistent buffers the
        # captured graph reads. Skipped when the dense-sparse-decode path owns decode.
        if self._msa_owns_decode and forward_batch.forward_mode.is_decode_or_idle():
            self._prepare_msa_decode_meta(forward_batch)

    def _prepare_msa_decode_meta(self, forward_batch: ForwardBatch):
        """Refresh the persistent per-batch-size MSA decode plan + page table in place."""
        from sglang.srt.layers.attention.minimax_sparse_ops.msa import (
            build_msa_decode_cg_plan,
            update_msa_decode_cg_meta,
        )

        bs = forward_batch.seq_lens.shape[0]
        if bs == 0:
            return
        entry = self._msa_cg.get(bs)
        if entry is None:
            device = forward_batch.seq_lens.device
            plan = build_msa_decode_cg_plan(
                self.num_q_heads,
                self.num_kv_heads,
                self.block_size_k,
                self.topk_blocks,
                bs,
                device=device,
            )
            kv_indices_buf = torch.zeros(
                bs * self._msa_nb_max, dtype=torch.int32, device=device
            )
            entry = (plan, kv_indices_buf)
            self._msa_cg[bs] = entry
        plan, kv_indices_buf = entry
        update_msa_decode_cg_meta(
            plan,
            kv_indices_buf,
            self.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            self.block_size_k,
            self.topk_blocks,
            self.num_q_heads,
            self.num_kv_heads,
        )
        self._msa_dec_meta = (kv_indices_buf, plan)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        pass

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def _raise_npu_sparse_not_ready(self, phase: str, reason: str) -> None:
        raise NotImplementedError(
            "MiniMax-M3 NPU sparse attention needs native fused operators for "
            f"{phase}: {reason}. Missing/target operators include "
            "flash_prefill_with_topk_index, flash_decode_with_topk_idx, "
            "flash_*_with_gqa_share_sparse, minimax_decode_topk, "
            "minimax_decode_topk_page_table, topk_index_reduce, and "
            "minimax_store_kv_index. The current NPU path provides a slow "
            "PyTorch correctness fallback only for supported score/cache layouts."
        )

    @staticmethod
    def _cache_as_slots(cache: torch.Tensor) -> torch.Tensor:
        if cache.dim() <= 3:
            return cache
        return cache.reshape(-1, cache.shape[-2], cache.shape[-1])

    def _merge_sparse_blocks(
        self,
        topk_blocks: torch.Tensor,
        query_positions: torch.Tensor,
        num_blocks: int,
    ) -> torch.Tensor:
        """Append forced init/local blocks to top-k block ids and deduplicate."""
        total = self.topk_blocks + self.init_blocks + self.local_blocks
        if self.init_blocks <= 0 and self.local_blocks <= 0:
            return topk_blocks

        block_size = self.block_size_k
        q_len = query_positions.shape[0]
        num_idx_heads = topk_blocks.shape[1]
        qcol = query_positions[:, None, None]

        if self.init_blocks == 0 and self.local_blocks == 1:
            local = (query_positions // block_size).clamp(
                min=0, max=max(num_blocks - 1, 0)
            )
            local = (
                local.to(topk_blocks.dtype)
                .view(q_len, 1, 1)
                .expand(-1, num_idx_heads, -1)
            )
            valid_topk = (topk_blocks >= 0) & (topk_blocks < num_blocks)
            valid_topk = valid_topk & (topk_blocks * block_size <= qcol)
            local_duplicate = ((topk_blocks == local) & valid_topk).any(
                dim=-1, keepdim=True
            )
            valid_local = (local >= 0) & (local < num_blocks)
            valid_local = valid_local & (local * block_size <= qcol) & ~local_duplicate
            return torch.cat(
                [
                    torch.where(
                        valid_topk, topk_blocks, torch.full_like(topk_blocks, -1)
                    ),
                    torch.where(valid_local, local, torch.full_like(local, -1)),
                ],
                dim=-1,
            )

        forced_parts = []
        if self.init_blocks > 0:
            forced_parts.append(
                torch.arange(
                    self.init_blocks,
                    device=topk_blocks.device,
                    dtype=topk_blocks.dtype,
                )
                .view(1, 1, -1)
                .expand(q_len, num_idx_heads, -1)
            )
        if self.local_blocks > 0:
            offsets = torch.arange(
                self.local_blocks,
                device=topk_blocks.device,
                dtype=query_positions.dtype,
            )
            block_ids = query_positions // block_size
            first = (block_ids - self.local_blocks + 1).clamp(min=0)
            forced_parts.append(
                (first[:, None] + offsets[None, :])
                .to(topk_blocks.dtype)
                .view(q_len, 1, -1)
                .expand(-1, num_idx_heads, -1)
            )

        forced = torch.cat(forced_parts, dim=-1)
        candidates = torch.cat([forced, topk_blocks], dim=-1)
        valid = (candidates >= 0) & (candidates < num_blocks)
        valid = valid & (candidates * block_size <= qcol)
        invalid_value = torch.full_like(candidates, num_blocks)
        sorted_candidates = torch.sort(
            torch.where(valid, candidates, invalid_value), dim=-1
        ).values
        sorted_valid = sorted_candidates < num_blocks
        previous = torch.cat(
            [
                torch.full_like(sorted_candidates[..., :1], -1),
                sorted_candidates[:, :, :-1],
            ],
            dim=-1,
        )
        keep = sorted_valid & (sorted_candidates != previous)
        ranks = torch.cumsum(keep.to(torch.int32), dim=-1) - 1
        output = torch.full(
            (q_len, num_idx_heads, total + 1),
            -1,
            dtype=topk_blocks.dtype,
            device=topk_blocks.device,
        )
        overflow_rank = torch.full_like(ranks, total)
        scatter_index = torch.where(keep & (ranks < total), ranks, overflow_rank).long()
        scatter_src = torch.where(keep, sorted_candidates, -1)
        output.scatter_(2, scatter_index, scatter_src)
        return output[:, :, :total]

    def _prepare_npu_triton_topk_idx(
        self,
        topk_idx: torch.Tensor,
        seq_lens: torch.Tensor,
        num_idx_heads: int,
        num_kv_heads: int,
        max_blocks: int,
    ) -> torch.Tensor:
        """Prepare NPU Triton top-k ids in the GQA kernel's native layout.

        MiniMax-M3 under TP=16 has one replicated index head and one KV head per
        rank. Its only forced block is the causal local block. Avoid the generic
        ``[KVH, B, K] -> [B, KVH, K]`` transpose, PyTorch append/dedup, and
        transpose back by directly emitting the GQA input layout on NPU. All
        other sparse layouts retain the validated generic path.
        """
        if (
            self.init_blocks == 0
            and self.local_blocks == 1
            and num_idx_heads == num_kv_heads
            and topk_idx.shape[0] == num_kv_heads
            and topk_idx.dtype == torch.int32
            and topk_idx.is_contiguous()
            and seq_lens.is_contiguous()
        ):
            from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton.topk_sparse_decode import (
                append_local_block_to_topk_idx,
            )

            return append_local_block_to_topk_idx(
                topk_idx, seq_lens, self.block_size_k, max_blocks
            )

        if num_idx_heads > num_kv_heads:
            idx_group_size = num_idx_heads // num_kv_heads
            topk_idx = topk_index_reduce(
                topk_idx.view(num_kv_heads, idx_group_size, -1, self.topk_blocks),
                dim=1,
            )

        topk_2d = topk_idx.permute(1, 0, 2).contiguous()
        query_positions = (seq_lens.to(torch.long) - 1).clamp(min=0)
        topk_merged = self._merge_sparse_blocks(
            topk_2d, query_positions, max_blocks
        )
        return topk_merged.permute(1, 0, 2).contiguous()

    def _select_sparse_blocks(
        self,
        idx_q_seq: torch.Tensor,
        idx_k_seq: torch.Tensor,
        query_positions: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Score index blocks with per-query causal masking."""
        block_size = self.block_size_k
        num_blocks = (seq_len + block_size - 1) // block_size
        total = self.topk_blocks + self.init_blocks + self.local_blocks
        if num_blocks == 0:
            return torch.full(
                (idx_q_seq.shape[0], idx_q_seq.shape[1], total),
                -1,
                dtype=torch.int32,
                device=idx_q_seq.device,
            )

        # bf16 matmul (fast on NPU), upcast to fp32 only for scoring/softmax
        # aggregation — matches vLLM-ascend MiniMax prefill (patch einsum+scores.float()).
        scores = torch.einsum("qhd,kd->qhk", idx_q_seq, idx_k_seq).float()
        padded = num_blocks * block_size
        if padded != seq_len:
            scores = torch.nn.functional.pad(scores, (0, padded - seq_len), value=-1e30)

        key_pos = torch.arange(
            padded, device=idx_q_seq.device, dtype=query_positions.dtype
        )
        valid = (key_pos[None, :] < seq_len) & (
            key_pos[None, :] <= query_positions[:, None]
        )
        scores = scores.masked_fill(~valid[:, None, :], -1e30)

        q_len, num_idx_heads, _ = idx_q_seq.shape
        blocked = scores.view(q_len, num_idx_heads, num_blocks, block_size)
        if self.score_type == "max":
            block_scores = blocked.amax(dim=-1)
        elif self.score_type == "lse":
            block_scores = torch.logsumexp(blocked, dim=-1)
        elif self.score_type == "sum":
            block_scores = blocked.sum(dim=-1)
        elif self.score_type in ("mean", "avg"):
            block_scores = blocked.mean(dim=-1)
        else:
            self._raise_npu_sparse_not_ready(
                "top-k block scoring", f"unsupported score_type={self.score_type!r}"
            )

        actual_topk = min(self.topk_blocks, num_blocks)
        blocks = torch.topk(block_scores, k=actual_topk, dim=-1).indices.to(torch.int32)
        if actual_topk < self.topk_blocks:
            blocks = torch.nn.functional.pad(
                blocks, (0, self.topk_blocks - actual_topk), value=-1
            )
        return self._merge_sparse_blocks(blocks, query_positions, num_blocks)

    def _expand_blocks_to_tokens(
        self, block_indices: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        offsets = torch.arange(
            self.block_size_k, device=block_indices.device, dtype=block_indices.dtype
        )
        token_idx = block_indices[..., None] * self.block_size_k + offsets
        valid = (block_indices[..., None] >= 0) & (token_idx < seq_len)
        token_idx = token_idx.flatten(start_dim=-2)
        return torch.where(
            valid.flatten(start_dim=-2), token_idx, torch.full_like(token_idx, -1)
        )

    @staticmethod
    def _sparse_attention_group(
        q_group: torch.Tensor,
        k_kvhead: torch.Tensor,
        v_kvhead: torch.Tensor,
        token_idx: torch.Tensor,
        query_positions: torch.Tensor,
        seq_len: int,
        scale: float,
    ) -> torch.Tensor:
        q_len, _, head_dim = q_group.shape
        num_selected = token_idx.shape[-1]
        if num_selected == 0 or seq_len == 0:
            return q_group.new_zeros(q_group.shape)

        # Masked-full attention (gather-free). The original gathered each query's
        # selected tokens via index_select — q_len*num_selected scattered rows,
        # ~1.9 GB/layer, ~91% of prefill time on NPU (GatherV3). The QK/PV matmuls
        # are <3% of prefill and NPU matmul is fast, so compute attention over the
        # full contiguous KV and mask non-selected positions to -inf. Numerically
        # identical to the sparse version (masked tokens take zero softmax weight).
        key_pos = torch.arange(seq_len, device=q_group.device, dtype=torch.long)
        causal = key_pos[None, :] <= query_positions[:, None]  # [q_len, seq_len]
        valid_sel = (token_idx >= 0) & (token_idx < seq_len)
        # A trash column at index `seq_len` absorbs invalid (padded -1) entries so
        # plain scatter never clobbers a genuinely selected position (no reduce=).
        sel = torch.zeros((q_len, seq_len + 1), dtype=torch.bool, device=q_group.device)
        safe_idx = torch.where(
            valid_sel, token_idx.long(), torch.full_like(token_idx, seq_len)
        )
        sel.scatter_(1, safe_idx, True)
        keep = sel[:, :seq_len] & causal  # [q_len, seq_len]

        scores = torch.einsum("qhd,kd->qhk", q_group, k_kvhead).float() * scale
        scores = scores.masked_fill(~keep[:, None, :], -1e30)
        probs = torch.softmax(scores, dim=-1).to(v_kvhead.dtype)
        return torch.einsum("qhk,kd->qhd", probs, v_kvhead)

    @staticmethod
    def _index_dense_attention(
        idx_q_seq: torch.Tensor,
        idx_k_seq: torch.Tensor,
        idx_v_seq: torch.Tensor,
        query_positions: torch.Tensor,
        seq_len: int,
        scale: float,
    ) -> torch.Tensor:
        # bf16 matmul (fast on NPU), upcast to fp32 only for scoring/softmax
        # aggregation — matches vLLM-ascend MiniMax prefill (patch einsum+scores.float()).
        scores = torch.einsum("qhd,kd->qhk", idx_q_seq, idx_k_seq).float()
        scores = scores * scale
        key_pos = torch.arange(
            seq_len, device=idx_q_seq.device, dtype=query_positions.dtype
        )
        valid = key_pos[None, :] <= query_positions[:, None]
        scores = scores.masked_fill(~valid[:, None, :], -1e30)
        probs = torch.softmax(scores, dim=-1)
        return torch.einsum("qhk,kd->qhd", probs.to(idx_v_seq.dtype), idx_v_seq)

    def _npu_sparse_seq(
        self,
        q_seq: torch.Tensor,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        idx_q_seq: torch.Tensor,
        idx_k_seq: torch.Tensor,
        idx_v_seq: Optional[torch.Tensor],
        query_positions: torch.Tensor,
        seq_len: int,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if seq_len <= 0:
            idx_o = None if idx_v_seq is None else idx_q_seq.new_zeros(idx_q_seq.shape)
            return idx_o, q_seq.new_zeros(q_seq.shape)

        num_q_heads = q_seq.shape[1]
        num_kv_heads = k_seq.shape[1]
        num_idx_heads = idx_q_seq.shape[1]
        head_dim = q_seq.shape[-1]
        if num_q_heads % num_kv_heads != 0:
            self._raise_npu_sparse_not_ready(
                "main sparse attention",
                f"num_q_heads={num_q_heads} not divisible by num_kv_heads={num_kv_heads}",
            )
        group_size = num_q_heads // num_kv_heads
        if num_idx_heads % num_kv_heads != 0:
            self._raise_npu_sparse_not_ready(
                "main sparse attention",
                f"num_idx_heads={num_idx_heads} not divisible by "
                f"num_kv_heads={num_kv_heads}",
            )
        idx_group_size = num_idx_heads // num_kv_heads

        blocks = self._select_sparse_blocks(
            idx_q_seq, idx_k_seq, query_positions, seq_len
        )
        token_idx = self._expand_blocks_to_tokens(blocks, seq_len)
        num_selected = token_idx.shape[-1]
        if idx_group_size > 1:
            main_token_idx = topk_index_reduce(
                token_idx.view(-1, num_kv_heads, idx_group_size, num_selected), dim=2
            )
        else:
            main_token_idx = token_idx

        main_scale = head_dim**-0.5
        out = q_seq.new_zeros(q_seq.shape)
        for kv_head in range(num_kv_heads):
            q_group = q_seq[:, kv_head * group_size : (kv_head + 1) * group_size, :]
            out[:, kv_head * group_size : (kv_head + 1) * group_size, :] = (
                self._sparse_attention_group(
                    q_group,
                    k_seq[:, kv_head, :],
                    v_seq[:, kv_head, :],
                    main_token_idx[:, kv_head, :],
                    query_positions,
                    seq_len,
                    main_scale,
                )
            )

        idx_out = None
        if idx_v_seq is not None:
            idx_scale = idx_q_seq.shape[-1] ** -0.5
            idx_out = self._index_dense_attention(
                idx_q_seq, idx_k_seq, idx_v_seq, query_positions, seq_len, idx_scale
            )
        return idx_out, out

    def _forward_npu_sparse_prefill(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        idx_q: torch.Tensor,
        idx_k_cache: torch.Tensor,
        idx_v_cache: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        cu_seqlens: torch.Tensor,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
    ):
        k_slots = self._cache_as_slots(k_cache)
        v_slots = self._cache_as_slots(v_cache)
        idx_k_slots = self._cache_as_slots(idx_k_cache)
        idx_v_slots = None if idx_v_cache is None else self._cache_as_slots(idx_v_cache)
        out = q.new_zeros(q.shape)
        idx_out = None if idx_v_slots is None else idx_q.new_zeros(idx_q.shape)

        for batch_id in range(forward_batch.req_pool_indices.shape[0]):
            req_idx = int(forward_batch.req_pool_indices[batch_id].item())
            q_start = int(cu_seqlens[batch_id].item())
            q_end = int(cu_seqlens[batch_id + 1].item())
            if q_end <= q_start:
                continue
            prefix_len = int(prefix_lens[batch_id].item())
            total_len = int(seq_lens[batch_id].item())
            q_len = q_end - q_start
            locs = self.req_to_token[req_idx, :total_len].to(
                device=k_slots.device, dtype=torch.long
            )
            # Fast path: NPU ``index_select`` on the paged KV pool is
            # pathologically slow here (~33 ms/call, ~90% of prefill time).
            # Prefill slots are handed out as a contiguous run by the token
            # pool, so when ``locs`` is contiguous a direct slice (a zero-copy
            # view) replaces the scattered gather and the GatherV3 cost
            # vanishes. Fall back to index_select for fragmented allocations.
            is_contig = total_len <= 1 or bool((locs[1:] - locs[:-1] == 1).all().item())
            if is_contig:
                sl = slice(int(locs[0].item()), int(locs[0].item()) + total_len)
                k_seq = k_slots[sl]
                v_seq = v_slots[sl]
                idx_k_seq = idx_k_slots[sl, 0, :]
                idx_v_seq = None if idx_v_slots is None else idx_v_slots[sl, 0, :]

            else:
                k_seq = k_slots.index_select(0, locs)
                v_seq = v_slots.index_select(0, locs)
                idx_k_seq = idx_k_slots.index_select(0, locs)[:, 0, :]
                idx_v_seq = (
                    None
                    if idx_v_slots is None
                    else idx_v_slots.index_select(0, locs)[:, 0, :]
                )
            query_positions = torch.arange(
                prefix_len,
                prefix_len + q_len,
                device=q.device,
                dtype=torch.long,
            )
            idx_o_seq, o_seq = self._npu_sparse_seq(
                q[q_start:q_end],
                k_seq,
                v_seq,
                idx_q[q_start:q_end],
                idx_k_seq,
                idx_v_seq,
                query_positions,
                total_len,
            )
            out[q_start:q_end] = o_seq
            if idx_out is not None:
                idx_out[q_start:q_end] = idx_o_seq
        return idx_out, out

    def _forward_npu_sparse_decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        idx_q: torch.Tensor,
        idx_k_cache: torch.Tensor,
        idx_v_cache: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ):
        k_slots = self._cache_as_slots(k_cache)
        v_slots = self._cache_as_slots(v_cache)
        idx_k_slots = self._cache_as_slots(idx_k_cache)
        idx_v_slots = None if idx_v_cache is None else self._cache_as_slots(idx_v_cache)
        out = q.new_zeros(q.shape)
        idx_out = None if idx_v_slots is None else idx_q.new_zeros(idx_q.shape)

        for batch_id in range(q.shape[0]):
            req_idx = int(forward_batch.req_pool_indices[batch_id].item())
            total_len = int(forward_batch.seq_lens[batch_id].item())
            locs = self.req_to_token[req_idx, :total_len].to(
                device=k_slots.device, dtype=torch.long
            )
            # Fast path: NPU ``index_select`` on the paged KV pool is
            # pathologically slow here (~33 ms/call, ~90% of prefill time).
            # Prefill slots are handed out as a contiguous run by the token
            # pool, so when ``locs`` is contiguous a direct slice (a zero-copy
            # view) replaces the scattered gather and the GatherV3 cost
            # vanishes. Fall back to index_select for fragmented allocations.
            is_contig = total_len <= 1 or bool((locs[1:] - locs[:-1] == 1).all().item())
            if is_contig:
                sl = slice(int(locs[0].item()), int(locs[0].item()) + total_len)
                k_seq = k_slots[sl]
                v_seq = v_slots[sl]
                idx_k_seq = idx_k_slots[sl, 0, :]
                idx_v_seq = None if idx_v_slots is None else idx_v_slots[sl, 0, :]
            else:
                k_seq = k_slots.index_select(0, locs)
                v_seq = v_slots.index_select(0, locs)
                idx_k_seq = idx_k_slots.index_select(0, locs)[:, 0, :]
                idx_v_seq = (
                    None
                    if idx_v_slots is None
                    else idx_v_slots.index_select(0, locs)[:, 0, :]
                )
            query_positions = torch.tensor(
                [max(total_len - 1, 0)], device=q.device, dtype=torch.long
            )
            idx_o_seq, o_seq = self._npu_sparse_seq(
                q[batch_id : batch_id + 1],
                k_seq,
                v_seq,
                idx_q[batch_id : batch_id + 1],
                idx_k_seq,
                idx_v_seq,
                query_positions,
                total_len,
            )
            out[batch_id : batch_id + 1] = o_seq
            if idx_out is not None:
                idx_out[batch_id : batch_id + 1] = idx_o_seq
        return idx_out, out

    def _forward_npu_triton_decode(
        self,
        q: torch.Tensor,  # [B, num_q_heads, head_dim]
        k_cache: torch.Tensor,  # [num_slots, num_kv_heads, head_dim] (NHD)
        v_cache: torch.Tensor,  # [num_slots, num_kv_heads, head_dim]
        idx_q: torch.Tensor,  # [B, num_idx_heads, idx_dim]
        idx_k_cache: torch.Tensor,  # [num_slots, idx_kv_heads, idx_dim]
        idx_v_cache: Optional[
            torch.Tensor
        ],  # [num_slots, idx_kv_heads, idx_dim] or None
        forward_batch: ForwardBatch,
    ):
        """NPU decode via the ported vLLM-ascend triton kernels (BNSD paged).

        sglang's NHD paged KV ([slots,H,D], page_size==block_size) reshapes
        directly to the kernels' [pages,block_size,H,D] layout; the block table
        is derived from req_to_token.
        """
        from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton.flash_block_score_decode import (
            flash_decode_bnsd_with_topk_idx,
        )
        from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton.topk_sparse_decode import (
            flash_decode_bnsd_with_gqa_share_sparse,
        )
        from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton.merge_topk_blocks import (
            _merge_topk_local_bnsd,
        )

        page_size = self.page_size  # == block_size_k
        num_q_heads = q.shape[1]
        head_dim = q.shape[2]
        num_idx_heads = idx_q.shape[1]
        idx_dim = idx_q.shape[2]
        import os as _os

        # k_cache layout: NHD slot-major [slots, head_num, head_dim] OR already
        # paged 4D [pages, page_size, head_num, head_dim]. Handle both.
        if k_cache.dim() == 4:
            num_pages, _ps, num_kv_heads, head_dim = k_cache.shape
            k_bnsd = k_cache
            v_bnsd = v_cache
        else:
            num_kv_heads = k_cache.shape[1]
            head_dim = k_cache.shape[2]
            num_pages = k_cache.shape[0] // page_size
            k_bnsd = k_cache.view(num_pages, page_size, num_kv_heads, head_dim)
            v_bnsd = v_cache.view(num_pages, page_size, num_kv_heads, head_dim)
        if _os.environ.get("MINIMAX_NPU_TRITON_DEBUG"):
            print(
                f"[DEBUG triton-decode] q={tuple(q.shape)} k_cache={tuple(k_cache.shape)} "
                f"dim={k_cache.dim()} -> k_bnsd={tuple(k_bnsd.shape)} "
                f"idx_q={tuple(idx_q.shape)} idx_k={tuple(idx_k_cache.shape)} dim={idx_k_cache.dim()} "
                f"idx_v={None if idx_v_cache is None else tuple(idx_v_cache.shape)} "
                f"page_size={page_size} num_kv_heads={num_kv_heads} head_dim={head_dim} "
                f"req_to_token={tuple(self.req_to_token.shape)} "
                f"seq_lens={forward_batch.seq_lens.tolist()}",
                flush=True,
            )

        # index cache -> BNSD
        if idx_k_cache.dim() == 4:
            idx_k_bnsd = idx_k_cache
            idx_v_bnsd = idx_v_cache
        else:
            idx_kv_heads = idx_k_cache.shape[1]
            idx_k_bnsd = idx_k_cache.view(num_pages, page_size, idx_kv_heads, idx_dim)
            idx_v_bnsd = (
                None
                if idx_v_cache is None
                else idx_v_cache.view(num_pages, page_size, idx_kv_heads, idx_dim)
            )

        # block_table[b, blk] = page holding logical block blk of request b.
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        max_seqlen = (
            int(self._max_seqlen_k)
            if self._max_seqlen_k
            else int(seq_lens.max().item())
        )
        max_blocks = (max_seqlen + page_size - 1) // page_size
        req_idx = forward_batch.req_pool_indices.long()
        max_cols = self.req_to_token.shape[1]
        blk_cols = (
            torch.arange(max_blocks, device=q.device, dtype=torch.long) * page_size
        )
        blk_cols = blk_cols.clamp(max=max_cols - 1)
        token_slots = self.req_to_token[req_idx][:, blk_cols]  # [B, max_blocks]
        block_table = (token_slots // page_size).to(torch.int32)

        disable_index_value = idx_v_cache is None

        # 1) indexer: block scoring (idx_k) + index attention (idx_q/k/v) + topk.
        # Pass init_blocks=0, local_blocks=0 on purpose: the ported triton score
        # kernel would otherwise *boost* the forced init/local blocks to 1e30/1e29
        # and let them take slots INSIDE the top-k budget (sentinel injection), so
        # the local block displaces the k-th real block -> only `topk` blocks
        # attended. The validated pure-PyTorch path instead selects top-k purely by
        # score and APPENDS init/local on top (concat + dedup, see
        # _merge_sparse_blocks), attending to topk+init+local blocks. We select the
        # pure top-k here and re-append the forced blocks below so the triton path
        # attends to the identical block set as the PyTorch path. Mismatched, this
        # diverges ~7% per sparse layer (one dropped 128-token block) and produces
        # different/garbled decode output under greedy decoding.
        idx_o, topk_idx = flash_decode_bnsd_with_topk_idx(
            q=idx_q,
            sink=None,
            k_cache_bnsd=idx_k_bnsd,
            v_cache_bnsd=idx_v_bnsd,
            block_table=block_table,
            seq_lens=seq_lens,
            max_seqlen=max_seqlen,
            block_size=page_size,
            topk=self.topk_blocks,
            init_blocks=0,
            local_blocks=0,
            sm_scale=idx_dim**-0.5,
            score_type=self.score_type,
            disable_index_value=disable_index_value,
        )

        # 2) Reduce heads and append forced blocks. MiniMax-M3 TP=16 uses the
        # direct NPU local-block path; all other layouts use the generic fallback.
        topk_idx = self._prepare_npu_triton_topk_idx(
            topk_idx, seq_lens, num_idx_heads, num_kv_heads, max_blocks
        )

        # 4) main sparse attention over the selected blocks
        o = flash_decode_bnsd_with_gqa_share_sparse(
            q=q,
            sink=None,
            k_cache_bnsd=k_bnsd,
            v_cache_bnsd=v_bnsd,
            block_table=block_table,
            seq_lens=seq_lens,
            block_size=page_size,
            topk_idx=topk_idx,
            sm_scale=head_dim**-0.5,
        )

        return idx_o, o

    def _forward_npu_triton_verify(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        idx_q: torch.Tensor,
        idx_k_cache: torch.Tensor,
        idx_v_cache: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        prefix_lens: torch.Tensor,
    ):
        """Capture-safe sparse attention for TARGET_VERIFY.

        ``_forward_npu_sparse_prefill`` is a per-batch Python loop driven by
        ``.item()`` host-syncs, which CANN forbids under cuda-graph capture
        (``AclrtSynchronizeStreamWithTimeout`` 107027, "Not allow to synchronize
        captured-stream"). DECODE avoids this by using the vectorized
        ``_forward_npu_triton_decode`` triton kernels (no host-sync). TARGET_VERIFY
        has ``speculative_num_draft_tokens`` (ndt) queries per request, each a
        CAUSAL step: query j attends to KV[0 : prefix + j + 1].

        The triton decode kernels are already per-query (``seq_lens`` and
        ``block_table`` are ``[batch_size]``/``[batch_size, max_blocks]`` device
        tensors). So we flatten the ndt verify tokens of each request to
        independent per-query rows and reuse those exact kernels with per-query
        CAUSAL seq_lens. Every per-query quantity is built with device ops
        (``repeat_interleave``, broadcast-add, gather) — no ``.item()`` in the
        hot path, so the whole forward is cuda-graph capturable.

        q arrives already flattened as [bs*ndt, num_q_heads, head_dim] in
        cu_seqlens (request) order, matching per-query prefix_lens+j+1.
        """
        from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton.flash_block_score_decode import (
            flash_decode_bnsd_with_topk_idx,
        )
        from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton.topk_sparse_decode import (
            flash_decode_bnsd_with_gqa_share_sparse,
        )
        from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton.merge_topk_blocks import (
            _merge_topk_local_bnsd,
        )

        page_size = self.page_size  # == block_size_k
        num_q_heads = q.shape[1]
        head_dim = q.shape[2]
        num_idx_heads = idx_q.shape[1]
        idx_dim = idx_q.shape[2]
        num_tokens = q.shape[0]
        bs = forward_batch.seq_lens.shape[0]
        ndt = num_tokens // max(bs, 1)

        # k_cache layout: NHD slot-major [slots, head_num, head_dim] OR already
        # paged 4D [pages, page_size, head_num, head_dim]. Handle both (mirrors
        # _forward_npu_triton_decode).
        if k_cache.dim() == 4:
            num_pages, _ps, num_kv_heads, head_dim = k_cache.shape
            k_bnsd = k_cache
            v_bnsd = v_cache
        else:
            num_kv_heads = k_cache.shape[1]
            head_dim = k_cache.shape[2]
            num_pages = k_cache.shape[0] // page_size
            k_bnsd = k_cache.view(num_pages, page_size, num_kv_heads, head_dim)
            v_bnsd = v_cache.view(num_pages, page_size, num_kv_heads, head_dim)

        # index cache -> BNSD
        if idx_k_cache.dim() == 4:
            idx_k_bnsd = idx_k_cache
            idx_v_bnsd = idx_v_cache
        else:
            idx_kv_heads = idx_k_cache.shape[1]
            idx_k_bnsd = idx_k_cache.view(num_pages, page_size, idx_kv_heads, idx_dim)
            idx_v_bnsd = (
                None
                if idx_v_cache is None
                else idx_v_cache.view(num_pages, page_size, idx_kv_heads, idx_dim)
            )

        # Per-query CAUSAL seq_lens + req_pool_indices (device ops, no host-sync).
        # forward_batch.seq_lens[req] = prefix + ndt (set by the TARGET_VERIFY
        # branch of forward_extend); prefix = seq_lens - ndt. Query j of a request
        # (0-indexed) sits at sequence position prefix + j and causally attends to
        # KV[0 : prefix + j + 1], so its seq_len = prefix + j + 1. repeat_interleave
        # maps each request's ndt flattened queries back to its req_pool_indices.
        prefix = (forward_batch.seq_lens.to(torch.long) - int(ndt)).clamp(min=0)
        offsets = torch.arange(
            1, int(ndt) + 1, device=q.device, dtype=torch.long
        )  # [ndt] = 1..ndt
        per_query_seq_lens = (
            (prefix.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1).to(torch.int32)
        )  # [bs*ndt]
        per_query_req = forward_batch.req_pool_indices.long().repeat_interleave(
            int(ndt)
        )  # [bs*ndt]

        # block_table[b, blk] = page holding logical block blk of query b's request.
        # ``max_seqlen`` comes from the capture-safe ``_max_seqlen_k`` (host-derived
        # in init_forward_metadata_out_graph) so no device->host sync here.
        max_seqlen = (
            int(self._max_seqlen_k)
            if self._max_seqlen_k
            else int(per_query_seq_lens.max().item())
        )
        max_blocks = (max_seqlen + page_size - 1) // page_size
        max_cols = self.req_to_token.shape[1]
        blk_cols = (
            torch.arange(max_blocks, device=q.device, dtype=torch.long) * page_size
        )
        blk_cols = blk_cols.clamp(max=max_cols - 1)
        token_slots = self.req_to_token[per_query_req][
            :, blk_cols
        ]  # [bs*ndt, max_blocks]
        block_table = (token_slots // page_size).to(torch.int32)
        # Sanitize block_table: short-prefix verify queries read req_to_token slots beyond
        # their real KV length, which may hold stale/garbage page ids from pool reuse. Clamp
        # every page id into [0, num_pages) so the indexer/main kernels can never OOB on
        # k_cache / idx_k_cache via a garbage block_table entry. Capture uses benign dummy
        # data so it passes; replay uses real routing — this guards the replay path. Real
        # page ids are already in-range, so correctness is unchanged.
        block_table = block_table.clamp(min=0, max=num_pages - 1)

        disable_index_value = idx_v_cache is None

        # 1) indexer: block scoring (idx_k) + index attention (idx_q/k/v) + topk.
        # init_blocks=0, local_blocks=0 (see _forward_npu_triton_decode): select
        # pure top-k, then re-append forced blocks below so we attend to the
        # identical block set as the validated PyTorch prefill path.
        idx_o, topk_idx = flash_decode_bnsd_with_topk_idx(
            q=idx_q,
            sink=None,
            k_cache_bnsd=idx_k_bnsd,
            v_cache_bnsd=idx_v_bnsd,
            block_table=block_table,
            seq_lens=per_query_seq_lens,
            max_seqlen=max_seqlen,
            block_size=page_size,
            topk=self.topk_blocks,
            init_blocks=0,
            local_blocks=0,
            sm_scale=idx_dim**-0.5,
            score_type=self.score_type,
            disable_index_value=disable_index_value,
        )

        # 2) Reduce heads and append forced blocks in the GQA kernel layout.
        topk_idx = self._prepare_npu_triton_topk_idx(
            topk_idx,
            per_query_seq_lens,
            num_idx_heads,
            num_kv_heads,
            max_blocks,
        )
        # Guard the merged top-k: clamp into valid logical-block range [-1, max_blocks-1].
        # -1 is the skip sentinel; real blocks are >= 0. Prevents the main kernel from
        # indexing block_table past dim-1 if _merge_sparse_blocks appended a local/init block
        # beyond the query's real KV extent (combined with the block_table clamp above, no
        # kernel can OOB on replay).
        topk_idx = topk_idx.clamp(min=-1, max=max_blocks - 1).to(torch.int32)

        # 4) main sparse attention over the selected blocks
        o = flash_decode_bnsd_with_gqa_share_sparse(
            q=q,
            sink=None,
            k_cache_bnsd=k_bnsd,
            v_cache_bnsd=v_bnsd,
            block_table=block_table,
            seq_lens=per_query_seq_lens,
            block_size=page_size,
            topk_idx=topk_idx,
            sm_scale=head_dim**-0.5,
        )
        return idx_o, o

    def _build_prefill_meta(
        self,
        forward_batch: ForwardBatch,
        cu_seqlens: torch.Tensor,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        device,
        page_size: int,
        num_pages: int,
        total_q: int,
    ) -> SimpleNamespace:
        """Build the layer-INVARIANT prefill flatten + main block_table ONCE per
        forward pass (cached on ``self._prefill_meta``, invalidated by
        ``init_forward_metadata_out_graph``).

        These tensors depend only on the batch shape (cu_seqlens / seq_lens /
        prefix_lens / req_pool_indices) and the read-only ``req_to_token`` mapping,
        which are identical across all 57 sparse layers of one forward pass
        (KV *values* are written per-layer into already-assigned slots, but the
        (req, logical_pos) -> slot mapping is fixed for the whole pass). Previously
        ``_forward_npu_triton_prefill`` rebuilt them every layer (57x redundant
        repeat_interleave + req_to_token gather). Ops copied verbatim from the
        former per-layer inline build so the cached tensors are byte-identical.

        Scope note (scaling_analysis §13.2): this hoist covers the MAIN-attention
        path only. The score path's ``_build_qblock_mappings`` (its own
        repeat_interleave + gather, every layer) is NOT hoisted here.
        """
        seq_lens_l = seq_lens.to(device=device, dtype=torch.long)
        prefix_lens_l = prefix_lens.to(device=device, dtype=torch.long)
        cu_q = cu_seqlens.to(device=device, dtype=torch.long)
        extend_lens = (seq_lens_l - prefix_lens_l).clamp(min=0)  # [bs]
        per_query_req = forward_batch.req_pool_indices.long().repeat_interleave(
            extend_lens
        )  # [total_q]
        # Query j of request r sits at position prefix_r + j and causally attends to
        # KV[0 : prefix_r + j + 1], so its seq_len = prefix_r + j + 1.
        per_query_prefix = prefix_lens_l.repeat_interleave(extend_lens)  # [total_q]
        per_query_within = torch.arange(
            total_q, device=device, dtype=torch.long
        ) - cu_q[:-1].repeat_interleave(extend_lens)  # 0-indexed within each request
        per_query_seq_lens = (per_query_prefix + per_query_within + 1).to(torch.int32)

        max_seqlen = (
            int(self._max_seqlen_k)
            if self._max_seqlen_k
            else int(per_query_seq_lens.max().item())
        )
        max_blocks = (max_seqlen + page_size - 1) // page_size
        extend_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
        block_size_q = self._get_safe_block_size_q(max_seqlen, extend_lens_cpu)

        # Main-attention block_table [total_q, max_blocks] -- layer-invariant
        # (req_to_token read-only during the forward; num_pages fixed per pass).
        # Advanced-index directly (NOT req_to_token[per_query_req][:, cols], which
        # materializes [total_q, max_context_len] and OOMs). Mirrors the former
        # _decode_main build, incl. the [0, num_pages) sanitize clamp.
        max_cols = self.req_to_token.shape[1]
        blk_cols_f = torch.arange(max_blocks, device=device, dtype=torch.long) * page_size
        blk_cols_f = blk_cols_f.clamp(max=max_cols - 1)
        token_slots_f = self.req_to_token[per_query_req[:, None], blk_cols_f]
        block_table_f = (token_slots_f // page_size).to(torch.int32).clamp(
            min=0, max=num_pages - 1
        )

        # Score-path qblock mappings (layer-invariant: same req_to_token + batch
        # shape). Built once here and passed into flash_prefill_bnsd_score(_attn)
        # to skip the per-layer _build_qblock_mappings rebuild (its 4x
        # repeat_interleave + req_to_token gather + cumsum, every layer -- see
        # scaling_analysis §13.2). max_blocks == score path's max_seqblock_k for
        # prefill (_max_seqlen_k == seq_lens.max()).
        from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton.prefill_block_score import (
            _build_qblock_mappings as _build_score_qblock_mappings,
        )
        qblock_mappings = _build_score_qblock_mappings(
            cu_seqlens,
            seq_lens,
            self.req_to_token,
            forward_batch.req_pool_indices,
            block_size_q,
            page_size,
            max_blocks,
            device,
        )

        # NOTE: per_query_req is a LOCAL here (used to build block_table_f above);
        # it is NOT returned -- the per-layer path reads only the 6 fields below
        # (see scaling_analysis §13.7 for the cache field reference).
        return SimpleNamespace(
            per_query_seq_lens=per_query_seq_lens,
            max_seqlen=max_seqlen,
            max_blocks=max_blocks,
            block_size_q=block_size_q,
            block_table_f=block_table_f,
            qblock_mappings=qblock_mappings,
        )

    def _forward_npu_triton_prefill(
        self,
        q: torch.Tensor,  # [total_extend_tokens, num_q_heads, head_dim]
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        idx_q: torch.Tensor,  # [total_extend_tokens, num_idx_heads, idx_dim]
        idx_k_cache: torch.Tensor,
        idx_v_cache: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        cu_seqlens: torch.Tensor,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        # Prefill main-attention (decode-kernel reused) launch tuning. Defaults are
        # the decode-validated (4, 2); prefill is compute/memory-bound (total_q~8192)
        # so re-tuning may help. A/B by editing these two -- the decode path
        # (forward_decode) is unaffected (it does not go through this method).
        # Sweep: num_warps in {2,4,8}, num_stages in {2,3}.
        main_num_warps: int = 4,
        main_num_stages: int = 2,
    ):
        """NPU block-sparse PREFILL via the ported triton decode kernels.

        Generalizes ``_forward_npu_triton_verify`` from a uniform ``ndt`` draft
        tokens per request to *variable* per-request extend lengths. Every extend
        token is flattened to an independent per-query row with a CAUSAL seq_len
        (query j of request r sits at position ``prefix_r + j`` and attends to
        ``KV[0 : prefix_r + j + 1]``), then the same decode kernels score/attend
        block-sparse over only the selected ``topk+init+local`` blocks. This
        replaces ``_forward_npu_sparse_prefill``'s full-seq_len QK + masked_fill
        (compute-equivalent to dense) with true block-sparse attention.

        Per-query quantities are built with device ops (``repeat_interleave``,
        broadcast-add). Extend/prefill runs eager (not cuda-graph captured), so
        ``.item()`` is tolerable, but device ops are kept for speed.
        """
        from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton.flash_block_score_decode import (
            flash_decode_bnsd_with_topk_idx,
        )
        from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton.topk_sparse_decode import (
            flash_decode_bnsd_with_gqa_share_sparse,
        )
        from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton.merge_topk_blocks import (
            _merge_topk_local_bnsd,
        )

        page_size = self.page_size  # == block_size_k
        num_q_heads = q.shape[1]
        head_dim = q.shape[2]
        num_idx_heads = idx_q.shape[1]
        idx_dim = idx_q.shape[2]
        total_q = q.shape[0]

        # k_cache layout: NHD slot-major [slots, head_num, head_dim] OR already
        # paged 4D [pages, page_size, head_num, head_dim]. Handle both (mirrors
        # _forward_npu_triton_decode/verify).
        if k_cache.dim() == 4:
            num_pages, _ps, num_kv_heads, head_dim = k_cache.shape
            k_bnsd = k_cache
            v_bnsd = v_cache
        else:
            num_kv_heads = k_cache.shape[1]
            head_dim = k_cache.shape[2]
            num_pages = k_cache.shape[0] // page_size
            k_bnsd = k_cache.view(num_pages, page_size, num_kv_heads, head_dim)
            v_bnsd = v_cache.view(num_pages, page_size, num_kv_heads, head_dim)

        # index cache -> BNSD
        if idx_k_cache.dim() == 4:
            idx_k_bnsd = idx_k_cache
            idx_v_bnsd = idx_v_cache
        else:
            idx_kv_heads = idx_k_cache.shape[1]
            idx_k_bnsd = idx_k_cache.view(num_pages, page_size, idx_kv_heads, idx_dim)
            idx_v_bnsd = (
                None
                if idx_v_cache is None
                else idx_v_cache.view(num_pages, page_size, idx_kv_heads, idx_dim)
            )

        # Layer-invariant flatten + block_table: built ONCE per forward pass and
        # cached on self._prefill_meta (see _build_prefill_meta). The first sparse
        # layer builds it; the other ~56 reuse the cached tensors -- previously
        # rebuilt every layer (see scaling_analysis §13.2). Invalidated each
        # forward by init_forward_metadata_out_graph.
        meta = self._prefill_meta
        if meta is None:
            meta = self._build_prefill_meta(
                forward_batch,
                cu_seqlens,
                seq_lens,
                prefix_lens,
                q.device,
                page_size,
                num_pages,
                total_q,
            )
            self._prefill_meta = meta
        per_query_seq_lens = meta.per_query_seq_lens
        max_seqlen = meta.max_seqlen
        max_blocks = meta.max_blocks
        block_size_q = meta.block_size_q
        block_table_f = meta.block_table_f  # [total_q, max_blocks], cached
        if not getattr(self, "_prefill_diag_logged", False):
            self._prefill_diag_logged = True
            logger.warning(
                "[MiniMax/NPU triton-prefill] max_seqlen=%d max_blocks=%d "
                "total_q=%d page=%d _max_seqlen_k=%d max_context_len=%d",
                max_seqlen,
                max_blocks,
                total_q,
                page_size,
                self._max_seqlen_k,
                self.max_context_len,
            )

        disable_index_value = idx_v_cache is None

        # 1) indexer: block scoring (idx_k) + index attention (idx_q/k/v) + topk.
        # init_blocks=0, local_blocks=0 (see _forward_npu_triton_decode): select
        # pure top-k, then re-append forced blocks below so we attend to the
        # identical block set as the validated PyTorch prefill path.
        # Approach A: batched varlen multi-token indexer -- tiles queries into
        # blocks of block_size_q and scores every query-block x kv-block in one 2D
        # dot (no per-query launch overhead, the decode-kernel failure mode).
        from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton.prefill_block_score import (
            flash_prefill_bnsd_indexer,
            flash_prefill_bnsd_with_topk_idx as _flash_prefill_score_topk,
        )
        if disable_index_value:
            idx_o = None
            topk_idx = _flash_prefill_score_topk(
                idx_q, idx_k_bnsd, cu_seqlens, seq_lens,
                self.req_to_token, forward_batch.req_pool_indices,
                block_size_q, page_size, self.topk_blocks,
                idx_dim**-0.5, self.score_type,
                qblock_mappings=meta.qblock_mappings,
            )
        else:
            idx_o, topk_idx = flash_prefill_bnsd_indexer(
                idx_q, idx_k_bnsd, idx_v_bnsd, cu_seqlens, seq_lens,
                self.req_to_token, forward_batch.req_pool_indices,
                block_size_q, page_size, self.topk_blocks,
                idx_dim**-0.5, self.score_type,
                qblock_mappings=meta.qblock_mappings,
            )

        # 2) Reduce heads and append forced blocks in the GQA kernel layout.
        topk_idx = self._prepare_npu_triton_topk_idx(
            topk_idx,
            per_query_seq_lens,
            num_idx_heads,
            num_kv_heads,
            max_blocks,
        )
        # Guard the merged top-k: clamp into valid logical-block range [-1, max_blocks-1].
        # -1 is the skip sentinel; real blocks are >= 0. Prevents the main kernel from
        # indexing block_table_f past its column dim if _merge_sparse_blocks appended a
        # local/init block beyond the query's real KV extent. Mirrors the verify path
        # (the clamp at the verify main). Real indices already in-range -> correctness
        # unchanged; this only sanitizes out-of-range entries.
        topk_idx = topk_idx.clamp(min=-1, max=max_blocks - 1).to(torch.int32)

        # 4) main sparse attention over the selected blocks: per-query decode-main
        # (flatten total_q extend tokens into total_q batch rows). The union-tile
        # kernel was an A/B-verified 1.71x deopt at 64K and has been removed.
        def _decode_main():
            # block_table_f [total_q, max_blocks] + per_query_seq_lens are the
            # cached layer-invariant tensors (built once per forward pass in
            # _build_prefill_meta). No per-layer req_to_token gather here.
            return flash_decode_bnsd_with_gqa_share_sparse(
                q=q,
                sink=None,
                k_cache_bnsd=k_bnsd,
                v_cache_bnsd=v_bnsd,
                block_table=block_table_f,
                seq_lens=per_query_seq_lens,
                block_size=page_size,
                topk_idx=topk_idx,
                sm_scale=head_dim**-0.5,
                num_warps=main_num_warps,
                num_stages=main_num_stages,
            )

        o = _decode_main()

        return idx_o, o

    @staticmethod
    def _is_sparse_kv_cached_by_fusion(
        forward_batch: ForwardBatch, layer_id: int
    ) -> bool:
        layer_ids = forward_batch.minimax_m3_precached_sparse_layers
        return layer_ids is not None and layer_id in layer_ids

    def forward(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if forward_batch.forward_mode.is_idle():
            idx_q = kwargs.get("idx_q")
            num_idx_heads = idx_q.shape[1]
            disable_value = layer.layer_id in self.disable_value_layer_ids
            idx_out: Optional[torch.Tensor] = (
                None
                if disable_value
                else q.new_zeros(q.shape[0], num_idx_heads * self.idx_head_dim)
            )
            out = q.new_zeros(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
            return idx_out, out
        else:
            return super().forward(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        *,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
    ):
        disable_value = layer.layer_id in self.disable_value_layer_ids
        kv_cached_by_fusion = self._is_sparse_kv_cached_by_fusion(
            forward_batch, layer.layer_id
        )
        if not kv_cached_by_fusion:
            self.kv_pool.set_fused_kv_index_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                idx_k,
                None if disable_value else idx_v,
            )
        k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
        if disable_value:
            idx_k_cache = self.kv_pool.get_index_k_buffer(layer.layer_id)
            idx_v_cache = None
        else:
            idx_k_cache, idx_v_cache = self.kv_pool.get_index_kv_buffer(layer.layer_id)

        if forward_batch.extend_seq_lens is None:
            # TARGET_VERIFY: ForwardBatch leaves extend_seq_lens(_cpu) None (it
            # only populates seq_lens = prefix + draft); each sequence verifies
            # `speculative_num_draft_tokens` draft tokens (see ascend_backend
            # forward_metadata). Reconstruct per-seq extend lengths so the sparse
            # prefill kernel receives correct cu_seqlens instead of crashing on
            # the None .device access.
            _bs = forward_batch.seq_lens.shape[0]
            _ndt = self.speculative_num_draft_tokens or (q.shape[0] // max(_bs, 1))
            forward_batch.extend_seq_lens = torch.full(
                (_bs,),
                int(_ndt),
                dtype=torch.int32,
                device=forward_batch.seq_lens.device,
            )
            forward_batch.extend_seq_lens_cpu = [int(_ndt)] * _bs
            # For TARGET_VERIFY, seq_lens = prefix + ndt (draft tokens are added to
            # KV during verify); the cached prefix per seq is seq_lens - ndt. The
            # default ``prefix_lens = 0`` branch below is wrong for verify and makes
            # the sparse block selection / positions read garbage, so materialize
            # extend_prefix_lens here.
            if forward_batch.extend_prefix_lens is None:
                forward_batch.extend_prefix_lens = (
                    forward_batch.seq_lens.to(torch.int32) - int(_ndt)
                ).clamp(min=0)

        cu_seqlens = torch.cat(
            [
                torch.zeros(
                    1, dtype=torch.int32, device=forward_batch.extend_seq_lens.device
                ),
                forward_batch.extend_seq_lens.to(torch.int32).cumsum(0).to(torch.int32),
            ]
        )
        seq_lens = forward_batch.seq_lens.to(torch.int32)  # prefix + extend
        if forward_batch.extend_prefix_lens is not None:
            prefix_lens = forward_batch.extend_prefix_lens.to(torch.int32)
        else:
            prefix_lens = torch.zeros_like(seq_lens)

        # In DP attention mode, q may be padded beyond the actual token count
        # for collective communication alignment. Trim to actual tokens so
        # the sparse attention kernel sees consistent shapes.
        #
        # Source the token count from CPU-side metadata when available so we do
        # not force a GPU->CPU sync (cu_seqlens[-1].item()) on every sparse
        # layer of every prefill. extend_seq_lens_cpu is a plain list of ints
        # (ForwardBatch sets it from extend_seq_lens.cpu()), so sum() is a host
        # op and the result is identical to cu_seqlens[-1]. Fall back to the
        # device tensor only when CPU metadata is absent.
        if forward_batch.extend_seq_lens_cpu is not None:
            actual_num_tokens = int(sum(forward_batch.extend_seq_lens_cpu))
        else:
            actual_num_tokens = int(cu_seqlens[-1].item())
        original_num_tokens = q.shape[0]
        if actual_num_tokens < original_num_tokens:
            q = q[:actual_num_tokens]
            idx_q = idx_q[:actual_num_tokens]

        if self.is_npu:
            if (
                _npu_use_triton_sparse()
                and forward_batch.forward_mode.is_target_verify()
            ):
                # TARGET_VERIFY must run under cuda-graph capture, but
                # _forward_npu_sparse_prefill is a per-batch .item() loop that
                # CANN refuses to capture (107027). Route to the capture-safe
                # triton verify path (reuses the decode kernels with per-query
                # causal seq_lens). Prefill (non-verify extend) still uses the
                # PyTorch prefill kernel.
                idx_o_t, o_t = self._forward_npu_triton_verify(
                    q,
                    k_cache,
                    v_cache,
                    idx_q,
                    idx_k_cache,
                    idx_v_cache,
                    forward_batch,
                    prefix_lens,
                )
                idx_o, o = idx_o_t, o_t
            elif _npu_use_triton_prefill() or _npu_triton_prefill_auto(seq_lens):
                # True prefill (extend, non-verify): block-sparse triton path that
                # reuses the decode kernels via per-query flattening
                # (_forward_npu_triton_prefill) -- attends only to the selected
                # topk+init+local blocks instead of computing QK over the full
                # seq_len like _forward_npu_sparse_prefill. Enabled when EITHER the
                # explicit SGLANG_MINIMAX_NPU_TRITON_PREFILL flag is set OR adaptively
                # when max KV length >= MINIMAX_NPU_TRITON_PREFILL_AUTO_MIN_SEQLEN
                # (long context: sparse wins; short context stays PyTorch via the
                # else branch, avoiding the sub-crossover regression).
                idx_o_t, o_t = self._forward_npu_triton_prefill(
                    q,
                    k_cache,
                    v_cache,
                    idx_q,
                    idx_k_cache,
                    idx_v_cache,
                    forward_batch,
                    cu_seqlens,
                    seq_lens,
                    prefix_lens,
                )
                idx_o, o = idx_o_t, o_t
            else:
                idx_o, o = self._forward_npu_sparse_prefill(
                    q,
                    k_cache,
                    v_cache,
                    idx_q,
                    idx_k_cache,
                    idx_v_cache,
                    forward_batch,
                    cu_seqlens,
                    seq_lens,
                    prefix_lens,
                )
        else:
            idx_o, o = minimax_sparse_prefill(
                q,
                k_cache,
                v_cache,
                None,
                idx_q,
                idx_k_cache,
                idx_v_cache,
                None,
                self.req_to_token,
                forward_batch.req_pool_indices,
                cu_seqlens,
                seq_lens,
                prefix_lens,
                self._max_seqlen_q,
                self._max_seqlen_k,
                self.block_size_q,
                self.block_size_k,
                self.topk_blocks,
                self.init_blocks,
                self.local_blocks,
                score_type=self.score_type,
                disable_index_value=disable_value,
                use_msa=self.use_msa,
                # Host seq-lens let get_cu_seqblocks avoid a per-layer .item() sync.
                seqlens_cpu=forward_batch.extend_seq_lens_cpu,
            )

        # Pad output back to original size for DP communication
        if actual_num_tokens < original_num_tokens:
            pad_len = original_num_tokens - actual_num_tokens
            o = torch.cat([o, o.new_zeros(pad_len, *o.shape[1:])], dim=0)
            if idx_o is not None:
                idx_o = torch.cat(
                    [idx_o, idx_o.new_zeros(pad_len, *idx_o.shape[1:])], dim=0
                )

        return (
            (
                None
                if idx_o is None
                else idx_o.reshape(original_num_tokens, -1).contiguous()
            ),
            o.reshape(original_num_tokens, -1).contiguous(),
        )

    def _dense_sparse_main_decode(
        self,
        q: torch.Tensor,  # [bs, num_q_heads, head_dim]
        page_table: torch.Tensor,  # [bs, max_sparse_pages] int32 (from the indexer)
        real_seq_lens: torch.Tensor,  # [bs] int32, effective KV length per query
        k_cache: torch.Tensor,  # [max_slots, 1, head_dim]
        v_cache: torch.Tensor,  # [max_slots, 1, head_dim]
        layer,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

        if isinstance(self.dense_backend, TRTLLMHAAttnBackend):
            import flashinfer

            ps = self.page_size
            nkv = 1
            head_dim = q.size(-1)
            # [max_slots, nkv, D] -> [num_pages, page_size, nkv, D]
            #                     -> [num_pages, nkv, page_size, D] (HND, trtllm default)
            kc = k_cache.view(-1, ps, nkv, head_dim).permute(0, 2, 1, 3)
            vc = v_cache.view(-1, ps, nkv, head_dim).permute(0, 2, 1, 3)
            return flashinfer.decode.trtllm_batch_decode_with_kv_cache(  # type: ignore
                query=q.contiguous(),
                kv_cache=(kc, vc),
                workspace_buffer=self.dense_backend.workspace_buffer,
                block_tables=page_table,
                seq_lens=real_seq_lens,
                max_seq_len=self.topk_blocks * self.block_size_k,
                bmm1_scale=layer.scaling,
                bmm2_scale=1.0,
            )
        raise NotImplementedError(
            "dense sparse decode currently supports trtllm_mha only (fa3 is TODO)"
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        *,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
        **kwargs,
    ):
        assert len(kwargs) == 0
        disable_value = layer.layer_id in self.disable_value_layer_ids
        self.kv_pool.set_fused_kv_index_buffer(
            layer,
            forward_batch.out_cache_loc,
            k,
            v,
            idx_k,
            None if disable_value else idx_v,
        )
        k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
        if disable_value:
            idx_k_cache = self.kv_pool.get_index_k_buffer(layer.layer_id)
            idx_v_cache = None
        else:
            idx_k_cache, idx_v_cache = self.kv_pool.get_index_kv_buffer(layer.layer_id)

        attn_fn = None
        if self.use_dense_sparse_decode and k_cache.shape[1] == 1:

            def attn_fn(main_q, page_table, real_seq_lens):
                return self._dense_sparse_main_decode(
                    main_q,
                    page_table,
                    real_seq_lens,
                    k_cache,
                    v_cache,
                    layer,
                    forward_batch,
                )

        # The MSA decode page table + plan are built once per forward in
        # init_forward_metadata_out_graph (eager, outside graph capture) and shared
        # across all sparse layers; here we just consume the cached metadata.
        msa_kv_indices = msa_plan = None
        if self._use_msa_decode and attn_fn is None:
            if self._msa_dec_meta is not None:
                msa_kv_indices, msa_plan = self._msa_dec_meta
            elif q.shape[0] > 0:
                # Rebuilding the plan inline would run host-side code inside
                # CUDA-graph capture; fail loudly instead.
                raise RuntimeError(
                    "MSA decode metadata missing: init_forward_metadata_out_graph "
                    "did not prepare the plan for this forward (gate mismatch)."
                )

        if self.is_npu:
            if _npu_use_triton_sparse():
                idx_o, o = self._forward_npu_triton_decode(
                    q,
                    k_cache,
                    v_cache,
                    idx_q,
                    idx_k_cache,
                    idx_v_cache,
                    forward_batch,
                )
            else:
                idx_o, o = self._forward_npu_sparse_decode(
                    q,
                    k_cache,
                    v_cache,
                    idx_q,
                    idx_k_cache,
                    idx_v_cache,
                    forward_batch,
                )
        else:
            idx_o, o = minimax_sparse_decode(
                q,
                None,
                k_cache,
                v_cache,
                idx_q,
                None,
                idx_k_cache,
                idx_v_cache,
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                self._max_seqlen_k,
                1,
                self.block_size_k,
                self.topk_blocks,
                self.init_blocks,
                self.local_blocks,
                score_type=self.score_type,
                disable_index_value=disable_value,
                dense_main_attn_fn=attn_fn,
                page_size=self.page_size,
                use_msa=self._use_msa_decode,
                msa_kv_indices=msa_kv_indices,
                msa_plan=msa_plan,
            )
        return (
            None if idx_o is None else idx_o.reshape(q.shape[0], -1).contiguous(),
            o.reshape(q.shape[0], -1).contiguous(),
        )


class MiniMaxHybridAttnBackend(AttentionBackend):
    """Combines a dense backend and a sparse backend, routing by call site."""

    def __init__(
        self,
        dense_backend: AttentionBackend,
        sparse_backend: MiniMaxSparseAttnBackend,
        sparse_layer_ids: list[int],
    ):
        self.dense = dense_backend
        self.sparse = sparse_backend
        self.sparse_layer_ids = sparse_layer_ids
        # Let the sparse decode reuse the dense paged backend (page table + workspace).
        self.sparse.dense_backend = dense_backend

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # delegate so the dense (FlashInfer) backend keeps its own eager init.
        self.sparse.init_forward_metadata(forward_batch)
        self.dense.init_forward_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        self.sparse.init_forward_metadata_out_graph(forward_batch, in_capture)
        self.dense.init_forward_metadata_out_graph(forward_batch, in_capture)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        self.sparse.init_forward_metadata_in_graph(forward_batch)
        self.dense.init_forward_metadata_in_graph(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.dense.init_cuda_graph_state(max_bs, max_num_tokens)
        self.sparse.init_cuda_graph_state(max_bs, max_num_tokens)

    def get_cuda_graph_seq_len_fill_value(self):
        return self.sparse.get_cuda_graph_seq_len_fill_value()

    def get_verify_buffers_to_fill_after_draft(self):
        # EAGLE3 verify buffer interface: the dense (ascend) backend owns the
        # tree-mask/position buffers consumed by the verify forward. The base
        # AttentionBackend raises NotImplementedError, so delegate to dense.
        return self.dense.get_verify_buffers_to_fill_after_draft()

    def update_verify_buffers_to_fill_after_draft(self, spec_info, cuda_graph_bs=None):
        return self.dense.update_verify_buffers_to_fill_after_draft(
            spec_info, cuda_graph_bs
        )

    def forward(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if layer.layer_id in self.sparse_layer_ids:
            return self.sparse.forward(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

        # Dense layers delegate to the stock backend (e.g. flashinfer). Under DP
        # attention the per-rank token block is padded to an even length
        # (prepare_mlp_sync_batch -> ceil_align(num_tokens, attn_cp_size * 2)), but
        # flashinfer builds qo_indptr from extend_seq_lens, so q.shape[0] (padded)
        # != qo_indptr[-1] (real) and the paged-prefill kernel raises. Trim q to
        # the real token count and re-pad the output; k/v stay untrimmed so the
        # KV-cache write stays aligned with out_cache_loc. Prefill-only.
        mode = forward_batch.forward_mode
        if mode.is_extend() and forward_batch.extend_seq_lens_cpu is not None:
            actual_num_tokens = int(sum(forward_batch.extend_seq_lens_cpu))
            original_num_tokens = q.shape[0]
            if actual_num_tokens < original_num_tokens:
                o = self.dense.forward(
                    q[:actual_num_tokens],
                    k,
                    v,
                    layer,
                    forward_batch,
                    save_kv_cache,
                    **kwargs,
                )
                pad_len = original_num_tokens - actual_num_tokens
                return torch.cat([o, o.new_zeros(pad_len, *o.shape[1:])], dim=0)

        return self.dense.forward(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_extend(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if layer.layer_id in self.sparse_layer_ids:
            return self.sparse.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        else:
            return self.dense.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

    def forward_decode(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if layer.layer_id in self.sparse_layer_ids:
            return self.sparse.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        else:
            return self.dense.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
