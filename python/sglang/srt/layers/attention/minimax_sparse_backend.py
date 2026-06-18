from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

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
from sglang.srt.utils import get_bool_env_var, is_npu
from sglang.srt.utils.async_probe import maybe_detect_oob

if not is_npu():
    from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
        minimax_sparse_decode,
        minimax_sparse_prefill,
    )

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
        # assert self.idx_head_dim == head_dim

        # max_seqlen for the current forward pass, stored as a plain Python int
        # so that it is safe to use inside CUDA graphs (no .item() at graph time).
        # Populated by init_forward_metadata* before each forward.
        self._max_seqlen_q: int = 1
        self._max_seqlen_k: int = 1

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

    def _debug_sync_npu(self, tag: str) -> None:
        if not self.is_npu:
            return
        if not get_bool_env_var("SGLANG_MINIMAX_NPU_DEBUG_SYNC", "False"):
            return
        logger.warning("[MiniMax/NPU debug] synchronize before: %s", tag)
        torch.npu.synchronize()
        logger.warning("[MiniMax/NPU debug] synchronize after: %s", tag)

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
        extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_lens is not None:
            self._max_seqlen_q = int(max(extend_lens))
        else:
            self._max_seqlen_q = 1
        if in_capture and forward_batch.forward_mode.is_decode_or_idle():
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

    def _block_topk(self, scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        num_heads = scores.shape[0]
        num_blocks = (seq_len + self.block_size_k - 1) // self.block_size_k
        total_blocks = self.topk_blocks + self.init_blocks + self.local_blocks
        topk_idx = torch.full(
            (num_heads, total_blocks),
            -1,
            dtype=torch.int32,
            device=scores.device,
        )
        if num_blocks == 0:
            return topk_idx

        block_scores = []
        for block_id in range(num_blocks):
            start = block_id * self.block_size_k
            end = min(start + self.block_size_k, seq_len)
            block = scores[:, start:end]
            if self.score_type == "max":
                reduced = block.max(dim=-1).values
            elif self.score_type == "sum":
                reduced = block.sum(dim=-1)
            elif self.score_type in ("mean", "avg"):
                reduced = block.mean(dim=-1)
            else:
                self._raise_npu_sparse_not_ready(
                    "top-k block scoring", f"unsupported score_type={self.score_type!r}"
                )
            block_scores.append(reduced)

        score_tensor = torch.stack(block_scores, dim=-1)

        actual_topk = min(self.topk_blocks, num_blocks)
        _, indices = torch.topk(score_tensor, k=actual_topk, dim=-1)
        topk_idx[:, :actual_topk] = indices.to(topk_idx.dtype)
        if total_blocks == self.topk_blocks:
            return topk_idx

        forced_parts = []
        if self.init_blocks > 0:
            init_blocks = torch.arange(
                min(self.init_blocks, num_blocks),
                dtype=topk_idx.dtype,
                device=scores.device,
            ).expand(num_heads, -1)
            forced_parts.append(init_blocks)
        if self.local_blocks > 0:
            local_start = max(0, num_blocks - self.local_blocks)
            local_blocks = torch.arange(
                local_start,
                num_blocks,
                dtype=topk_idx.dtype,
                device=scores.device,
            ).expand(num_heads, -1)
            forced_parts.append(local_blocks)

        if not forced_parts:
            return topk_idx[:, : self.topk_blocks]

        candidates = torch.cat([topk_idx[:, : self.topk_blocks], *forced_parts], dim=-1)
        valid = (candidates >= 0) & (candidates < num_blocks)
        invalid_value = torch.full_like(candidates, num_blocks)
        sorted_candidates = torch.sort(
            torch.where(valid, candidates, invalid_value), dim=-1
        ).values
        sorted_valid = sorted_candidates < num_blocks
        previous = torch.cat(
            [
                torch.full_like(sorted_candidates[:, :1], -1),
                sorted_candidates[:, :-1],
            ],
            dim=-1,
        )
        keep = sorted_valid & (sorted_candidates != previous)
        sort_idx = torch.argsort((~keep).int(), dim=-1, stable=True)
        packed = torch.gather(sorted_candidates, -1, sort_idx)
        valid_count = keep.sum(dim=-1, keepdim=True)
        idx_range = torch.arange(packed.shape[-1], device=scores.device)
        packed = torch.where(idx_range < valid_count, packed, -1)
        topk_idx[:, : packed.shape[-1]] = packed.to(topk_idx.dtype)
        return topk_idx

    def _token_indices_from_blocks(
        self, block_idx: torch.Tensor, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        chunks = []
        for block in block_idx.tolist():
            if block < 0:
                continue
            start = int(block) * self.block_size_k
            end = min(start + self.block_size_k, seq_len)
            if start < end:
                chunks.append(torch.arange(start, end, device=device))
        if len(chunks) == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.cat(chunks).to(torch.long)

    @staticmethod
    def _dense_attention_heads(
        q_heads: torch.Tensor,
        k_tokens: torch.Tensor,
        v_tokens: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        num_q_heads = q_heads.shape[0]
        num_kv_heads = k_tokens.shape[1]
        group_size = max(1, num_q_heads // num_kv_heads)
        out = q_heads.new_zeros((num_q_heads, v_tokens.shape[-1]))
        if k_tokens.shape[0] == 0:
            return out

        for qh in range(num_q_heads):
            kvh = min(qh // group_size, num_kv_heads - 1)
            scores = torch.matmul(
                q_heads[qh].float(), k_tokens[:, kvh, :].float().transpose(0, 1)
            )
            probs = torch.softmax(scores * scale, dim=-1, dtype=torch.float32)
            out[qh] = torch.matmul(probs.to(v_tokens.dtype), v_tokens[:, kvh, :])
        return out

    def _sparse_attention_heads(
        self,
        q_heads: torch.Tensor,
        k_tokens: torch.Tensor,
        v_tokens: torch.Tensor,
        topk_idx: torch.Tensor,
        seq_len: int,
        scale: float,
    ) -> torch.Tensor:
        num_q_heads = q_heads.shape[0]
        num_kv_heads = k_tokens.shape[1]
        group_size = max(1, num_q_heads // num_kv_heads)
        out = q_heads.new_zeros((num_q_heads, v_tokens.shape[-1]))

        for qh in range(num_q_heads):
            kvh = min(qh // group_size, num_kv_heads - 1)
            token_idx = self._token_indices_from_blocks(
                topk_idx[kvh], seq_len, q_heads.device
            )
            if token_idx.numel() == 0:
                continue
            k_selected = k_tokens.index_select(0, token_idx)[:, kvh, :]
            v_selected = v_tokens.index_select(0, token_idx)[:, kvh, :]
            scores = torch.matmul(
                q_heads[qh].float(), k_selected.float().transpose(0, 1)
            )
            probs = torch.softmax(scores * scale, dim=-1, dtype=torch.float32)
            out[qh] = torch.matmul(probs.to(v_selected.dtype), v_selected)
        return out

    def _npu_sparse_one(
        self,
        q_token: torch.Tensor,
        k_tokens: torch.Tensor,
        v_tokens: torch.Tensor,
        idx_q_token: torch.Tensor,
        idx_k_tokens: torch.Tensor,
        idx_v_tokens: Optional[torch.Tensor],
        seq_len: int,
    ):
        idx_scale = self.idx_head_dim**-0.5
        main_scale = q_token.shape[-1] ** -0.5
        idx_scores = torch.matmul(
            idx_q_token.float(), idx_k_tokens[:, 0, :].float().transpose(0, 1)
        )
        idx_topk = self._block_topk(idx_scores * idx_scale, seq_len)

        idx_o = None
        if idx_v_tokens is not None:
            idx_o = self._dense_attention_heads(
                idx_q_token, idx_k_tokens, idx_v_tokens, idx_scale
            )

        num_idx_heads = idx_q_token.shape[0]
        num_kv_heads = k_tokens.shape[1]
        if num_idx_heads % num_kv_heads != 0:
            self._raise_npu_sparse_not_ready(
                "main sparse attention",
                f"num_idx_heads={num_idx_heads} is not divisible by "
                f"num_kv_heads={num_kv_heads}",
            )
        idx_group_size = num_idx_heads // num_kv_heads
        if idx_group_size > 1:
            main_topk = topk_index_reduce(
                idx_topk.view(num_kv_heads, idx_group_size, -1), dim=1
            )
        else:
            main_topk = idx_topk

        out = self._sparse_attention_heads(
            q_token, k_tokens, v_tokens, main_topk, seq_len, main_scale
        )
        return idx_o, out

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
        idx_v_slots = (
            None if idx_v_cache is None else self._cache_as_slots(idx_v_cache)
        )
        out = q.new_zeros(q.shape)
        idx_out = None if idx_v_slots is None else idx_q.new_zeros(idx_q.shape)

        for batch_id in range(forward_batch.req_pool_indices.shape[0]):
            req_idx = int(forward_batch.req_pool_indices[batch_id].item())
            q_start = int(cu_seqlens[batch_id].item())
            q_end = int(cu_seqlens[batch_id + 1].item())
            prefix_len = int(prefix_lens[batch_id].item())
            total_len = int(seq_lens[batch_id].item())
            for offset, q_pos in enumerate(range(q_start, q_end)):
                kv_len = min(prefix_len + offset + 1, total_len)
                locs = self.req_to_token[req_idx, :kv_len].to(
                    device=k_slots.device, dtype=torch.long
                )
                maybe_detect_oob(
                    locs,
                    0,
                    k_slots.shape[0],
                    "MiniMax NPU sparse prefill req_to_token -> main KV",
                )
                maybe_detect_oob(
                    locs,
                    0,
                    idx_k_slots.shape[0],
                    "MiniMax NPU sparse prefill req_to_token -> index KV",
                )
                k_tokens = k_slots.index_select(0, locs)
                v_tokens = v_slots.index_select(0, locs)
                idx_k_tokens = idx_k_slots.index_select(0, locs)
                idx_v_tokens = (
                    None if idx_v_slots is None else idx_v_slots.index_select(0, locs)
                )
                self._debug_sync_npu(
                    f"sparse prefill gather batch_id={batch_id} q_pos={q_pos}"
                )
                cur_idx_o, cur_o = self._npu_sparse_one(
                    q[q_pos],
                    k_tokens,
                    v_tokens,
                    idx_q[q_pos],
                    idx_k_tokens,
                    idx_v_tokens,
                    kv_len,
                )
                out[q_pos] = cur_o
                if idx_out is not None:
                    idx_out[q_pos] = cur_idx_o
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
        idx_v_slots = (
            None if idx_v_cache is None else self._cache_as_slots(idx_v_cache)
        )
        out = q.new_zeros(q.shape)
        idx_out = None if idx_v_slots is None else idx_q.new_zeros(idx_q.shape)

        for batch_id in range(q.shape[0]):
            self._debug_sync_npu(f"sparse decode before metadata batch_id={batch_id}")
            req_idx = int(forward_batch.req_pool_indices[batch_id].item())
            kv_len = int(forward_batch.seq_lens[batch_id].item())
            locs = self.req_to_token[req_idx, :kv_len].to(
                device=k_slots.device, dtype=torch.long
            )
            maybe_detect_oob(
                locs,
                0,
                k_slots.shape[0],
                "MiniMax NPU sparse decode req_to_token -> main KV",
            )
            maybe_detect_oob(
                locs,
                0,
                idx_k_slots.shape[0],
                "MiniMax NPU sparse decode req_to_token -> index KV",
            )
            k_tokens = k_slots.index_select(0, locs)
            v_tokens = v_slots.index_select(0, locs)
            idx_k_tokens = idx_k_slots.index_select(0, locs)
            idx_v_tokens = (
                None if idx_v_slots is None else idx_v_slots.index_select(0, locs)
            )
            self._debug_sync_npu(f"sparse decode gather batch_id={batch_id}")
            cur_idx_o, cur_o = self._npu_sparse_one(
                q[batch_id],
                k_tokens,
                v_tokens,
                idx_q[batch_id],
                idx_k_tokens,
                idx_v_tokens,
                kv_len,
            )
            out[batch_id] = cur_o
            if idx_out is not None:
                idx_out[batch_id] = cur_idx_o
        return idx_out, out

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
