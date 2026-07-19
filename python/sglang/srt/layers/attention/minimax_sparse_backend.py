from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence

import torch

from sglang.kernels.ops.attention.minimax_sparse.common.utils import get_cu_seqblocks
from sglang.srt.configs.model_config import (
    get_minimax_sparse_attention_config,
    get_minimax_sparse_disable_value_layer_ids,
    get_minimax_sparse_layer_ids,
    get_minimax_sparse_score_type,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
    _warn_msa_fallback,
    minimax_sparse_decode,
    minimax_sparse_prefill,
)
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import m3_fp8_attn_gemm_enabled

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _quant_q_fp8(q: torch.Tensor, q_scale: Optional[float]) -> torch.Tensor:
    # Same convention as the KV pools: the fp8 tensor stores value/scale and
    # the attention kernels multiply the logits back by the scale (None = unit).
    if q_scale is not None:
        q = q / q_scale
    return q.to(torch.float8_e4m3fn)


@dataclass
class MiniMaxSparsePrefillMetadata:
    """Layer-invariant metadata for one sparse prefill forward."""

    cu_seqlens: torch.Tensor
    seq_lens: torch.Tensor
    prefix_lens: torch.Tensor
    actual_num_tokens: int
    cu_seqblocks_q: torch.Tensor
    max_seqblock_q: int
    all_seqblock_q: int
    use_msa: bool
    msa_prefill_metadata: Optional[Any] = None
    msa_kv_indices: Optional[torch.Tensor] = None
    msa_plan: Optional[Any] = None


class MiniMaxSparseAttnBackend(AttentionBackend):
    def __init__(self, runner: ModelRunner):
        assert isinstance(runner.token_to_kv_pool, MiniMaxSparseKVPool)
        self.kv_pool = runner.token_to_kv_pool
        self.req_to_token = runner.req_to_token_pool.req_to_token
        self.max_context_len = int(runner.model_config.context_len)
        self.fp8_attn_gemm = m3_fp8_attn_gemm_enabled(runner.server_args)
        if self.fp8_attn_gemm:
            assert self.kv_pool.main_pool.dtype == torch.float8_e4m3fn, (
                "fp8 attn-GEMM mode requires an fp8_e4m3fn main KV pool, got "
                f"{self.kv_pool.main_pool.dtype}"
            )

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

        # Plain Python int so it is safe inside CUDA graphs (no .item() at graph time).
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

        # MSA (fmha_sm100) is SM100-only; fall back to the Triton sparse path when
        # the kernel is unavailable or its constraints don't hold.
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.minimax_sparse_ops.msa import (
            msa_available,
            msa_direct_prefill_available,
        )

        # MSA (fmha_sm100) runs bf16, or uniform fp8_e4m3 under fp8 attn-GEMM mode
        # (which also casts q to fp8). An fp8 main KV cache WITHOUT the flag
        # would pair a bf16 q with fp8 K/V — unsupported by fmha_sm100's
        # uniform-dtype kernels — so it stays on the Triton sparse path (which
        # dequants fp8 on load). e5m2 is never allowed into MSA (fmha_sm100's
        # variant lookup would silently dispatch the e4m3 kernel).
        _main_kv_is_fp8 = self.kv_pool.main_pool.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        )
        _msa_fp8_ok = (
            self.fp8_attn_gemm and self.kv_pool.main_pool.dtype == torch.float8_e4m3fn
        )
        self.use_msa = (
            not envs.SGLANG_DISABLE_MSA.get()
            and msa_available()
            and self.block_size_k == 128
            and self.kv_pool.page_size == self.block_size_k
            and self.topk_blocks in (4, 8, 16, 32)
            and (not _main_kv_is_fp8 or _msa_fp8_ok)
        )
        if (
            not self.use_msa
            and not envs.SGLANG_DISABLE_MSA.get()
            and msa_available()
            and self.block_size_k == 128
            and self.kv_pool.page_size != self.block_size_k
        ):
            logger.warning(
                "MiniMax-M3 MSA decode disabled: page_size=%d != sparse block size "
                "%d. Pass --page-size 128 (with an attention backend that allows it, "
                "e.g. fa4 or trtllm_mha) to enable the faster MSA kernel; falling "
                "back to the Triton sparse path.",
                self.kv_pool.page_size,
                self.block_size_k,
            )
        self._msa_dec_meta = None
        if self.use_msa:
            from sglang.srt.runtime_context import get_parallel

            self.num_q_heads = (
                runner.model_config.num_attention_heads // get_parallel().attn_tp_size
            )
            self.num_kv_heads = self.kv_pool.main_pool.head_num
            self._msa_nb_max = (
                self.max_context_len + self.block_size_k - 1
            ) // self.block_size_k
            self._msa_cg: dict[int, tuple] = {}

        self.page_size = self.kv_pool.page_size
        self.use_dense_sparse_decode = (
            envs.SGLANG_OPT_USE_MINIMAX_DENSE_SPARSE_DECODE.get()
            and self.block_size_k % self.page_size == 0
            # _dense_sparse_main_decode calls trtllm decode with a bf16 q and
            # unit bmm scales — no fp8 handling yet (follow-up).
            and not self.fp8_attn_gemm
        )
        from sglang.srt.model_executor.cuda_graph_config import (
            Backend,
            Phase,
            check_cuda_graph_backend,
        )

        _sa = getattr(runner, "server_args", None)
        _decode_cuda_graph = not check_cuda_graph_backend(
            Phase.DECODE, Backend.DISABLED
        )
        _full_prefill_cuda_graph = check_cuda_graph_backend(Phase.PREFILL, Backend.FULL)
        # Full prefill graphs capture metadata tensor addresses. Until MSA has
        # a graph-stable prefill plan/update API, preserve the old per-layer
        # Triton path there. Eager, breakable, and piecewise prefill all run the
        # request-level preparation once before entering the layer loop.
        self._cache_prefill_metadata = not _full_prefill_cuda_graph
        # MSA sparse prefill supports BF16 and uniform FP8 through the fmha_sm100
        # bridge. The stable direct sparse-prefill API is only enabled for BF16;
        # FP8 keeps the bridge so its plan and dequant scales stay explicit.
        self._use_msa_prefill = self.use_msa and not _full_prefill_cuda_graph
        self._use_msa_direct_prefill = (
            self._use_msa_prefill
            and self.kv_pool.main_pool.dtype == torch.bfloat16
            and msa_direct_prefill_available()
        )
        self._prefill_meta: Optional[MiniMaxSparsePrefillMetadata] = None
        self._use_msa_decode = self.use_msa and (
            not _decode_cuda_graph or envs.SGLANG_OPT_USE_MSA_DECODE_UNDER_GRAPH.get()
        )

        # MSA + spec decode + cuda graph crashes mid-capture: TARGET_VERIFY batches
        # route to forward_extend, dereferencing absent extend metadata. Fail at startup.
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
        self._msa_owns_decode = self._use_msa_decode and not (
            self.use_dense_sparse_decode and self.kv_pool.main_pool.head_num == 1
        )
        self.dense_backend: Optional[AttentionBackend] = None

        if self._use_msa_prefill:
            prefill_attn = (
                "MSA-direct" if self._use_msa_direct_prefill else "MSA-bridge"
            )
        else:
            prefill_attn = "triton"
        logger.info(
            f"[MiniMaxSparse] Backend initialized "
            f"(score_type={self.score_type!r}, "
            f"prefill_attn={prefill_attn}, "
            f"decode_attn={'MSA' if self._use_msa_decode else 'triton'}, "
            f"msa_owns_decode={self._msa_owns_decode}, "
            f"decode_cuda_graph={_decode_cuda_graph}, "
            f"fp8_attn_gemm={self.fp8_attn_gemm}, "
            f"disable_value_layers={sorted(self.disable_value_layer_ids)})"
        )
        if self.fp8_attn_gemm and self.use_msa:
            logger.info(
                "[MiniMaxSparse] fp8 MSA active: the first forward may "
                "JIT-compile fmha_sm100 fp8 kernel variants (cold cache can "
                "take minutes; compiles serialize across TP ranks)."
            )

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        # cuda-graph replay views are a SimpleNamespace without extend_seq_lens_cpu,
        # and TARGET_VERIFY sets it to None despite is_extend() — getattr covers both.
        self._msa_dec_meta = None
        self._prefill_meta = None
        extend_lens_cpu = self._to_int_list(
            getattr(forward_batch, "extend_seq_lens_cpu", None)
        )
        if extend_lens_cpu:
            self._max_seqlen_q = max(extend_lens_cpu)
        else:
            extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
            self._max_seqlen_q = (
                int(extend_seq_lens.max().item())
                if extend_seq_lens is not None and extend_seq_lens.numel() > 0
                else 1
            )
        if in_capture and forward_batch.forward_mode.is_decode_or_idle():
            self._max_seqlen_k = self.max_context_len
        else:
            seq_lens_cpu = self._to_int_list(forward_batch.seq_lens_cpu)
            self._max_seqlen_k = (
                max(seq_lens_cpu)
                if seq_lens_cpu
                else (
                    int(forward_batch.seq_lens.max().item())
                    if forward_batch.seq_lens.numel() > 0
                    else 1
                )
            )

        if (
            self._cache_prefill_metadata
            and forward_batch.forward_mode.is_extend(include_draft_extend_v2=True)
            and getattr(forward_batch, "extend_seq_lens", None) is not None
        ):
            self._prefill_meta = self._build_prefill_metadata(
                forward_batch, use_host_lengths=True
            )

        # Build plan + page table eager (outside capture) so captured forward_decode
        # runs only device-side ops; host-side code can't be captured.
        if self._msa_owns_decode and forward_batch.forward_mode.is_decode_or_idle():
            self._prepare_msa_decode_meta(forward_batch)

    @staticmethod
    def _to_int_list(values) -> Optional[list[int]]:
        if values is None:
            return None
        if torch.is_tensor(values):
            return [int(value) for value in values.tolist()]
        return [int(value) for value in values]

    @staticmethod
    def _make_cu_seqlens(
        lengths: torch.Tensor,
        lengths_cpu: Optional[Sequence[int]],
        *,
        use_host_lengths: bool,
    ) -> torch.Tensor:
        if (
            use_host_lengths
            and lengths_cpu is not None
            and len(lengths_cpu) == lengths.numel()
        ):
            cumulative = [0]
            for length in lengths_cpu:
                cumulative.append(cumulative[-1] + int(length))
            return torch.tensor(cumulative, dtype=torch.int32, device=lengths.device)
        lengths_i32 = lengths.to(torch.int32)
        return torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=lengths.device),
                lengths_i32.cumsum(0),
            ]
        )

    def _build_prefill_metadata(
        self,
        forward_batch: ForwardBatch,
        *,
        use_host_lengths: bool,
    ) -> MiniMaxSparsePrefillMetadata:
        extend_seq_lens = forward_batch.extend_seq_lens.to(torch.int32)
        extend_lens_cpu = self._to_int_list(
            getattr(forward_batch, "extend_seq_lens_cpu", None)
        )
        if extend_lens_cpu is not None:
            actual_num_tokens = sum(extend_lens_cpu)
        else:
            actual_num_tokens = int(extend_seq_lens.sum().item())

        cu_seqlens = self._make_cu_seqlens(
            extend_seq_lens,
            extend_lens_cpu,
            use_host_lengths=use_host_lengths,
        )
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        if forward_batch.extend_prefix_lens is None:
            prefix_lens = torch.zeros_like(seq_lens)
        else:
            prefix_lens = forward_batch.extend_prefix_lens.to(torch.int32)

        # MiniMax-M3 fixes block_size_q=1. In that case query block offsets are
        # exactly token offsets, so alias cu_seqlens instead of launching the
        # generic diff/div/cumsum helper (which also computes unused K blocks).
        if self.block_size_q == 1:
            cu_seqblocks_q = cu_seqlens
            max_seqblock_q = self._max_seqlen_q
            all_seqblock_q = actual_num_tokens
        else:
            (
                cu_seqblocks_q,
                max_seqblock_q,
                all_seqblock_q,
                _,
                _,
                _,
            ) = get_cu_seqblocks(
                cu_seqlens,
                self._max_seqlen_q,
                self.block_size_q,
                self.block_size_k,
                extend_lens_cpu,
            )

        use_msa = self._use_msa_prefill
        msa_prefill_metadata = None
        msa_kv_indices = None
        msa_plan = None
        if use_msa:
            from sglang.srt.layers.attention.minimax_sparse_ops.msa import (
                MSAUnavailableError,
                build_msa_prefill_bridge_meta,
                build_msa_prefill_metadata,
            )

            seq_lens_cpu = self._to_int_list(forward_batch.seq_lens_cpu)
            if seq_lens_cpu is not None and len(seq_lens_cpu) != seq_lens.numel():
                seq_lens_cpu = None
            try:
                # MSA selects sparse prefill when at least one row has more
                # than 32 query tokens. Its direct CuTe API also supports the
                # shorter rows in a mixed varlen batch, so keep the whole batch
                # on the direct path and avoid rebuilding bridge metadata in
                # every sparse layer.
                use_direct_msa = (
                    self._use_msa_direct_prefill
                    and extend_lens_cpu is not None
                    and len(extend_lens_cpu) == extend_seq_lens.numel()
                    and max(extend_lens_cpu, default=0) > 32
                )
                if use_direct_msa:
                    msa_prefill_metadata = build_msa_prefill_metadata(
                        self.req_to_token,
                        forward_batch.req_pool_indices,
                        cu_seqlens,
                        extend_seq_lens,
                        seq_lens,
                        prefix_lens,
                        self.block_size_k,
                        self._max_seqlen_q,
                        self._max_seqlen_k,
                        seq_lens_cpu,
                    )
                else:
                    msa_kv_indices, msa_plan = build_msa_prefill_bridge_meta(
                        self.req_to_token,
                        forward_batch.req_pool_indices,
                        extend_seq_lens,
                        seq_lens,
                        prefix_lens,
                        self.num_q_heads,
                        self.num_kv_heads,
                        self.block_size_k,
                        self.topk_blocks,
                        is_fp8=self.fp8_attn_gemm,
                    )
            except MSAUnavailableError as err:
                _warn_msa_fallback(err)
                use_msa = False

        return MiniMaxSparsePrefillMetadata(
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            actual_num_tokens=actual_num_tokens,
            cu_seqblocks_q=cu_seqblocks_q,
            max_seqblock_q=max_seqblock_q,
            all_seqblock_q=all_seqblock_q,
            use_msa=use_msa,
            msa_prefill_metadata=msa_prefill_metadata,
            msa_kv_indices=msa_kv_indices,
            msa_plan=msa_plan,
        )

    def _prepare_msa_decode_meta(self, forward_batch: ForwardBatch):
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
                is_fp8=self.fp8_attn_gemm,
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
                layer.k_scale_float,
                layer.v_scale_float,
                layer.idx_k_scale_float,
                layer.idx_v_scale_float,
            )
        k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
        if disable_value:
            idx_k_cache = self.kv_pool.get_index_k_buffer(layer.layer_id)
            idx_v_cache = None
        else:
            idx_k_cache, idx_v_cache = self.kv_pool.get_index_kv_buffer(layer.layer_id)

        prefill_meta = self._prefill_meta
        if prefill_meta is None:
            # Experimental full-prefill CUDA graph keeps the old per-layer
            # construction so metadata ops are captured with stable addresses.
            prefill_meta = self._build_prefill_metadata(
                forward_batch, use_host_lengths=self._cache_prefill_metadata
            )

        cu_seqlens = prefill_meta.cu_seqlens
        seq_lens = prefill_meta.seq_lens
        prefix_lens = prefill_meta.prefix_lens
        actual_num_tokens = prefill_meta.actual_num_tokens

        # DP attention pads q beyond the real token count for collective alignment;
        # trim to actual tokens so the sparse kernel sees consistent shapes.
        original_num_tokens = q.shape[0]
        if actual_num_tokens < original_num_tokens:
            q = q[:actual_num_tokens]
            idx_q = idx_q[:actual_num_tokens]

        # fp8 attention GEMMs: quantize q/idx_q AFTER the KV store (which reads
        # the bf16 k/v) and the DP trim.
        if self.fp8_attn_gemm:
            q = _quant_q_fp8(q, layer.q_scale_float)
            idx_q = _quant_q_fp8(idx_q, layer.idx_q_scale_float)

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
            use_msa=prefill_meta.use_msa,
            cu_seqblocks_q=prefill_meta.cu_seqblocks_q,
            max_seqblock_q=prefill_meta.max_seqblock_q,
            all_seqblock_q=prefill_meta.all_seqblock_q,
            seqlens_cpu=getattr(forward_batch, "extend_seq_lens_cpu", None),
            msa_prefill_metadata=prefill_meta.msa_prefill_metadata,
            msa_kv_indices=prefill_meta.msa_kv_indices,
            msa_plan=prefill_meta.msa_plan,
            q_scale=layer.q_scale_float,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
            idx_q_scale=layer.idx_q_scale_float,
            idx_k_scale=layer.idx_k_scale_float,
            idx_v_scale=layer.idx_v_scale_float,
        )

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
        q: torch.Tensor,
        page_table: torch.Tensor,
        real_seq_lens: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
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
            layer.k_scale_float,
            layer.v_scale_float,
            layer.idx_k_scale_float,
            layer.idx_v_scale_float,
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

        # fp8 attention GEMMs: quantize q/idx_q AFTER the KV store (which reads
        # the bf16 k/v).
        if self.fp8_attn_gemm:
            q = _quant_q_fp8(q, layer.q_scale_float)
            idx_q = _quant_q_fp8(idx_q, layer.idx_q_scale_float)

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
            q_scale=layer.q_scale_float,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
            idx_q_scale=layer.idx_q_scale_float,
            idx_k_scale=layer.idx_k_scale_float,
            idx_v_scale=layer.idx_v_scale_float,
        )
        return (
            None if idx_o is None else idx_o.reshape(q.shape[0], -1).contiguous(),
            o.reshape(q.shape[0], -1).contiguous(),
        )


class MiniMaxHybridAttnBackend(AttentionBackend):
    def __init__(
        self,
        dense_backend: AttentionBackend,
        sparse_backend: MiniMaxSparseAttnBackend,
        sparse_layer_ids: list[int],
    ):
        self.dense = dense_backend
        self.sparse = sparse_backend
        self.sparse_layer_ids = sparse_layer_ids
        self.sparse.dense_backend = dense_backend
        self.extend_dummy_seqs_capped_by_req_pool = getattr(
            dense_backend, "extend_dummy_seqs_capped_by_req_pool", False
        ) or getattr(sparse_backend, "extend_dummy_seqs_capped_by_req_pool", False)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
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

        # DP attention pads q to an even length but flashinfer builds qo_indptr from
        # extend_seq_lens, so padded q.shape[0] != qo_indptr[-1] and paged-prefill
        # raises. Trim q and re-pad output; k/v stay untrimmed so KV-cache writes
        # align with out_cache_loc.
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
