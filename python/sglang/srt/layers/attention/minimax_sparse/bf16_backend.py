# SPDX-License-Identifier: Apache-2.0
"""CUDA BF16 / FP8 MiniMax sparse attention backend (Triton + SM100 MSA)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.minimax_sparse.base import MiniMaxSparseBaseBackend
from sglang.srt.layers.attention.minimax_sparse.metadata import _quant_q_fp8
from sglang.srt.layers.attention.minimax_sparse_ops.msa import msa_available
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel, get_spec

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class MiniMaxBF16SparseAttnBackend(MiniMaxSparseBaseBackend):
    """CUDA BF16 / FP8 sparse attention backend (Triton + optional fmha_sm100 MSA)."""

    def __init__(self, runner: ModelRunner):
        super().__init__(runner)
        self._native_sparse_ok = False

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
            and self.hisparse_coordinator is None
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
            self.num_q_heads = (
                runner.model_config.num_attention_heads // get_parallel().attn_tp_size
            )
            self.num_kv_heads = self.kv_pool.main_pool.head_num
            self._msa_nb_max = (
                self.max_kv_len + self.block_size_k - 1
            ) // self.block_size_k
            self._msa_cg: dict[int, tuple] = {}

        self.use_dense_sparse_decode = (
            self.hisparse_coordinator is None
            and envs.SGLANG_OPT_USE_MINIMAX_DENSE_SPARSE_DECODE.get()
            and self.block_size_k % self.page_size == 0
            # _dense_sparse_main_decode calls trtllm decode with a bf16 q and
            # unit bmm scales — no fp8 handling yet (follow-up).
            and not self.fp8_attn_gemm
        )

        spec = get_spec()
        self.kv_is_nvfp4 = False
        self.index_topk_freq = 1
        self.index_cache_enabled = False
        self._init_platform_extensions(spec)

        _decode_cuda_graph = not check_cuda_graph_backend(
            Phase.DECODE, Backend.DISABLED
        )
        self._use_msa_decode = self.use_msa and (
            not _decode_cuda_graph or envs.SGLANG_OPT_USE_MSA_DECODE_UNDER_GRAPH.get()
        )

        # MSA + spec decode + cuda graph crashes mid-capture: TARGET_VERIFY batches
        # route to forward_extend, dereferencing absent extend metadata. Fail at startup.
        if (
            self.use_msa
            and _decode_cuda_graph
            and spec.speculative_algorithm is not None
        ):
            raise NotImplementedError(
                "MiniMax-M3 MSA attention does not support speculative decoding under "
                "CUDA graph. Use --disable-decode-cuda-graph --disable-prefill-cuda-graph, set SGLANG_DISABLE_MSA=1, or "
                "disable speculative decoding."
            )
        self._msa_owns_decode = self._use_msa_decode and not (
            self.use_dense_sparse_decode and self.kv_pool.main_pool.head_num == 1
        )
        # Without the MSA decode plan, a TARGET_VERIFY step runs the capture-safe
        # decode kernels with one causal query per draft row.
        self._linear_verify = not self._use_msa_decode

        from sglang.srt.layers.attention.minimax_sparse_ops.indexer_cp import (
            make_indexer_cp,
        )

        self.indexer_cp = make_indexer_cp(self, runner, self._sparse_cfg)

        logger.info(
            f"[MiniMaxSparse] Backend initialized "
            f"(score_type={self.score_type!r}, "
            f"main_attn={'MSA' if self.use_msa else 'triton'}, "
            f"index_topk_freq={self.index_topk_freq}, "
            f"msa_decode={self._use_msa_decode}, "
            f"msa_owns_decode={self._msa_owns_decode}, "
            f"decode_cuda_graph={_decode_cuda_graph}, "
            f"fp8_attn_gemm={self.fp8_attn_gemm}, "
            f"hisparse={'enabled' if self._loc_mapping is not None else 'disabled'}, "
            f"npu_native_attn=off, "
            f"disable_value_layers={sorted(self.disable_value_layer_ids)})"
        )
        if self.fp8_attn_gemm and self.use_msa:
            logger.info(
                "[MiniMaxSparse] fp8 MSA active: the first forward may "
                "JIT-compile fmha_sm100 fp8 kernel variants (cold cache can "
                "take minutes; compiles serialize across TP ranks)."
            )

    def _init_platform_extensions(self, spec) -> None:
        del spec

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        # getattr covers replay views lacking extend_seq_lens_cpu and TARGET_VERIFY.
        self._msa_dec_meta = None
        # Per-forward cache of the layer-invariant prefill seqblock trio.
        self._prefill_seqblock_meta = None
        extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_lens is not None:
            self._max_seqlen_q = int(max(extend_lens))
        else:
            self._max_seqlen_q = 1
        if in_capture and (
            forward_batch.forward_mode.is_decode_or_idle()
            or (self._linear_verify and forward_batch.forward_mode.is_target_verify())
        ):
            # Capture uses tiny dummy seq_lens; bound by full context so replay
            # (longer sequences) does not miss KV blocks.
            self._max_seqlen_k = self.max_kv_len
        else:
            self._max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item())
            if self._linear_verify and forward_batch.forward_mode.is_target_verify():
                self._max_seqlen_k += self.speculative_num_draft_tokens

        # Build plan + page table eager (outside capture) so captured forward_decode
        # runs only device-side ops; host-side code can't be captured.
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
        if self._linear_verify and forward_batch.forward_mode.is_target_verify():
            self._init_linear_verify_metadata(forward_batch)

    def _resolve_extend_meta(self, forward_batch: ForwardBatch):
        cu_seqlens = torch.cat(
            [
                torch.zeros(
                    1, dtype=torch.int32, device=forward_batch.extend_seq_lens.device
                ),
                forward_batch.extend_seq_lens.to(torch.int32).cumsum(0).to(torch.int32),
            ]
        )
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        if forward_batch.extend_prefix_lens is not None:
            prefix_lens = forward_batch.extend_prefix_lens.to(torch.int32)
        else:
            prefix_lens = torch.zeros_like(seq_lens)
        return cu_seqlens, seq_lens, prefix_lens

    def _should_skip_extend_kv_store(
        self, forward_batch: ForwardBatch, layer_id: int, save_kv_cache: bool
    ) -> bool:
        del save_kv_cache
        return self._is_sparse_kv_cached_by_fusion(forward_batch, layer_id)

    def _get_layer_caches(self, layer_id: int, disable_value: bool):
        k_cache, v_cache = self.kv_pool.layer_router.read(layer_id).kv()
        if disable_value:
            idx_k_cache = self.kv_pool.get_index_k_buffer(layer_id)
            idx_v_cache = None
        else:
            idx_k_cache, idx_v_cache = self.kv_pool.get_index_kv_buffer(layer_id)
        return k_cache, v_cache, idx_k_cache, idx_v_cache

    def _prepare_prefill_topk_reuse(
        self, forward_batch: ForwardBatch, layer_id: int, disable_value: bool
    ) -> tuple[Optional[torch.Tensor], bool, Optional[int]]:
        del forward_batch, layer_id, disable_value
        return None, False, None

    def _store_prefill_topk_reuse(
        self, group: Optional[int], reduced_topk_idx: torch.Tensor
    ) -> None:
        del group, reduced_topk_idx

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
        out_quant: str = "",
    ):
        if self._linear_verify and forward_batch.forward_mode.is_target_verify():
            return self._forward_linear_verify_funnel(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                idx_q=idx_q,
                idx_k=idx_k,
                idx_v=idx_v,
                out_quant=out_quant,
            )
        disable_value = layer.layer_id in self.disable_value_layer_ids
        if not self._should_skip_extend_kv_store(
            forward_batch, layer.layer_id, save_kv_cache
        ):
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
        k_cache, v_cache, idx_k_cache, idx_v_cache = self._get_layer_caches(
            layer.layer_id, disable_value
        )

        cached = self._prefill_seqblock_meta
        if cached is None or cached[0] is not forward_batch:
            cu_seqlens, seq_lens, prefix_lens = self._resolve_extend_meta(forward_batch)
            from sglang.kernels.ops.attention.minimax_sparse.common.utils import (
                get_cu_seqblocks,
            )

            cu_seqblocks_q, max_seqblock_q, all_seqblock_q, _, _, _ = get_cu_seqblocks(
                cu_seqlens,
                self._max_seqlen_q,
                self.block_size_q,
                self.block_size_k,
                forward_batch.extend_seq_lens_cpu,
            )
            cached = (
                forward_batch,
                cu_seqlens,
                seq_lens,
                prefix_lens,
                cu_seqblocks_q,
                max_seqblock_q,
                all_seqblock_q,
            )
            self._prefill_seqblock_meta = cached
        (
            _,
            cu_seqlens,
            seq_lens,
            prefix_lens,
            cu_seqblocks_q,
            max_seqblock_q,
            all_seqblock_q,
        ) = cached

        # DP attention pads q beyond real tokens; trim (CPU list avoids a sync).
        if forward_batch.extend_seq_lens_cpu is not None:
            actual_num_tokens = int(sum(forward_batch.extend_seq_lens_cpu))
        else:
            actual_num_tokens = int(cu_seqlens[-1].item())
        original_num_tokens = q.shape[0]
        if actual_num_tokens < original_num_tokens:
            q = q[:actual_num_tokens]
            idx_q = idx_q[:actual_num_tokens]

        # fp8 attention GEMMs: quantize q/idx_q AFTER the KV store (which reads
        # the bf16 k/v) and the DP trim.
        if self.fp8_attn_gemm:
            q = _quant_q_fp8(q, layer.q_scale_float)
            idx_q = _quant_q_fp8(idx_q, layer.idx_q_scale_float)

        from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
            minimax_sparse_prefill,
        )

        cached_topk_idx, want_topk, group = self._prepare_prefill_topk_reuse(
            forward_batch, layer.layer_id, disable_value
        )

        result = minimax_sparse_prefill(
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
            seqlens_cpu=forward_batch.extend_seq_lens_cpu,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            cu_seqblocks_q=cu_seqblocks_q,
            max_seqblock_q=max_seqblock_q,
            all_seqblock_q=all_seqblock_q,
            q_scale=layer.q_scale_float,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
            idx_q_scale=layer.idx_q_scale_float,
            idx_k_scale=layer.idx_k_scale_float,
            idx_v_scale=layer.idx_v_scale_float,
            page_size=self.page_size,
            cached_topk_idx=cached_topk_idx,
            return_topk_idx=want_topk,
            loc_mapping=self._loc_mapping,
            # A quantized O cannot take the DP padding rows appended below.
            out_quant=(out_quant if actual_num_tokens == original_num_tokens else ""),
        )
        if want_topk:
            idx_o, o, reduced_topk_idx = result
            self._store_prefill_topk_reuse(group, reduced_topk_idx)
        else:
            idx_o, o = result

        if actual_num_tokens < original_num_tokens:
            pad_len = original_num_tokens - actual_num_tokens
            o = torch.cat([o, o.new_zeros(pad_len, *o.shape[1:])], dim=0)
            if idx_o is not None:
                idx_o = torch.cat(
                    [idx_o, idx_o.new_zeros(pad_len, *idx_o.shape[1:])], dim=0
                )

        if not isinstance(o, tuple):
            o = o.reshape(original_num_tokens, -1).contiguous()
        return (
            (
                None
                if idx_o is None
                else idx_o.reshape(original_num_tokens, -1).contiguous()
            ),
            o,
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
        del forward_batch
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

    def _prepare_decode_topk_reuse(
        self,
        num_q_rows: int,
        layer_id: int,
        disable_value: bool,
        has_dense_attn_fn: bool,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        del num_q_rows, layer_id, disable_value, has_dense_attn_fn
        return None, None

    def _verify_num_draft(self, forward_batch: ForwardBatch) -> Optional[int]:
        del forward_batch
        return None

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
        out_quant: str = "",  # o_proj's fp8_input_quant_format: O may come back as (fp8, scale)
        **kwargs,
    ):
        del save_kv_cache
        assert len(kwargs) == 0
        disable_value = layer.layer_id in self.disable_value_layer_ids
        if not self._is_sparse_kv_cached_by_fusion(forward_batch, layer.layer_id):
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
        k_cache, v_cache, idx_k_cache, idx_v_cache = self._get_layer_caches(
            layer.layer_id, disable_value
        )

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

        # fp8 attn-GEMM: quantize q/idx_q after the KV store (reads bf16 k/v).
        if self.fp8_attn_gemm:
            q = _quant_q_fp8(q, layer.q_scale_float)
            idx_q = _quant_q_fp8(idx_q, layer.idx_q_scale_float)

        from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
            minimax_sparse_decode,
        )

        _cached_topk, _topk_out = self._prepare_decode_topk_reuse(
            q.shape[0],
            layer.layer_id,
            disable_value,
            attn_fn is not None,
        )

        hisparse_swap_in_fn = None
        if self.hisparse_coordinator is not None:

            def hisparse_swap_in_fn(topk_idx):
                return self._hisparse_swap_in_blocks(
                    forward_batch=forward_batch,
                    topk_idx=topk_idx,
                    layer_id=layer.layer_id,
                )

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
            cached_topk_idx=_cached_topk,
            topk_out=_topk_out,
            hisparse_swap_in_fn=hisparse_swap_in_fn,
            indexer_cp=self.indexer_cp,
            # the verify funnel lays out each request's draft rows consecutively
            packed_queries=(
                self.speculative_num_draft_tokens
                if forward_batch.forward_mode.is_target_verify()
                else 1
            ),
            verify_num_draft=self._verify_num_draft(forward_batch),
            out_quant=out_quant,
        )
        if not isinstance(o, tuple):
            o = o.reshape(q.shape[0], -1).contiguous()
        return (
            None if idx_o is None else idx_o.reshape(q.shape[0], -1).contiguous(),
            o,
        )
