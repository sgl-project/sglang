# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from types import SimpleNamespace

import torch

from sglang.srt.configs.model_config import get_minimax_sparse_attention_config
from sglang.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    SharedReadEnds,
)
from sglang.srt.layers.attention.minimax_sparse_ops.msa_plan import (
    MSAV1DecodeAttentionPlan,
    MSAV1DecodeIndexerPlan,
    MSAV1PrefillPlan,
    build_msa_v1_page_table,
    msa_v1_qmul4_decode_capability,
)
from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import MiniMaxM3Nvfp4KVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel, get_spec

logger = logging.getLogger(__name__)


class MiniMaxSparseNvfp4AttnBackend(AttentionBackend):
    """MSA-v1 Q8KV4 attention with a K-only sparse index."""

    needs_cpu_seq_lens: bool = False
    skips_prefill_metadata_on_idle_dp_rank: bool = True

    def __init__(self, runner):
        self.kv_pool = runner.token_to_kv_pool
        if not isinstance(self.kv_pool, MiniMaxM3Nvfp4KVPool):
            raise ValueError("M3.1 NVFP4 attention requires a packed NVFP4 KV pool")
        self.token_to_kv_pool = self.kv_pool
        self.req_to_token_pool = runner.req_to_token_pool
        self.req_to_token = runner.req_to_token_pool.req_to_token
        self.max_context_len = int(runner.model_config.context_len)
        self.page_size = self.kv_pool.page_size
        self.idx_head_dim = 128
        sparse_cfg = get_minimax_sparse_attention_config(runner.model_config.hf_config)
        for name, expected in {
            "sparse_index_dim": 128,
            "sparse_block_size": 128,
            "sparse_topk_blocks": 16,
            "sparse_num_index_heads": 4,
        }.items():
            if sparse_cfg.get(name) != expected:
                raise ValueError(f"MSA-v1 requires {name}={expected}")
        config = runner.model_config.hf_text_config
        if (
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        ) != (64, 4, 128):
            raise ValueError(
                "MSA-v1 requires 64 query heads, 4 KV heads and head_dim=128"
            )
        self.speculative_num_draft_tokens = get_spec().speculative_num_draft_tokens
        self.sparse_layer_ids = list(self.kv_pool.sparse_layer_id_mapping)
        self._msa_v1_num_q_heads = (
            config.num_attention_heads // get_parallel().attn_tp_size
        )
        self._msa_v1_num_kv_heads = self.kv_pool.head_num
        # One index head per KV head: decode attention reads each head's top-k.
        self._msa_v1_num_index_heads = self._msa_v1_num_kv_heads
        self._msa_v1_qmul4_available, self.qmul4_unavailable_reason = (
            msa_v1_qmul4_decode_capability()
        )
        self._msa_v1_index_fp8_scratch = None
        if not self._msa_v1_qmul4_available:
            logger.warning(
                "MSA-v1 Q8KV4 decode is unavailable (%s); "
                "using the slower FP8 fallback.",
                self.qmul4_unavailable_reason,
            )
            self._msa_v1_index_fp8_scratch = torch.empty(
                (
                    self.kv_pool.index_k_pool.num_pages,
                    self.page_size,
                    self.idx_head_dim,
                ),
                dtype=torch.float8_e4m3fn,
                device=runner.device,
            )
        self._msa_v1_attention_workspace = None
        self._msa_v1_cuda_graph_plans = {}
        self._msa_v1_cuda_graph_attention_plans = {}
        self._msa_v1_using_cuda_graph_plan = False
        self._max_seqlen_q = self._max_seqlen_k = 1
        self.forward_metadata = None

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        if forward_batch.forward_mode.is_idle():
            # Follow v, not q: q may already be FP8 from the qk-norm epilogue.
            return None, v.new_zeros(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        return super().forward(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)

    def shared_read_ends(self, fm):
        return SharedReadEnds.PRE_REPLAY

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        self.forward_metadata = SimpleNamespace(
            msa_v1_decode_plan=None,
            msa_v1_attention_plan=None,
            msa_v1_prefill_plan=None,
        )
        mode = forward_batch.forward_mode
        if forward_batch.seq_lens.numel() == 0:
            return
        if mode.is_decode_or_idle() or mode.is_target_verify():
            self._prepare_msa_v1_decode_plan(
                forward_batch, use_cuda_graph=True, max_seq_len=self.max_context_len
            )
            return
        lens = forward_batch.extend_seq_lens_cpu
        self._max_seqlen_q = max(lens)
        self._max_seqlen_k = max(
            prefix_len + extend_len
            for prefix_len, extend_len in zip(
                forward_batch.extend_prefix_lens_cpu, lens, strict=True
            )
        )
        q_lens = forward_batch.extend_seq_lens.to(torch.int32)
        cu_seqlens = torch.cat(
            (q_lens.new_zeros(1), q_lens.cumsum(0, dtype=torch.int32))
        )
        self._prepare_msa_v1_prefill_plan(
            forward_batch, cu_seqlens, forward_batch.seq_lens.to(torch.int32), sum(lens)
        )

    def _forward(
        self, q, k, v, layer, forward_batch, save_kv_cache, *, idx_q, idx_k, idx_v=None
    ):
        if save_kv_cache:
            self.kv_pool.set_fused_kv_index_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                idx_k,
                None,
                layer.k_scale_float,
                layer.v_scale_float,
                layer.idx_k_scale_float,
            )
        q = q.to(torch.float8_e4m3fn)
        idx_q = idx_q.to(torch.float8_e4m3fn)
        meta = self.forward_metadata
        if meta.msa_v1_decode_plan is not None:
            plan = meta.msa_v1_decode_plan
            total_q = plan.batch_size * plan.query_length
            topk = self._run_msa_v1_decode_indexer(idx_q[:total_q], layer)
            out = self._run_msa_v1_decode_attention(q[:total_q], topk, layer)
        else:
            # DP attention pads local Q to a common length; kernel metadata
            # describes only the real tokens on this attention DP rank.
            total_q = sum(forward_batch.extend_seq_lens_cpu)
            plan = meta.msa_v1_prefill_plan
            topk = plan.run_indexer(
                idx_q[:total_q],
                self.kv_pool.get_index_k_buffer(layer.layer_id),
                self.kv_pool.get_index_k_scale_buffer(layer.layer_id),
                self.kv_pool.index_k_pool.global_scale,
            )
            kc, vc = self.kv_pool.get_kv_buffer(layer.layer_id)
            out = plan.run_attention(
                q[:total_q],
                kc,
                vc,
                topk,
                sm_scale=layer.scaling,
                k_scale=self.kv_pool.get_main_k_scale_buffer(layer.layer_id),
                v_scale=self.kv_pool.get_main_v_scale_buffer(layer.layer_id),
                global_scale=self.kv_pool.main_pool.global_scale,
            )
        if total_q < q.shape[0]:
            out = torch.cat((out, out.new_zeros(q.shape[0] - total_q, *out.shape[1:])))
        return None, out.reshape(q.shape[0], -1).contiguous()

    def forward_extend(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        return self._forward(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)

    def forward_decode(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        return self._forward(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)

    def _prepare_msa_v1_decode_plan(
        self,
        forward_batch: ForwardBatch,
        *,
        use_cuda_graph: bool,
        max_seq_len: int,
    ) -> MSAV1DecodeIndexerPlan:
        batch_size = int(forward_batch.seq_lens.shape[0])
        query_length = (
            int(self.speculative_num_draft_tokens)
            if forward_batch.forward_mode.is_target_verify()
            else 1
        )
        max_pages = max(1, (int(max_seq_len) + self.page_size - 1) // self.page_size)
        key = (
            forward_batch.forward_mode,
            batch_size,
            query_length,
            max_pages,
        )
        if self._msa_v1_attention_workspace is None and (
            not self._msa_v1_qmul4_available
            or self._msa_v1_num_q_heads // self._msa_v1_num_kv_heads != 16
        ):
            from sglang.kernels.ops.attention.msa_v1.flashinfer_decode import (
                WORKSPACE_BYTES,
            )

            # Sequential graph buckets/layers reuse the backend's scratch.
            self._msa_v1_attention_workspace = torch.empty(
                WORKSPACE_BYTES, dtype=torch.uint8, device=forward_batch.seq_lens.device
            )
        if use_cuda_graph:
            plan = self._msa_v1_cuda_graph_plans.get(key)
            if plan is None:
                plan = MSAV1DecodeIndexerPlan(
                    batch_size=batch_size,
                    query_length=query_length,
                    max_pages=max_pages,
                    device=forward_batch.seq_lens.device,
                    use_cuda_graph=True,
                    num_index_heads=self._msa_v1_num_index_heads,
                    prefer_qmul4=self._msa_v1_qmul4_available,
                    fp8_fallback_scratch=self._msa_v1_index_fp8_scratch,
                )
                self._msa_v1_cuda_graph_plans[key] = plan
            attention_plan = self._msa_v1_cuda_graph_attention_plans.get(key)
            if attention_plan is None:
                attention_plan = MSAV1DecodeAttentionPlan(
                    batch_size=batch_size,
                    query_length=query_length,
                    max_pages=max_pages,
                    num_q_heads=self._msa_v1_num_q_heads,
                    num_kv_heads=self._msa_v1_num_kv_heads,
                    device=forward_batch.seq_lens.device,
                    sm_scale=self.idx_head_dim**-0.5,
                    prefer_qmul4=self._msa_v1_qmul4_available,
                    workspace_buffer=self._msa_v1_attention_workspace,
                    page_table_buffer=plan.page_table,
                    final_seq_lens_buffer=plan.final_seq_lens,
                )
                self._msa_v1_cuda_graph_attention_plans[key] = attention_plan
        else:
            plan = MSAV1DecodeIndexerPlan(
                batch_size=batch_size,
                query_length=query_length,
                max_pages=max_pages,
                device=forward_batch.seq_lens.device,
                use_cuda_graph=False,
                num_index_heads=self._msa_v1_num_index_heads,
                prefer_qmul4=self._msa_v1_qmul4_available,
                fp8_fallback_scratch=self._msa_v1_index_fp8_scratch,
            )
            attention_plan = MSAV1DecodeAttentionPlan(
                batch_size=batch_size,
                query_length=query_length,
                max_pages=max_pages,
                num_q_heads=self._msa_v1_num_q_heads,
                num_kv_heads=self._msa_v1_num_kv_heads,
                device=forward_batch.seq_lens.device,
                sm_scale=self.idx_head_dim**-0.5,
                prefer_qmul4=self._msa_v1_qmul4_available,
                workspace_buffer=self._msa_v1_attention_workspace,
            )
        if (
            self._msa_v1_attention_workspace is None
            and attention_plan.fp8_attention is not None
        ):
            self._msa_v1_attention_workspace = attention_plan.fp8_attention.workspace
        self._refresh_msa_v1_decode_plan(
            use_cuda_graph=use_cuda_graph,
            plan=plan,
            attention_plan=attention_plan,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            is_target_verify=forward_batch.forward_mode.is_target_verify(),
        )
        self.forward_metadata.msa_v1_decode_plan = plan
        self.forward_metadata.msa_v1_attention_plan = attention_plan
        self._msa_v1_using_cuda_graph_plan = use_cuda_graph
        return plan

    def _refresh_msa_v1_decode_plan(
        self,
        *,
        use_cuda_graph: bool,
        plan: MSAV1DecodeIndexerPlan,
        attention_plan: MSAV1DecodeAttentionPlan,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        is_target_verify: bool,
    ) -> None:
        if use_cuda_graph:
            # Out of graph: shared_read_ends() returns PRE_REPLAY, which capturing it
            # would break.
            plan.refresh(
                attention_plan=attention_plan,
                req_to_token=self.req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                is_target_verify=is_target_verify,
            )
            return
        page_table = build_msa_v1_page_table(
            self.req_to_token,
            req_pool_indices,
            max_seqlen_k=plan.max_pages * self.page_size,
        )
        final_seq_lens = seq_lens.to(torch.int32)
        if is_target_verify:
            final_seq_lens = final_seq_lens + plan.query_length
        plan.plan(page_table, final_seq_lens)
        attention_plan.plan(page_table, final_seq_lens)

    def _run_msa_v1_decode_indexer(
        self,
        idx_q: torch.Tensor,
        layer,
    ) -> torch.Tensor:
        plan = self.forward_metadata.msa_v1_decode_plan
        if plan is None:
            raise RuntimeError(
                "MSA-v1 decode metadata was not prepared before the layer forward."
            )
        return plan.run(
            idx_q,
            self.kv_pool.get_index_k_buffer(layer.layer_id),
            self.kv_pool.get_index_k_scale_buffer(layer.layer_id),
            self.kv_pool.index_k_pool.global_scale,
            out=self.forward_metadata.msa_v1_attention_plan.topk_qht.permute(1, 0, 2),
        )

    def _run_msa_v1_decode_attention(
        self,
        q: torch.Tensor,
        topk_indices: torch.Tensor,
        layer,
    ) -> torch.Tensor:
        plan = self.forward_metadata.msa_v1_attention_plan
        if plan is None:
            raise RuntimeError(
                "MSA-v1 decode attention metadata was not prepared before forward."
            )
        k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
        return plan.run(
            q,
            topk_indices,
            k_cache,
            v_cache,
            self.kv_pool.get_main_k_scale_buffer(layer.layer_id),
            self.kv_pool.get_main_v_scale_buffer(layer.layer_id),
            self.kv_pool.main_pool.global_scale,
        )

    def _prepare_msa_v1_prefill_plan(
        self,
        forward_batch: ForwardBatch,
        cu_seqlens: torch.Tensor,
        seq_lens: torch.Tensor,
        total_q: int,
    ) -> MSAV1PrefillPlan:
        page_table = build_msa_v1_page_table(
            self.req_to_token,
            forward_batch.req_pool_indices,
            max_seqlen_k=self._max_seqlen_k,
        )
        self.forward_metadata.msa_v1_prefill_plan = MSAV1PrefillPlan(
            cu_seqlens_q=cu_seqlens,
            seq_lens=seq_lens,
            page_table=page_table,
            total_q=total_q,
            max_seqlen_q=self._max_seqlen_q,
            max_seqlen_k=self._max_seqlen_k,
            prefer_qmul4=self._msa_v1_qmul4_available,
            num_index_heads=self._msa_v1_num_index_heads,
        )
        return self.forward_metadata.msa_v1_prefill_plan
