from __future__ import annotations

from functools import lru_cache

import torch

from sglang.srt.distributed.parallel_state import (
    get_attn_tensor_model_parallel_rank,
    get_attn_tensor_model_parallel_world_size,
)
from sglang.srt.environ import envs
from sglang.srt.layers.communicator import ScatterMode
from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor
from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_output
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.utils import is_npu

if is_npu():
    import torch_npu

    from sglang.srt.hardware_backend.npu.utils import get_indexer_weight_stream

_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()


@lru_cache(maxsize=1)
def _create_hadamard_128_cpu() -> torch.Tensor:
    matrix = [[1.0]]
    while len(matrix) < 128:
        matrix = [row + row for row in matrix] + [
            row + [-value for value in row] for row in matrix
        ]
    return torch.tensor(matrix, dtype=torch.bfloat16)


def create_npu_hadamard_128(head_dim: int, device) -> torch.Tensor:
    assert head_dim == 128
    return (_create_hadamard_128_cpu().to(device=device) / (128**0.5)).contiguous()


def _quantize_npu_indexer_activation(x, hadamard, dst_type):
    assert x.dtype == torch.bfloat16 and x.shape[-1] == 128
    if x.numel() == 0:
        return (
            torch.empty_like(x, dtype=dst_type),
            torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device),
        )
    rotated = x @ hadamard
    quantized, scale = torch_npu.npu_dynamic_quant(
        rotated.reshape(-1, 128), dst_type=dst_type
    )
    return quantized.reshape(x.shape), scale.to(torch.float32).reshape(x.shape[:-1])


class DSANPUIndexerMixin:
    def forward_npu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        layer_scatter_modes=None,
        dynamic_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        backend = get_attn_backend()
        fm = backend.forward_metadata
        if fm is None:
            fm = getattr(
                getattr(backend, "full_attn_backend", None),
                "forward_metadata",
                None,
            )
        if fm is not None:
            if fm.seq_lens_cpu_int is None:
                actual_seq_lengths_kv = fm.seq_lens
            else:
                actual_seq_lengths_kv = fm.seq_lens_cpu_int
        else:
            actual_seq_lengths_kv = forward_batch.seq_lens_cpu.int()
        # During cuda graph capture seq_lens are filled with 0, which causes
        # npu_lightning_indexer to crash (sparse_count > 0 but 0 KV positions).
        # Clamp to 1: a no-op during real inference (active requests always have
        # seq_len >= 1) but keeps the kernel alive inside the captured graph.
        if actual_seq_lengths_kv is not None:
            actual_seq_lengths_kv = actual_seq_lengths_kv.clamp(min=1)
        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_target_verify()
        )

        bs = q_lora.shape[0]

        if self.rotary_emb.is_neox_style:
            if not hasattr(forward_batch, "npu_indexer_sin_cos_cache"):
                cos_sin = self.rotary_emb.cos_sin_cache[positions]
                cos, sin = cos_sin.chunk(2, dim=-1)
                cos = cos.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
                sin = sin.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
                forward_batch.npu_indexer_sin_cos_cache = (sin, cos)
            else:
                sin, cos = forward_batch.npu_indexer_sin_cos_cache

            if self.alt_stream is not None:
                self.alt_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(self.alt_stream):
                    q_lora = (
                        (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                    )
                    q = self.wq_b(q_lora)[
                        0
                    ]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
                    q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
                    q_pe, q_nope = torch.split(
                        q,
                        [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                        dim=-1,
                    )  # [bs, 64, 64 + 64]
                    q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
                    q_pe = torch_npu.npu_rotary_mul(q_pe, cos, sin).view(
                        bs, self.n_heads, self.rope_head_dim
                    )  # [bs, n, d]
                    q = torch.cat([q_pe, q_nope], dim=-1)
                    q.record_stream(self.alt_stream)
                    q_rope_event = self.alt_stream.record_event()
            else:
                q_lora = (
                    (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                )
                q = self.wq_b(q_lora)[
                    0
                ]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
                q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
                q_pe, q_nope = torch.split(
                    q,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )  # [bs, 64, 64 + 64]
                q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
                q_pe = torch_npu.npu_rotary_mul(q_pe, cos, sin).view(
                    bs, self.n_heads, self.rope_head_dim
                )  # [bs, n, d]
                q = torch.cat([q_pe, q_nope], dim=-1)

            if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
                indexer_weight_stream = get_indexer_weight_stream()
                indexer_weight_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(indexer_weight_stream):
                    x = x.view(-1, self.hidden_size)
                    weights = self.weights_proj(x.float())[0].to(torch.bfloat16)
                    weights.record_stream(indexer_weight_stream)
                    weights_event = indexer_weight_stream.record_event()
            else:
                x = x.view(-1, self.hidden_size)
                weights = self.weights_proj(x.float())[0].to(torch.bfloat16)

            k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
            k = self.k_norm(k_proj)
            if (
                _use_ag_after_qlora
                and layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED
                and layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
            ):
                k = scattered_to_tp_attn_full(k, forward_batch)
            k_pe, k_nope = torch.split(
                k,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64 + 64]

            k_pe = k_pe.view(-1, 1, 1, self.rope_head_dim)
            k_pe = torch.ops.npu.npu_rotary_mul(k_pe, cos, sin).view(
                bs, 1, self.rope_head_dim
            )  # [bs, 1, d]
            k = torch.cat([k_pe, k_nope.unsqueeze(1)], dim=-1)  # [bs, 1, 128]

        else:
            if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
                indexer_weight_stream = get_indexer_weight_stream()
                indexer_weight_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(indexer_weight_stream):
                    x = x.view(-1, self.hidden_size)
                    weights = self.weights_proj(x.float())[0].to(torch.bfloat16)
                    weights.record_stream(indexer_weight_stream)
                    weights_event = indexer_weight_stream.record_event()
            else:
                x = x.view(-1, self.hidden_size)
                weights = self.weights_proj(x.float())[0].to(torch.bfloat16)

            q_lora = (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
            q = self.wq_b(q_lora)[0]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
            q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
            q_pe, q_nope = torch.split(
                q,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64, 64 + 64]

            k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
            k = self.k_norm(k_proj)
            k_pe, k_nope = torch.split(
                k,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64 + 64]

            k_pe = k_pe.unsqueeze(1)

            if layer_id == 0:
                self.rotary_emb.sin_cos_cache = (
                    self.rotary_emb.cos_sin_cache.index_select(0, positions)
                )

            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            k_pe = k_pe.squeeze(1)
            q = torch.cat([q_pe, q_nope], dim=-1)
            k = torch.cat([k_pe, k_nope], dim=-1)

        if (
            is_prefill
            and self.dsa_enable_prefill_cp
            and forward_batch.attn_cp_metadata is not None
        ):
            k = cp_all_gather_rerange_output(
                k.contiguous().view(-1, self.head_dim),
                self.cp_size,
                forward_batch,
                torch.npu.current_stream(),
            )

        pool = get_token_to_kv_pool()
        use_quant_indexer = pool.index_k_scale_buffer is not None
        if use_quant_indexer:
            k, k_scale = _quantize_npu_indexer_activation(
                k, pool.indexer_hadamard_128, pool.dtype
            )
            pool.set_index_k_scale_buffer(
                layer_id, forward_batch.out_cache_loc, k_scale
            )
        pool.set_index_k_buffer(layer_id, forward_batch.out_cache_loc, k)
        if is_prefill:
            if (
                fm is not None
                and self.dsa_enable_prefill_cp
                and forward_batch.attn_cp_metadata is not None
            ):
                fm.actual_seq_lengths_q = (
                    forward_batch.attn_cp_metadata.actual_seq_q_prev_tensor,
                    forward_batch.attn_cp_metadata.actual_seq_q_next_tensor,
                )
                if sum(forward_batch.extend_prefix_lens_cpu) > 0:
                    total_kv_len_prev_tensor = (
                        forward_batch.attn_cp_metadata.kv_len_prev_tensor
                        + forward_batch.extend_prefix_lens.squeeze()
                    )
                    total_kv_len_next_tensor = (
                        forward_batch.attn_cp_metadata.kv_len_next_tensor
                        + forward_batch.extend_prefix_lens.squeeze()
                    )
                    fm.actual_seq_lengths_kv = (
                        total_kv_len_prev_tensor,
                        total_kv_len_next_tensor,
                    )
                else:
                    fm.actual_seq_lengths_kv = (
                        forward_batch.attn_cp_metadata.kv_len_prev_tensor,
                        forward_batch.attn_cp_metadata.kv_len_next_tensor,
                    )
                actual_seq_lengths_q = fm.actual_seq_lengths_q
                actual_seq_lengths_kv = fm.actual_seq_lengths_kv
            else:
                actual_seq_lengths_kv = forward_batch.seq_lens
                actual_seq_lengths_q = forward_batch.extend_seq_lens.cumsum(dim=0)
        else:
            if fm is None or fm.actual_seq_lengths_q is None:
                if (
                    forward_batch.forward_mode.is_draft_extend_v2()
                    or forward_batch.forward_mode.is_target_verify()
                ):
                    num_draft_tokens = getattr(
                        get_attn_backend(), "speculative_num_draft_tokens", None
                    )
                    if num_draft_tokens is None:
                        from sglang.srt.runtime_context import get_spec

                        num_draft_tokens = get_spec().speculative_num_draft_tokens
                    actual_seq_lengths_q = torch.arange(
                        num_draft_tokens,
                        num_draft_tokens + bs,
                        num_draft_tokens,
                        dtype=torch.int32,
                        device=k.device,
                    )
                else:
                    actual_seq_lengths_q = torch.tensor(
                        [1 + i * 1 for i in range(bs)],
                        dtype=torch.int32,
                        device=k.device,
                    )
            else:
                actual_seq_lengths_q = fm.actual_seq_lengths_q

        past_key_states = get_token_to_kv_pool().get_index_k_buffer(layer_id)

        if self.rotary_emb.is_neox_style and self.alt_stream is not None:
            torch.npu.current_stream().wait_event(q_rope_event)
        if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
            torch.npu.current_stream().wait_event(weights_event)
        if (
            _use_ag_after_qlora
            and layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED
            and layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
        ):
            weights = scattered_to_tp_attn_full(weights, forward_batch)
        if fm is not None:
            block_table = fm.block_tables
        else:
            backend = get_attn_backend()
            req_pool = forward_batch.req_pool_indices
            max_len = int(forward_batch.seq_lens_cpu.max().item())
            page_size = getattr(backend, "page_size", None)
            if page_size is None:
                page_size = getattr(
                    getattr(backend, "full_attn_backend", None),
                    "page_size",
                    1,
                )
            block_table = (
                backend.req_to_token_pool.req_to_token[
                    req_pool, :max_len:page_size
                ]
                // page_size
            )
        if (
            is_prefill
            and self.dsa_enable_prefill_cp
            and forward_batch.attn_cp_metadata is not None
        ):
            block_table = block_table[: actual_seq_lengths_q[0].numel()]
            topk_indices = self.do_npu_cp_balance_indexer(
                q.view(-1, self.n_heads, self.head_dim),
                past_key_states,
                weights,
                actual_seq_lengths_q,
                actual_seq_lengths_kv,
                block_table,
            )
            return topk_indices
        else:
            block_table = (
                block_table[: actual_seq_lengths_q.size()[0]]
                if is_prefill
                else block_table
            )

            # --- attn-tp batch split for indexer ---
            # Each rank handles a contiguous slice of requests, runs the
            # lightning indexer on its local Q slice against the shared full
            # KV cache, then all-gathers the top-k indices.
            attn_tp_size = get_attn_tensor_model_parallel_world_size()
            q_view = q.view(-1, self.n_heads, self.head_dim)
            total_tokens = q_view.shape[0]
            num_requests = actual_seq_lengths_q.shape[0]
            if attn_tp_size > 1 and num_requests % attn_tp_size == 0:
                attn_tp_rank = get_attn_tensor_model_parallel_rank()
                # num_requests = actual_seq_lengths_q.shape[0]
                assert (
                    num_requests % attn_tp_size == 0
                ), f"batch {num_requests} not divisible by attn_tp_size {attn_tp_size}"
                local_bs = num_requests // attn_tp_size
                req_start = attn_tp_rank * local_bs
                req_end = req_start + local_bs

                if total_tokens % num_requests == 0:
                    # Uniform Q tokens per request (decode / target_verify).
                    # All offsets are Python ints — graph-safe, no .item().
                    tokens_per_req = total_tokens // num_requests
                    token_start = req_start * tokens_per_req
                    token_end = req_end * tokens_per_req
                    q_cumsum_offset = req_start * tokens_per_req
                else:
                    # Variable-length Q (prefill without CP) — never graph-captured,
                    # so .item() synchronization is safe here.
                    token_start = (
                        int(actual_seq_lengths_q[req_start - 1].item())
                        if attn_tp_rank > 0
                        else 0
                    )
                    token_end = int(actual_seq_lengths_q[req_end - 1].item())
                    q_cumsum_offset = token_start

                q_local = q_view[token_start:token_end]
                weights_local = weights[token_start:token_end]
                actual_seq_lengths_q_local = (
                    actual_seq_lengths_q[req_start:req_end] - q_cumsum_offset
                ).to(torch.int32)
                actual_seq_lengths_kv_local = actual_seq_lengths_kv[
                    req_start:req_end
                ]
                block_table_local = block_table[req_start:req_end]
            else:
                q_local = q_view
                weights_local = weights
                actual_seq_lengths_q_local = actual_seq_lengths_q.to(torch.int32)
                actual_seq_lengths_kv_local = actual_seq_lengths_kv
                block_table_local = block_table

            if use_quant_indexer:
                query_local, query_scale_local = _quantize_npu_indexer_activation(
                    q_local,
                    pool.indexer_hadamard_128,
                    pool.dtype,
                )
                topk_indices = torch_npu.npu_quant_lightning_indexer(
                    query=query_local,
                    key=past_key_states,
                    weights=weights_local,
                    query_dequant_scale=query_scale_local,
                    key_dequant_scale=pool.get_index_k_scale_buffer(layer_id),
                    actual_seq_lengths_query=actual_seq_lengths_q_local,
                    actual_seq_lengths_key=actual_seq_lengths_kv_local.to(
                        device=k.device, dtype=torch.int32
                    ),
                    block_table=block_table_local,
                    layout_query="TND",
                    layout_key="PA_BSND",
                    sparse_count=self.index_topk,
                    sparse_mode=3,
                    query_quant_mode=0,
                    key_quant_mode=0,
                )
                topk_indices = topk_indices.squeeze(1)
            else:
                topk_indices = torch_npu.npu_lightning_indexer(
                    query=q_local,
                    key=past_key_states,
                    weights=weights_local,
                    actual_seq_lengths_query=actual_seq_lengths_q_local,
                    actual_seq_lengths_key=actual_seq_lengths_kv_local.to(
                        k.device
                    ).to(torch.int32),
                    block_table=block_table_local,
                    layout_query="TND",
                    layout_key="PA_BSND",
                    sparse_count=self.index_topk,
                    sparse_mode=3,
                )
                # Keep DSA top-k as [T, K]; NPU attention expands it when needed.
                topk_indices = topk_indices[0].squeeze(1)

            if attn_tp_size > 1 and num_requests % attn_tp_size == 0:
                topk_full = torch.empty(
                    (total_tokens, topk_indices.shape[-1]),
                    dtype=topk_indices.dtype,
                    device=topk_indices.device,
                )
                attn_tp_all_gather_into_tensor(
                    topk_full, topk_indices.contiguous()
                )
                return topk_full

            return topk_indices

    def do_npu_cp_balance_indexer(
        self,
        q,
        past_key_states,
        indexer_weights,
        actual_seq_lengths_q,
        actual_seq_lengths_kv,
        block_table,
    ):
        q_prev, q_next = torch.split(q, (q.size(0) + 1) // 2, dim=0)
        weights_prev, weights_next = None, None
        if indexer_weights is not None:
            weights_prev, weights_next = torch.split(
                indexer_weights, (indexer_weights.size(0) + 1) // 2, dim=0
            )
            weights_prev = weights_prev.contiguous().view(-1, weights_prev.shape[-1])
            weights_next = weights_next.contiguous().view(-1, weights_next.shape[-1])

        actual_seq_lengths_q_prev, actual_seq_lengths_q_next = actual_seq_lengths_q
        actual_seq_lengths_kv_prev, actual_seq_lengths_kv_next = actual_seq_lengths_kv

        topk_indices_prev = torch_npu.npu_lightning_indexer(
            query=q_prev,
            key=past_key_states,
            weights=weights_prev,
            actual_seq_lengths_query=actual_seq_lengths_q_prev.to(
                device=q.device, dtype=torch.int32
            ),
            actual_seq_lengths_key=actual_seq_lengths_kv_prev.to(
                device=q.device, dtype=torch.int32
            ),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
        )
        topk_indices_next = torch_npu.npu_lightning_indexer(
            query=q_next,
            key=past_key_states,
            weights=weights_next,
            actual_seq_lengths_query=actual_seq_lengths_q_next.to(
                device=q.device, dtype=torch.int32
            ),
            actual_seq_lengths_key=actual_seq_lengths_kv_next.to(
                device=q.device, dtype=torch.int32
            ),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
        )
        return torch.cat([topk_indices_prev[0], topk_indices_next[0]], dim=0).squeeze(1)


def scattered_to_tp_attn_full(
    hidden_states: torch.Tensor,
    forward_batch,
) -> torch.Tensor:
    hidden_states, local_hidden_states = (
        torch.empty(
            (forward_batch.input_ids.shape[0], hidden_states.shape[1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ),
        hidden_states,
    )
    attn_tp_all_gather_into_tensor(hidden_states, local_hidden_states.contiguous())
    return hidden_states
