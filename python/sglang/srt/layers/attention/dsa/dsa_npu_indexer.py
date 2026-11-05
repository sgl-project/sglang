from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.cp.utils import cp_gather_full_sequence_states
from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_npu

if is_npu():
    import torch_npu

    from sglang.srt.hardware_backend.npu.utils import get_indexer_weight_stream

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
        AscendAttnBackend,
    )

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
    # Match vllm-ascend SFA: BF16 matrix, normalized once on the pool's device.
    return (_create_hadamard_128_cpu().to(device=device) / (128**0.5)).contiguous()


# The vendor quant indexer tiling hardcodes num_heads_q == 64; a narrower indexer
# is padded at the call site, and ascend_backend.py builds its handle at this count.
VENDOR_QUANT_INDEXER_NUM_HEADS = 64
# Any finite non-zero dequant scale works for a padded head: its query is zero,
# so the head contributes nothing whatever the scale is.
_PADDED_HEAD_QUERY_SCALE = 1.0


def _pad_indexer_heads(
    *,
    query: torch.Tensor,
    weights: torch.Tensor,
    query_scale: torch.Tensor,
    num_heads: int,
    padded_num_heads: int,
):
    # query is [T, N, D], weights and query_scale are [T, N].
    assert query.shape[1] == num_heads
    assert weights.shape[1] == num_heads and query_scale.shape[1] == num_heads
    if num_heads > padded_num_heads:
        raise ValueError(
            f"The quantized NPU indexer supports at most {padded_num_heads} "
            f"heads, got {num_heads}."
        )
    if num_heads == padded_num_heads:
        return query, weights, query_scale
    pad = padded_num_heads - num_heads
    rows = query.shape[0]
    # A zero query and a zero weight each suppress a padded head on their own;
    # the 910B P7 probe matched this padding's top-k to a float64 reference.
    return (
        torch.cat([query, query.new_zeros((rows, pad, query.shape[2]))], dim=1),
        torch.cat([weights, weights.new_zeros((rows, pad))], dim=1),
        torch.cat(
            [
                query_scale,
                query_scale.new_full((rows, pad), _PADDED_HEAD_QUERY_SCALE),
            ],
            dim=1,
        ),
    )


def _quantize_npu_indexer_activation(x, hadamard, dst_type, scale_dtype=torch.float32):
    assert x.dtype == torch.bfloat16 and x.shape[-1] == 128
    if x.numel() == 0:
        return (
            torch.empty_like(x, dtype=dst_type),
            torch.empty(x.shape[:-1], dtype=scale_dtype, device=x.device),
        )
    rotated = x @ hadamard
    quantized, scale = torch_npu.npu_dynamic_quant(
        rotated.reshape(-1, 128), dst_type=dst_type
    )
    return quantized.reshape(x.shape), scale.to(scale_dtype).reshape(x.shape[:-1])


def _resolve_int8_indexer_metadata(attn_backend: AscendAttnBackend) -> torch.Tensor:
    fm = attn_backend.forward_metadata
    # A graph binds a handle buffer here; replay refreshes it outside the graph.
    if fm.quant_indexer_metadata is not None:
        return fm.quant_indexer_metadata
    # The vendor metadata op rejects a decode-shape build at the eager
    # init_forward_metadata call site (910B, AICPU error; replay-time rebuilds
    # outside the forward succeed, so the restriction is site-specific, cause
    # unknown to us); AscendAttnBackend._init_quant_indexer_metadata defers it here.
    fm.quant_indexer_metadata = attn_backend.build_quant_indexer_metadata_in_forward()
    return fm.quant_indexer_metadata


class DSANPUIndexerMixin:
    def forward_npu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        input_on_attn_tp_slices: bool = False,
        dynamic_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        # DSA prefill CP metadata is Ascend-local and is first consumed by this
        # indexer (before the attention backend's forward_extend), so prepare it
        # here rather than in the shared eager runner.
        attn_backend = get_attn_backend()
        if hasattr(attn_backend, "prepare_dsa_cp_metadata"):
            attn_backend.prepare_dsa_cp_metadata(forward_batch)
        if get_attn_backend().forward_metadata.seq_lens_cpu_int is None:
            actual_seq_lengths_kv = get_attn_backend().forward_metadata.seq_lens
        else:
            actual_seq_lengths_kv = get_attn_backend().forward_metadata.seq_lens_cpu_int
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
            if _use_ag_after_qlora and input_on_attn_tp_slices:
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

            if layer_id == get_token_to_kv_pool().start_layer:
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
            k = cp_gather_full_sequence_states(
                k.contiguous().view(-1, self.head_dim),
                forward_batch,
                torch.npu.current_stream(),
            )

        indexer_cache_loc = forward_batch.out_cache_loc
        parallel = get_parallel()
        if (
            parallel.dcp_enabled
            and parallel.attn_dcp_size > 1
            and not get_attn_backend().is_draft_worker
        ):
            indexer_cache_loc = (
                get_attn_backend().forward_metadata.dcp_origin_out_cache_loc
            )
            assert indexer_cache_loc is not None, (
                "NPU DSA+DCP requires allocator-global origin_out_cache_loc metadata"
            )
            assert indexer_cache_loc.shape[0] == positions.shape[0], (
                "NPU DSA+DCP origin_out_cache_loc metadata has an incompatible "
                f"length: {indexer_cache_loc.shape[0]} != {positions.shape[0]}"
            )
        pool = get_token_to_kv_pool()
        use_quant_indexer = pool.index_k_scale_buffer is not None
        if use_quant_indexer:
            k, k_scale = _quantize_npu_indexer_activation(
                k,
                pool.indexer_hadamard_128,
                pool.indexer_kv_dtype,
                pool.indexer_scale_dtype,
            )
            pool.set_index_k_scale_buffer(layer_id, indexer_cache_loc, k_scale)
        pool.set_index_k_buffer(layer_id, indexer_cache_loc, k)
        if is_prefill:
            if (
                self.dsa_enable_prefill_cp
                and forward_batch.attn_cp_metadata is not None
            ):
                # V2: local metadata already prepared by prepare_dsa_cp_metadata.
                actual_seq_lengths_q = (
                    get_attn_backend().forward_metadata.actual_seq_lengths_q
                )
                actual_seq_lengths_kv = (
                    get_attn_backend().forward_metadata.actual_seq_lengths_kv
                )
            else:
                actual_seq_lengths_kv = forward_batch.seq_lens
                actual_seq_lengths_q = forward_batch.extend_seq_lens.cumsum(dim=0)
        else:
            if get_attn_backend().forward_metadata.actual_seq_lengths_q is None:
                if (
                    forward_batch.forward_mode.is_draft_extend_v2()
                    or forward_batch.forward_mode.is_target_verify()
                ):
                    num_draft_tokens = get_attn_backend().speculative_num_draft_tokens
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
                actual_seq_lengths_q = (
                    get_attn_backend().forward_metadata.actual_seq_lengths_q
                )

        past_key_states = get_token_to_kv_pool().get_index_k_buffer(layer_id)

        if self.rotary_emb.is_neox_style and self.alt_stream is not None:
            torch.npu.current_stream().wait_event(q_rope_event)
        if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
            torch.npu.current_stream().wait_event(weights_event)
        if _use_ag_after_qlora and input_on_attn_tp_slices:
            weights = scattered_to_tp_attn_full(weights, forward_batch)
        block_table = get_attn_backend().forward_metadata.block_tables
        if (
            is_prefill
            and self.dsa_enable_prefill_cp
            and forward_batch.attn_cp_metadata is not None
        ):
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

            if use_quant_indexer:
                query, query_scale = _quantize_npu_indexer_activation(
                    q.view(-1, self.n_heads, self.head_dim),
                    pool.indexer_hadamard_128,
                    pool.indexer_kv_dtype,
                    pool.indexer_scale_dtype,
                )
                if pool.indexer_kv_dtype == torch.int8:
                    # Idle ranks get no handle from the backend; skip the kernel.
                    if forward_batch.forward_mode.is_idle():
                        return torch.full(
                            (query.shape[0], self.index_topk),
                            -1,
                            dtype=torch.int32,
                            device=query.device,
                        )
                    # INT8 indexer K is answered by the vendor kernel;
                    # it needs a metadata handle the binding below cannot pass.
                    attn_backend = get_attn_backend()
                    metadata = _resolve_int8_indexer_metadata(attn_backend)
                    fm = attn_backend.forward_metadata
                    return self._npu_vendor_quant_indexer(
                        query=query,
                        query_scale=query_scale,
                        key=past_key_states,
                        key_scale=pool.get_index_k_scale_buffer(layer_id),
                        weights=weights,
                        block_table=block_table,
                        metadata=metadata,
                        cu_q_lens=fm.quant_indexer_cu_q_lens,
                        kv_lens=fm.quant_indexer_kv_lens,
                    )
                topk_indices = torch_npu.npu_quant_lightning_indexer(
                    query=query,
                    key=past_key_states,
                    weights=weights,
                    query_dequant_scale=query_scale,
                    key_dequant_scale=pool.get_index_k_scale_buffer(layer_id),
                    actual_seq_lengths_query=actual_seq_lengths_q.to(torch.int32),
                    actual_seq_lengths_key=actual_seq_lengths_kv.to(
                        device=k.device, dtype=torch.int32
                    ),
                    block_table=block_table,
                    layout_query="TND",
                    layout_key="PA_BSND",
                    sparse_count=self.index_topk,
                    sparse_mode=3,
                    query_quant_mode=0,
                    key_quant_mode=0,
                )
                return topk_indices.squeeze(1)

            topk_indices = torch_npu.npu_lightning_indexer(
                query=q.view(-1, self.n_heads, self.head_dim),
                key=past_key_states,
                weights=weights,
                actual_seq_lengths_query=actual_seq_lengths_q.to(torch.int32),
                actual_seq_lengths_key=actual_seq_lengths_kv.to(k.device).to(
                    torch.int32
                ),
                block_table=block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=3,
            )
            # Keep DSA top-k as [T, K]; NPU attention expands it when needed.
            return topk_indices[0].squeeze(1)

    def _npu_vendor_quant_indexer(
        self,
        *,
        query: torch.Tensor,
        query_scale: torch.Tensor,
        key: torch.Tensor,
        key_scale: torch.Tensor,
        weights: torch.Tensor,
        block_table: torch.Tensor,
        metadata: torch.Tensor,
        cu_q_lens: torch.Tensor,
        kv_lens: torch.Tensor,
    ) -> torch.Tensor:
        # The handle is built from exactly cu_q_lens and kv_lens, by the first
        # indexer layer in eager or by the backend outside the graph on replay.
        if metadata is None:
            raise RuntimeError(
                "The INT8 DSA NPU indexer found no quant indexer metadata on the "
                "attention backend's forward metadata for this forward mode."
            )
        # The vendor op takes FP16 weights and FP16 for both dequant scales;
        # INT8 query / key is the only signature it accepts.
        query, weights, query_scale = _pad_indexer_heads(
            query=query,
            weights=weights.to(torch.float16),
            query_scale=query_scale.to(torch.float16),
            num_heads=self.n_heads,
            padded_num_heads=VENDOR_QUANT_INDEXER_NUM_HEADS,
        )
        # The literals below must match the handle kwargs in ascend_backend.py.
        topk_indices, _ = torch.ops.custom.npu_quant_lightning_indexer(
            query=query,
            key=key,
            key_dequant_scale=key_scale.to(torch.float16),
            weights=weights,
            query_dequant_scale=query_scale,
            actual_seq_lengths_query=cu_q_lens,
            actual_seq_lengths_key=kv_lens,
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            cmp_ratio=1,
            query_quant_mode=0,
            key_quant_mode=0,
            sparse_mode=3,
            sparse_count=self.index_topk,
            metadata=metadata,
        )
        # Keep DSA top-k as [T, K]; sparse FA drops the -1 slots,
        # which the in-chunk causal mask leaves on early prefill rows.
        return topk_indices.view(-1, self.index_topk)

    def do_npu_cp_balance_indexer(
        self,
        q,
        past_key_states,
        indexer_weights,
        actual_seq_lengths_q,
        actual_seq_lengths_kv,
        block_table,
    ):
        # V2: Q is local (sharded); single indexer call against full KV.
        if indexer_weights is not None:
            indexer_weights = indexer_weights.contiguous().view(
                -1, indexer_weights.shape[-1]
            )
        topk_indices = torch_npu.npu_lightning_indexer(
            query=q,
            key=past_key_states,
            weights=indexer_weights,
            actual_seq_lengths_query=actual_seq_lengths_q.to(
                device=q.device, dtype=torch.int32
            ),
            actual_seq_lengths_key=actual_seq_lengths_kv.to(
                device=q.device, dtype=torch.int32
            ),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
        )
        return topk_indices[0].squeeze(1)


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
