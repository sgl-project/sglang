from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_cp import get_dsa_cp_plan
from sglang.srt.layers.communicator import ScatterMode
from sglang.srt.layers.cp.utils import cp_gather_full_sequence_states
from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_npu, print_info_once

if is_npu():
    import torch_npu

    from sglang.srt.hardware_backend.npu.utils import get_indexer_weight_stream

_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()
_shard_indexer_queries = envs.SGLANG_NPU_ENABLE_DSA_INDEXER_QUERY_SHARDING.get()


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


def plan_indexer_query_shard(
    prefix_lens: List[int], extend_lens: List[int], tp_size: int, tp_rank: int
):
    """One attention-TP rank's share of an extend batch's indexer queries.

    Follows vLLM-Ascend DSA-CP (``_prepare_parallel_metadata`` in ``sfa_cp.py``):
    the batch's flat token rows are padded to a multiple of ``tp_size`` and rank
    r owns rows ``[r * rows, (r + 1) * rows)``. A request's local query count is
    its overlap with that range, and its key length ends at its last local
    token -- prefix plus local end -- which is what ``sparse_mode=3``'s
    right-down causal crop needs. A request with no token here gets key length 0.

    Returns ``(start, rows, num_real, cum_query_lens, key_lens)``; the first
    ``num_real`` of the rank's ``rows`` rows are real tokens.
    """
    total = sum(extend_lens)
    rows = -(-total // tp_size)
    start = tp_rank * rows
    end = start + rows
    num_real = max(0, min(end, total) - start)
    cum_query_lens, key_lens = [], []
    num_local = 0
    req_start = 0
    for prefix_len, extend_len in zip(prefix_lens, extend_lens):
        req_end = req_start + extend_len
        local_end = min(req_end, end)
        req_local = max(0, local_end - max(req_start, start))
        num_local += req_local
        cum_query_lens.append(num_local)
        key_lens.append(prefix_len + local_end - req_start if req_local else 0)
        req_start = req_end
    return start, rows, num_real, cum_query_lens, key_lens


@dataclass
class _IndexerQueryShard:
    """This rank's rows of an extend batch, and how to reassemble the top-k."""

    start: int
    rows: int
    num_real: int
    total: int
    tp_size: int
    actual_seq_lengths_q: torch.Tensor
    actual_seq_lengths_kv: torch.Tensor

    def take(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_real == self.rows:
            return x[self.start : self.start + self.rows]
        # Padding rows lie past actual_seq_lengths_q[-1]. The operator leaves
        # uninitialized values in their top-k, which gather() slices off.
        out = x.new_zeros((self.rows, *x.shape[1:]))
        out[: self.num_real] = x[self.start : self.start + self.num_real]
        return out

    def gather(self, topk_indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """All-gather the per-rank top-k back to ``num_tokens`` rows.

        ``num_tokens`` is the width of the query tensor this call was sliced
        from: ``total`` padded up to a multiple of ``tp_size`` (see the gate in
        ``_get_indexer_query_shard``). ``rows * tp_size`` is that same padded
        width, so the gather buffer is already the right length and the slice
        only trims when the caller passed an unpadded tensor.
        """
        out = topk_indices.new_empty((self.rows * self.tp_size, topk_indices.shape[-1]))
        attn_tp_all_gather_into_tensor(out, topk_indices.contiguous())
        return out[:num_tokens]

    def resolve(
        self, topk_indices: torch.Tensor, num_tokens: int, forward_batch
    ) -> torch.Tensor:
        """Full-width top-k, or this rank's own rows when DSA-CP will slice anyway.

        Handoff §8 W2. Under DSA-CP the gather above is pure waste: attention
        immediately cuts the gathered tensor back to ``[local_start, local_end)``
        with ``dsa_cp_slice``, and that is the same row range this rank already
        holds. Both planners compute ``rows = ceil(total / tp_size)`` and
        ``start = rank * rows``, which
        ``test/registered/dcp/test_dsa_cp_indexer_row_agreement.py`` pins --
        **that test is what makes this safe, and it is why it landed first.**
        Until now the agreement was incidental, because the gather hid any
        disagreement; skipping the gather makes it load-bearing.

        Removes one all-gather of ``index_topk`` int32 per token, and its sync
        point, from each indexer layer.

        Sets ``forward_batch.npu_indexer_topk_is_local`` so the attention side
        knows which of the two it was handed. The flag describes the most
        recently produced top-k, which is exactly the tensor the skip-topk
        layers pass along -- and a layer that declines to shard rewrites it to
        False before its own attention runs.

        Rows past ``num_real`` are zeroed rather than left as the operator wrote
        them. The gather-then-slice path fills them with zeros (``dsa_cp_slice``
        pads), so this keeps the result bitwise identical rather than merely
        equivalent. It is at most ``tp_size - 1`` rows, on the last rank only.
        """
        plan = get_dsa_cp_plan(forward_batch)
        local = plan is not None
        if local and (plan.rows != self.rows or plan.local_start != self.start):
            # Cannot happen while the row-agreement test passes. Gather anyway
            # rather than fail: the gathered path stays correct either way.
            print_info_once(
                "DSA-CP and indexer sharding disagree on the row range "
                f"(plan {plan.local_start}+{plan.rows} against shard "
                f"{self.start}+{self.rows}); keeping the top-k all-gather"
            )
            local = False
        forward_batch.npu_indexer_topk_is_local = local
        if not local:
            return self.gather(topk_indices, num_tokens)
        # Deliberately not gated on attn_tp_rank, so a grep returns the rank
        # count and not 1. Without this line a W2 that silently never engaged
        # would look exactly like a W2 that worked: the output is identical
        # either way, and only the timing would differ.
        print_info_once(
            "DSA-CP W2: skipping the indexer top-k all-gather; each rank keeps "
            "the rows it scored"
        )
        if self.num_real < self.rows:
            topk_indices[self.num_real :] = 0
        return topk_indices


def _build_indexer_query_shard(
    forward_batch: ForwardBatch,
) -> Optional[_IndexerQueryShard]:
    parallel = get_parallel()
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    extend_lens = forward_batch.extend_seq_lens_cpu
    if (
        parallel.attn_tp_size == 1
        or forward_batch.forward_mode not in (ForwardMode.EXTEND, ForwardMode.MIXED)
        or prefix_lens is None
        or extend_lens is None
        or sum(extend_lens) == 0
    ):
        return None
    start, rows, num_real, cum_query_lens, key_lens = plan_indexer_query_shard(
        prefix_lens, extend_lens, parallel.attn_tp_size, parallel.attn_tp_rank
    )
    if parallel.attn_tp_rank == 0:
        # The switch is an env var and /server_info cannot show it, so this line
        # is the evidence that a run was sharded at all.
        print_info_once(
            "DSA indexer query sharding is active: prefill indexer queries split "
            f"across attn_tp_size={parallel.attn_tp_size}"
        )
    device = forward_batch.seq_lens.device
    return _IndexerQueryShard(
        start=start,
        rows=rows,
        num_real=num_real,
        total=sum(extend_lens),
        tp_size=parallel.attn_tp_size,
        actual_seq_lengths_q=torch.tensor(
            cum_query_lens, dtype=torch.int32, device=device
        ),
        actual_seq_lengths_kv=torch.tensor(key_lens, dtype=torch.int32, device=device),
    )


def _get_indexer_query_shard(
    forward_batch: ForwardBatch, num_tokens: int, layer_scatter_modes
) -> Optional[_IndexerQueryShard]:
    """The query shard for a prefill indexer call, or None to score every row.

    Every attention-TP rank holds the same full batch and has already written
    the full index-K cache, so each rank can score only its own
    ``1/attn_tp_size`` of the queries and all-gather the top-k. The kernel's
    work per call drops by the TP size -- measured 15.97x at 1M context and
    TP 16 on A3 -- and the indexer is the part of long-context prefill that
    grows with n^2. Top-k sets match the unsharded call row for row; order
    within a row may differ, which sparse attention does not see.

    Planned once per forward. Every input to the decision is identical across
    the attention-TP group, so all ranks take the collective or none do.
    """
    if (
        layer_scatter_modes is not None
        and layer_scatter_modes.attn_mode != ScatterMode.TP_ATTN_FULL
    ):
        # The plan assumes this rank was handed the whole batch, which is what
        # TP_ATTN_FULL means; any other mode would slice a slice.
        print_info_once(
            "DSA indexer query sharding is off: attention scatter mode is "
            f"{layer_scatter_modes.attn_mode}, not TP_ATTN_FULL"
        )
        return None
    if not hasattr(forward_batch, "npu_indexer_query_shard"):
        forward_batch.npu_indexer_query_shard = _build_indexer_query_shard(
            forward_batch
        )
    shard = forward_batch.npu_indexer_query_shard
    if shard is None:
        return None
    # SGLang pads the query width to a multiple of attn_tp_size, which is what
    # ``rows * tp_size`` is, so admit anything inside [total, padded]. Comparing
    # against total alone disables sharding for 15 token counts in 16.
    padded = shard.rows * shard.tp_size
    if not shard.total <= num_tokens <= padded:
        print_info_once(
            "DSA indexer query sharding is off for this forward: "
            f"num_tokens={num_tokens} is outside [{shard.total}, {padded}]"
        )
        return None
    return shard


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

        # W3 (handoff §8): when this rank will only SCORE its own rows, project
        # only those rows too. ``wq_b`` is [T, 1536] @ [1536, 64 * 128], 25.2
        # MFLOP per token against ``weights_proj``'s 0.46, so slicing q here
        # removes (tp - 1)/tp of the indexer's projection work. The k path below
        # stays full width on purpose: every attention-TP rank writes the whole
        # index-K cache, which is what lets any of them score a subset.
        #
        # Neox only. The other branch rotates q and k in one
        # ``rotary_emb(positions, q_pe, k_pe)`` call, so q cannot take a
        # different row slice than k without splitting that call in two, and
        # ``is_neox_style`` is ``not config.indexer_rope_interleave``, a key
        # GLM-5.2's config does not set.
        #
        # Row-wise exact: every output row of a Linear and of the rotary depends
        # only on its own input row, so the result is bitwise what the late
        # ``shard.take`` produced -- fewer rows change the tiling, not the
        # reduction each row sums over.
        # Prefill CP is excluded explicitly. It has its own indexer
        # (``do_npu_cp_balance_indexer``) further down, which knows nothing about
        # this shard and would be handed a q that is a slice of a slice. Today CP
        # forces attn_tp_size to 1 and the planner declines on its own, but that
        # is a property of the tp8/cp8 layout, not a guarantee -- cp2 at tp8
        # leaves attn_tp at 4.
        uses_prefill_cp = (
            self.dsa_enable_prefill_cp and forward_batch.attn_cp_metadata is not None
        )
        shard = (
            _get_indexer_query_shard(forward_batch, bs, layer_scatter_modes)
            if is_prefill and _shard_indexer_queries and not uses_prefill_cp
            else None
        )
        q_sliced_early = shard is not None and self.rotary_emb.is_neox_style
        if q_sliced_early:
            if dynamic_scale is not None:
                dynamic_scale = shard.take(dynamic_scale)
            q_lora = shard.take(q_lora)
            bs_q = shard.rows
        else:
            bs_q = bs

        if self.rotary_emb.is_neox_style:
            if not hasattr(forward_batch, "npu_indexer_sin_cos_cache"):
                cos_sin = self.rotary_emb.cos_sin_cache[positions]
                cos, sin = cos_sin.chunk(2, dim=-1)
                cos = cos.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
                sin = sin.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
                forward_batch.npu_indexer_sin_cos_cache = (sin, cos)
            else:
                sin, cos = forward_batch.npu_indexer_sin_cos_cache

            # q rotates on this rank's rows; k still needs every one of them.
            sin_q, cos_q = (
                (shard.take(sin), shard.take(cos)) if q_sliced_early else (sin, cos)
            )

            if self.alt_stream is not None:
                self.alt_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(self.alt_stream):
                    q_lora = (
                        (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                    )
                    q = self.wq_b(q_lora)[
                        0
                    ]  # [rows, 1536] @ [1536, 64 * 128] = [rows, 64 * 128]
                    q = q.view(bs_q, self.n_heads, self.head_dim)  # [rows, 64, 128]
                    q_pe, q_nope = torch.split(
                        q,
                        [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                        dim=-1,
                    )  # [rows, 64, 64 + 64]
                    q_pe = q_pe.view(bs_q, self.n_heads, 1, self.rope_head_dim)
                    q_pe = torch_npu.npu_rotary_mul(q_pe, cos_q, sin_q).view(
                        bs_q, self.n_heads, self.rope_head_dim
                    )  # [rows, n, d]
                    q = torch.cat([q_pe, q_nope], dim=-1)
                    q.record_stream(self.alt_stream)
                    q_rope_event = self.alt_stream.record_event()
            else:
                q_lora = (
                    (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                )
                q = self.wq_b(q_lora)[
                    0
                ]  # [rows, 1536] @ [1536, 64 * 128] = [rows, 64 * 128]
                q = q.view(bs_q, self.n_heads, self.head_dim)  # [rows, 64, 128]
                q_pe, q_nope = torch.split(
                    q,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )  # [rows, 64, 64 + 64]
                q_pe = q_pe.view(bs_q, self.n_heads, 1, self.rope_head_dim)
                q_pe = torch_npu.npu_rotary_mul(q_pe, cos_q, sin_q).view(
                    bs_q, self.n_heads, self.rope_head_dim
                )  # [rows, n, d]
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
                self.dsa_enable_prefill_cp
                and forward_batch.attn_cp_metadata is not None
            ):
                get_attn_backend().forward_metadata.actual_seq_lengths_q = (
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
                    get_attn_backend().forward_metadata.actual_seq_lengths_kv = (
                        total_kv_len_prev_tensor,
                        total_kv_len_next_tensor,
                    )
                else:
                    get_attn_backend().forward_metadata.actual_seq_lengths_kv = (
                        forward_batch.attn_cp_metadata.kv_len_prev_tensor,
                        forward_batch.attn_cp_metadata.kv_len_next_tensor,
                    )
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
        if (
            _use_ag_after_qlora
            and layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED
            and layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
        ):
            weights = scattered_to_tp_attn_full(weights, forward_batch)
        block_table = get_attn_backend().forward_metadata.block_tables
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
            query = q.view(-1, self.n_heads, self.head_dim)
            # The full width, for the gather to rebuild -- NOT query.shape[0],
            # which W3 has already cut to shard.rows when it slices q early.
            num_query_tokens = bs
            if shard is not None:
                if not q_sliced_early:
                    query = shard.take(query)
                # weights stays a late take: under _use_ag_after_qlora with a
                # SCATTERED input it is all-gathered to full width above, so
                # slicing it before that would cut a slice twice. It is 0.46
                # MFLOP per token against wq_b's 25.2, so ~2% of W3's saving.
                weights = shard.take(weights)
                actual_seq_lengths_q = shard.actual_seq_lengths_q
                actual_seq_lengths_kv = shard.actual_seq_lengths_kv

            if use_quant_indexer:
                query, query_scale = _quantize_npu_indexer_activation(
                    query,
                    pool.indexer_hadamard_128,
                    pool.dtype,
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
                ).squeeze(1)
            else:
                topk_indices = torch_npu.npu_lightning_indexer(
                    query=query,
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
                )[0].squeeze(1)
            if shard is not None:
                topk_indices = shard.resolve(
                    topk_indices, num_query_tokens, forward_batch
                )
            else:
                forward_batch.npu_indexer_topk_is_local = False
            # Keep DSA top-k as [T, K]; NPU attention expands it when needed.
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
