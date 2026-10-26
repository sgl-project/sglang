from __future__ import annotations

from functools import lru_cache

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.attention.tree_sparse_indices import (
    DRAFT_TREE_ANCESTORS_ATTR,
    DRAFT_TREE_PREFIX_ATTR,
    ancestor_width_for,
    draft_ancestors,
    gather_tree_blocks,
    merge_tree_sparse_indices,
)
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
        input_on_attn_tp_slices: bool = False,
        dynamic_scale: torch.Tensor = None,
        tree_verify: bool = False,
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
                k, pool.indexer_hadamard_128, pool.dtype
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

            if tree_verify:
                return self._build_tree_topk_indices(
                    q=q,
                    weights=weights,
                    pool=pool,
                    layer_id=layer_id,
                    block_table=block_table,
                    actual_seq_lengths_q=actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    forward_batch=forward_batch,
                    use_quant_indexer=use_quant_indexer,
                )

            draft_tree_ancestors = getattr(
                forward_batch, DRAFT_TREE_ANCESTORS_ATTR, None
            )
            if draft_tree_ancestors is not None:
                return self._build_draft_tree_topk_indices(
                    q=q,
                    weights=weights,
                    pool=pool,
                    layer_id=layer_id,
                    block_table=block_table,
                    actual_seq_lengths_q=actual_seq_lengths_q,
                    forward_batch=forward_batch,
                    ancestors=draft_tree_ancestors,
                    use_quant_indexer=use_quant_indexer,
                )

            if use_quant_indexer:
                query, query_scale = _quantize_npu_indexer_activation(
                    q.view(-1, self.n_heads, self.head_dim),
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

    def _run_lightning_indexer(
        self,
        *,
        q: torch.Tensor,
        weights: torch.Tensor,
        pool,
        layer_id: int,
        block_table: torch.Tensor,
        actual_seq_lengths_q: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        sparse_count: int,
        use_quant_indexer: bool,
        sparse_mode: int = 3,
    ) -> torch.Tensor:
        """Fused lightning indexer with a caller-chosen key length / width.

        The tree path uses ``sparse_mode=0`` over a prefix-only key range:
        mode 3 is right-down causal and aligns the query block to the END of the
        key range, so shrinking the key range from P+N to P would silently clip
        the last N-1 prefix tokens.
        """
        key_len = actual_seq_lengths_key.to(device=q.device, dtype=torch.int32)
        query = q.view(-1, self.n_heads, self.head_dim)

        if use_quant_indexer:
            query, query_scale = _quantize_npu_indexer_activation(
                query, pool.indexer_hadamard_128, pool.dtype
            )
            out = torch_npu.npu_quant_lightning_indexer(
                query=query,
                key=pool.get_index_k_buffer(layer_id),
                weights=weights,
                query_dequant_scale=query_scale,
                key_dequant_scale=pool.get_index_k_scale_buffer(layer_id),
                actual_seq_lengths_query=actual_seq_lengths_q.to(torch.int32),
                actual_seq_lengths_key=key_len,
                block_table=block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=sparse_count,
                sparse_mode=sparse_mode,
                query_quant_mode=0,
                key_quant_mode=0,
            )
            return out.squeeze(1)

        out = torch_npu.npu_lightning_indexer(
            query=query,
            key=pool.get_index_k_buffer(layer_id),
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_q.to(torch.int32),
            actual_seq_lengths_key=key_len,
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=sparse_count,
            sparse_mode=sparse_mode,
        )
        return out[0].squeeze(1)

    def _build_tree_topk_indices(
        self,
        *,
        q: torch.Tensor,
        weights: torch.Tensor,
        pool,
        layer_id: int,
        block_table: torch.Tensor,
        actual_seq_lengths_q: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
        forward_batch: ForwardBatch,
        use_quant_indexer: bool,
    ) -> torch.Tensor:
        """Sparse indices that respect the EAGLE draft-tree topology.

        ``row_r = [ ancestors(r) | prefix top-* | -1 padding ]``: see
        ``hardware_backend/npu/attention/tree_sparse_indices.py`` for why the
        order matters (SFA truncates the tail under its causal budget, and
        ``-1`` terminates the scan) and why no position may repeat (SFA does
        not deduplicate, so a repeated position double-counts in the softmax).
        """
        from sglang.srt.runtime_context import get_spec

        spec = forward_batch.spec_info
        if spec is None or getattr(spec, "custom_mask", None) is None:
            raise RuntimeError("tree_verify=True requires spec_info.custom_mask")

        n = int(spec.draft_token_num)
        total = int(q.shape[0])
        if n <= 1 or total % n:
            raise ValueError(
                f"tree verify geometry mismatch: total_tokens={total}, N={n}"
            )

        topk = int(self.index_topk)
        steps = int(
            getattr(spec, "spec_steps", None)
            or get_spec().speculative_num_steps
            or 0
        )
        a = ancestor_width_for(steps, n)
        prefix_width = topk - a
        if prefix_width <= 0:
            raise ValueError(
                f"index_topk={topk} is too small for ancestor_width={a}"
            )

        prefix_lens = (
            actual_seq_lengths_kv.to(device=q.device, dtype=torch.int32) - n
        )
        # actual_seq_lengths_kv is padded to the batch bucket, while the verify
        # mask only carries blocks for the real requests (bs = rows / N). Keeping
        # the padding rows makes the expected mask size over-count by N*N each.
        bs_verify = total // n
        if prefix_lens.numel() > bs_verify:
            prefix_lens = prefix_lens[:bs_verify]
        if prefix_lens.numel() and int(prefix_lens.min()) < 0:
            raise ValueError(
                "prefix length underflow: KV length smaller than the draft block"
            )

        # (1) lightning top-k over the prefix only, leaving A slots for ancestors.
        prefix_idx = self._run_lightning_indexer(
            q=q,
            weights=weights,
            pool=pool,
            layer_id=layer_id,
            block_table=block_table,
            actual_seq_lengths_q=actual_seq_lengths_q,
            actual_seq_lengths_key=prefix_lens,
            sparse_count=prefix_width,
            use_quant_indexer=use_quant_indexer,
            sparse_mode=0,
        )
        if prefix_idx.shape[0] != total:
            raise RuntimeError(
                f"prefix indexer returned {prefix_idx.shape[0]} rows, "
                f"expected {total}"
            )
        if prefix_idx.shape[1] < prefix_width:
            raise RuntimeError(
                f"prefix indexer returned width {prefix_idx.shape[1]}, "
                f"expected at least {prefix_width}"
            )
        if prefix_idx.shape[1] > prefix_width:
            raise RuntimeError(
                "prefix indexer rounded sparse_count up; the ordering contract "
                "(score vs position) must be settled before truncating"
            )

        # (2) The tree topology is already in the mask the tree kernel wrote: a
        # row's visible set is exactly {self} | ancestors, of size depth + 1,
        # which never exceeds A = num_steps + 1. Nothing has to be ranked, so
        # read it and let the merge shift the prefix block per row. Ranking was
        # both unnecessary and impossible to pad safely: padding the unused
        # reserved slots with the node itself repeats a KV position, and SFA
        # reads the list verbatim.
        visible_blocks = gather_tree_blocks(spec.custom_mask, prefix_lens, n)
        ancestor_ids = draft_ancestors(visible_blocks, a)

        # (3) ancestors first, then the prefix block shifted per row.
        return merge_tree_sparse_indices(
            prefix_idx,
            ancestor_ids.reshape(total, -1),
            prefix_lens,
            n,
            topk,
        )

    def _build_draft_tree_topk_indices(
        self,
        *,
        q: torch.Tensor,
        weights: torch.Tensor,
        pool,
        layer_id: int,
        block_table: torch.Tensor,
        actual_seq_lengths_q: torch.Tensor,
        forward_batch: ForwardBatch,
        ancestors: torch.Tensor,
        use_quant_indexer: bool,
    ) -> torch.Tensor:
        """Sparse indices for one draft-decode step of a DSA model (topk > 1).

        ``ancestors`` is built by the draft loop and already holds absolute KV
        positions, ordered ``[.. ancestors .., self]``, one row per candidate.
        Every candidate of a level has the same number of ancestors, so unlike
        verify there is no per-row padding: the prefix block simply follows and
        the row is exactly ``index_topk`` wide.
        """
        prefix_len = getattr(forward_batch, DRAFT_TREE_PREFIX_ATTR, None)
        if prefix_len is None:
            raise RuntimeError(
                "draft tree sparse indices need forward_batch."
                f"{DRAFT_TREE_PREFIX_ATTR}"
            )

        topk = int(self.index_topk)
        rows = int(q.shape[0])
        if int(ancestors.shape[0]) != rows:
            raise RuntimeError(
                f"draft tree ancestors have {int(ancestors.shape[0])} rows, "
                f"expected one per query ({rows})"
            )
        anc = ancestors.reshape(rows, -1).to(torch.int32)
        ancestor_width = int(anc.shape[1])
        prefix_width = topk - ancestor_width
        if prefix_width <= 0:
            raise ValueError(
                f"index_topk={topk} is too small for "
                f"ancestor_width={ancestor_width}"
            )

        prefix_idx = self._run_lightning_indexer(
            q=q,
            weights=weights,
            pool=pool,
            layer_id=layer_id,
            block_table=block_table,
            actual_seq_lengths_q=actual_seq_lengths_q,
            actual_seq_lengths_key=prefix_len,
            sparse_count=prefix_width,
            use_quant_indexer=use_quant_indexer,
            sparse_mode=0,
        )
        if int(prefix_idx.shape[0]) != rows:
            raise RuntimeError(
                f"prefix indexer returned {int(prefix_idx.shape[0])} rows, "
                f"expected {rows}"
            )
        if int(prefix_idx.shape[1]) != prefix_width:
            raise RuntimeError(
                f"prefix indexer returned width {int(prefix_idx.shape[1])}, "
                f"expected {prefix_width}"
            )
        return torch.cat([anc, prefix_idx.to(torch.int32)], dim=-1)

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
