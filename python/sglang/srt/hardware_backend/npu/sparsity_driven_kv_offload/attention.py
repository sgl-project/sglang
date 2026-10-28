"""Ascend attention path backed by sparsity-driven KV offload."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Optional

import torch
import torch_npu

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager import (
    _wait_stream_event,
    normalize_batch_topk_indices,
    unidex_copy_inplace,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.mtp import (
    is_sparse_mtp_batch,
    mtp_compact_topk,
    mtp_selected_kv_view,
    mtp_topk_rows,
    prepare_mtp_rows,
)
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
        AscendAttnBackend,
    )
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def _get_sparse_kv_manager(backend: AscendAttnBackend):
    if backend.sparse_kv_manager is None:
        raise RuntimeError(
            "Sparsity-driven KV offload is disabled or was not initialized."
        )
    return backend.sparse_kv_manager


def _expand_dsa_sparse_indices(topk_indices: torch.Tensor) -> torch.Tensor:
    """Expand [T, K] to [T, 1, K] for NPU sparse attention."""
    if topk_indices.dim() == 2:
        return topk_indices.unsqueeze(-2)
    return topk_indices


def _copy_sfa_decode_eligible(backend, forward_batch, layer, q, topk_indices):
    manager = backend.sparse_kv_manager
    if not envs.SGLANG_NPU_ENABLE_COPY_SFA.get() or manager is None:
        return False
    if (
        not forward_batch.forward_mode.is_decode()
        or getattr(forward_batch, "spec_info", None) is not None
    ):
        return False
    if not getattr(forward_batch, "sparse_kv_copy_sfa_capture", False):
        seq_lens_cpu = backend.forward_metadata.seq_lens_cpu_int
        if seq_lens_cpu is None:
            return False
        live = [length for length in seq_lens_cpu if int(length) > 0]
        if not live or min(map(int, live)) < manager.cache_capacity:
            return False
    return (
        layer.tp_k_head_num == 1
        and backend.kv_lora_rank == 512
        and backend.qk_rope_head_dim == 64
        and q.dtype in (torch.bfloat16, torch.float16)
        and normalize_batch_topk_indices(topk_indices).shape[1] == 2048
        and 2048 <= manager.cache_capacity <= 32640
        and manager.cache_capacity % 128 == 0
        and manager.sparse_context_len == 2048
        and q.shape[0] == forward_batch.batch_size
    )


def _forward_sparse_mtp(backend, q, q_rope, layer, forward_batch, topk_indices):
    """Consume chain queries in order, reusing the bounded Q1 KV workspace.

    The native LIM planner updates residency before attention copies payloads.
    Its Q1 plans cannot be generated for all queries up front on one request.
    MTP uses the vendor causal indexer, then plans/copies each query in order.
    """
    manager = _get_sparse_kv_manager(backend)
    batch = forward_batch.batch_size
    num_tokens = q.shape[0]
    requests, positions, live = prepare_mtp_rows(
        forward_batch, q.shape[0], manager.size, manager.logical_max_context_len
    )
    width = positions.shape[1]
    logical_tokens = batch * width
    source_rows = mtp_topk_rows(normalize_batch_topk_indices(topk_indices), num_tokens)
    sources = source_rows[:logical_tokens].view(batch, width, source_rows.shape[-1])
    top_k = sources.shape[-1]
    if top_k > manager.sparse_context_len:
        raise ValueError("Sparse MTP top-k exceeds the selected KV workspace")
    q = q[:logical_tokens].view(batch, width, -1, backend.kv_lora_rank)
    q_rope = q_rope[:logical_tokens].view(batch, width, -1, backend.qk_rope_head_dim)
    output = q.new_empty((batch, width, layer.tp_q_head_num * backend.kv_lora_rank))
    # A rejected draft may already have populated a resident slot. Rewriting
    # host KV at that absolute position must not leave a hit on the old value.
    manager.invalidate_mtp_cache(layer, forward_batch.req_pool_indices)
    cpu_lengths = forward_batch.seq_lens_cpu
    use_copy_sfa = (
        manager.copy_sfa_enabled
        # The standalone cache planner currently implements C=K only.
        and manager.cache_capacity == 2048
        # Draft/verify graphs also replay short prompts. Keep their dispatch
        # independent of capture-time CPU lengths; Q1 Copy-SFA is fixed-window.
        and not backend.graph_mode
        and top_k == 2048
        and cpu_lengths is not None
        and not any(0 < int(n) < 2048 for n in cpu_lengths.tolist())
    )
    manager.last_decode_path = "mtp_copy_sfa" if use_copy_sfa else "mtp_reference"
    forward_batch.sparse_kv_decode_path = manager.last_decode_path
    for column in range(width):
        source = sources[:, column].contiguous()
        # SFA expects valid sparse entries before its -1 suffix. IndexShare
        # can put future entries anywhere; retain the vendor's valid KV order.
        source = mtp_compact_topk(source, positions[:, column], live[:, column])
        valid = source >= 0
        if use_copy_sfa:
            # copy_sfa_decode's output is shared scratch; copy it before the
            # next query overwrites that storage. No native LIM plan is used.
            query_batch = copy(forward_batch)
            query_batch.seq_lens = torch.where(
                live[:, column], positions[:, column] + 1, 0
            )
            query_nope = q[:, column : column + 1]
            query_rope = q_rope[:, column : column + 1]
            padding = manager.planner_attn_out.shape[1] - query_nope.shape[2]
            if padding > 0:
                query_nope = torch.nn.functional.pad(query_nope, (0, 0, 0, padding))
                query_rope = torch.nn.functional.pad(query_rope, (0, 0, 0, padding))
            result = manager.copy_sfa_decode(
                layer,
                query_batch,
                query_nope,
                query_rope,
                layer.tp_q_head_num,
                layer.scaling,
                source,
            )
        else:
            selected = mtp_selected_kv_view(manager.selected_kv_buffer, batch, top_k)
            selected.zero_()
            safe_requests = requests[:, column].clamp(0, manager.size - 1)
            layer_index = layer.layer_id - manager.start_layer
            unidex_copy_inplace(
                manager.host_kv_buffer[layer_index],
                selected,
                (safe_requests[:, None] * manager.max_context_len + source)
                .reshape(-1)
                .contiguous(),
                torch.arange(batch * top_k, device=q.device),
                valid.reshape(-1).contiguous(),
                2,
                2,
                block_dim=24,
                src_ptr=manager.dev_ptr_list[layer_index],
            )
            compact = torch.arange(top_k, device=q.device, dtype=torch.int32)
            indices = compact[None].expand(batch, -1).masked_fill(~valid, -1)
            nonempty = valid.any(dim=1)
            # The SFA ABI needs a dummy KV on empty/padded query rows.
            indices[:, 0] = torch.where(nonempty, indices[:, 0], 0)
            k_nope, k_rope = selected.split(
                [backend.kv_lora_rank, backend.qk_rope_head_dim], dim=-1
            )
            ret = torch_npu.npu_sparse_flash_attention(
                q[:, column : column + 1].contiguous(),
                k_nope.contiguous(),
                k_nope.contiguous(),
                indices[:, None, None].contiguous(),
                layer.scaling,
                actual_seq_lengths_query=torch.ones(
                    batch, dtype=torch.int32, device=q.device
                ),
                actual_seq_lengths_kv=valid.sum(dim=1).clamp(min=1).int(),
                query_rope=q_rope[:, column : column + 1].contiguous(),
                key_rope=k_rope.contiguous(),
                sparse_block_size=1,
                layout_query="BSND",
                layout_kv="BSND",
                sparse_mode=0,
                attention_mode=2,
                return_softmax_lse=False,
            )
            result = ret[0] if isinstance(ret, (tuple, list)) else ret
            result = result[:, :, : layer.tp_q_head_num].reshape(batch, -1)
        result = result.masked_fill(~valid.any(dim=1, keepdim=True), 0)
        output[:, column].copy_(result)
    return torch.nn.functional.pad(
        output.view(logical_tokens, -1), (0, 0, 0, num_tokens - logical_tokens)
    )


def forward_sparsity_driven_kv_offload(
    backend: AscendAttnBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    save_kv_cache: bool = True,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
):
    """Run sparse attention using host-offloaded compact MLA KV."""
    del v
    if q_rope is None or k_rope is None or topk_indices is None:
        raise ValueError(
            "Sparsity-driven KV offload requires q_rope, k_rope, and topk_indices."
        )

    is_prefill = forward_batch.forward_mode.is_extend_without_speculative()

    q_nope, q_pe = q, q_rope
    k_nope = k.view(-1, layer.tp_k_head_num, backend.kv_lora_rank).contiguous()
    k_pe = k_rope.view(-1, layer.tp_k_head_num, backend.qk_rope_head_dim).contiguous()
    sparse_kv_manager = _get_sparse_kv_manager(backend)
    stream = torch.npu.current_stream(backend.device)

    if save_kv_cache:
        sparse_kv_manager.offload_v2(k_nope, k_pe, layer, forward_batch, stream)

    if is_sparse_mtp_batch(forward_batch):
        return _forward_sparse_mtp(
            backend, q_nope, q_pe, layer, forward_batch, topk_indices
        )

    if is_prefill:
        if backend.forward_metadata.actual_seq_lengths_q is not None:
            actual_seq_qlen = backend.forward_metadata.actual_seq_lengths_q
        else:
            actual_seq_qlen = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
    elif backend.forward_metadata.actual_seq_lengths_q is None:
        if (
            forward_batch.forward_mode.is_draft_extend_v2()
            or forward_batch.forward_mode.is_target_verify()
        ):
            actual_seq_qlen = torch.arange(
                backend.speculative_num_draft_tokens,
                backend.speculative_num_draft_tokens + q.shape[0],
                backend.speculative_num_draft_tokens,
                dtype=torch.int32,
                device=q_nope.device,
            )
        else:
            actual_seq_qlen = torch.arange(
                1, q.shape[0] + 1, device=q_nope.device, dtype=torch.int32
            )
    else:
        actual_seq_qlen = backend.forward_metadata.actual_seq_lengths_q

    if backend.forward_metadata.actual_seq_lengths_kv is not None:
        actual_seq_lengths_kv = backend.forward_metadata.actual_seq_lengths_kv
    elif backend.forward_metadata.seq_lens_cpu_int is not None:
        actual_seq_lengths_kv = backend.forward_metadata.seq_lens_cpu_int
    else:
        actual_seq_lengths_kv = backend.forward_metadata.seq_lens

    if (
        is_prefill
        and is_dsa_enable_prefill_cp()
        and forward_batch.attn_cp_metadata is not None
    ):
        attn_out = backend.do_cp_balance_attn(
            q_nope,
            k_nope,
            q_pe,
            k_pe,
            topk_indices,
            layer,
            actual_seq_qlen,
            actual_seq_lengths_kv,
        )
    elif forward_batch.forward_mode.is_decode() and _copy_sfa_decode_eligible(
        backend, forward_batch, layer, q_nope, topk_indices
    ):
        sparse_kv_manager.last_decode_path = "copy_sfa"
        forward_batch.sparse_kv_decode_path = "copy_sfa"
        batch_size = forward_batch.batch_size
        padded_query_heads = q_nope.numel() // (batch_size * backend.kv_lora_rank)
        attn_out = sparse_kv_manager.copy_sfa_decode(
            layer,
            forward_batch,
            q_nope.view(batch_size, 1, padded_query_heads, backend.kv_lora_rank),
            q_pe.view(batch_size, 1, padded_query_heads, backend.qk_rope_head_dim),
            layer.tp_q_head_num,
            layer.scaling,
            topk_indices,
            lim_plan=(
                backend.forward_metadata.sparse_kv_lim_plan
                if sparse_kv_manager.lim_adapter is not None
                and sparse_kv_manager.lim_adapter.is_native
                else None
            ),
        )
    elif forward_batch.forward_mode.is_decode():
        sparse_kv_manager.last_decode_path = "reference"
        forward_batch.sparse_kv_decode_path = "reference"
        batch_size = forward_batch.batch_size
        num_kv_heads = layer.tp_k_head_num
        num_query_heads = layer.tp_q_head_num
        nope_head_dim = backend.kv_lora_rank
        rope_head_dim = backend.qk_rope_head_dim

        topk_2d_input = normalize_batch_topk_indices(topk_indices)
        effective_topk_length = topk_2d_input.shape[1]
        selected_kv_length = sparse_kv_manager.sparse_context_len
        if effective_topk_length <= 0:
            raise RuntimeError("SFA BSND compact path expects a positive top-k length.")
        if effective_topk_length > selected_kv_length:
            raise RuntimeError(
                "DSA top-k length exceeds sparse KV device cache capacity: "
                f"topk_len={effective_topk_length}, "
                f"sparse_context_len={selected_kv_length}."
            )
        if effective_topk_length == selected_kv_length:
            topk_2d = topk_2d_input.contiguous()
        else:
            topk_2d = torch.full(
                (batch_size, selected_kv_length),
                -1,
                dtype=topk_2d_input.dtype,
                device=topk_2d_input.device,
            )
            topk_2d[:, :effective_topk_length] = topk_2d_input
            topk_2d = topk_2d.contiguous()

        assert num_kv_heads == 1, (
            "FIA_v2 MLA selected KV path expects KV_N == 1, "
            f"got num_kv_heads={num_kv_heads}"
        )

        padded_query_heads = q_nope.numel() // (batch_size * nope_head_dim)
        assert padded_query_heads >= num_query_heads, (
            "query head count mismatch: "
            f"padded_query_heads={padded_query_heads}, "
            f"num_query_heads={num_query_heads}"
        )

        # Materialize the compact top-k KV for the current attention step:
        # device-cache hits and host misses are copied into this buffer, the
        # device cache is refilled from it, and sparse attention consumes it
        # directly.
        selected_kv_buffer = sparse_kv_manager.selected_kv_buffer[:batch_size]
        selected_kv_buffer.zero_()
        adapter = sparse_kv_manager.lim_adapter
        if adapter is not None and adapter.full_visible:
            plan = backend.forward_metadata.sparse_kv_lim_plan
            if plan is None:
                raise RuntimeError(
                    "Full-visible LIM attention requires an indexer plan."
                )
            adapter.gather_full_visible(
                sparse_kv_manager,
                plan,
                layer.layer_id - sparse_kv_manager.start_layer,
                selected_kv_buffer,
                copy=unidex_copy_inplace,
            )
        elif adapter is not None and adapter.is_native:
            plan = backend.forward_metadata.sparse_kv_lim_plan
            if plan is None:
                raise RuntimeError(
                    "LIM attention has no anchor plan; an indexer must precede IndexShare layers."
                )
            adapter.materialize(
                sparse_kv_manager,
                plan,
                layer.layer_id - sparse_kv_manager.start_layer,
                selected_kv_buffer,
                stream,
                copy=unidex_copy_inplace,
            )
        else:
            sparse_kv_manager.materialize_selected_kv(
                layer, forward_batch, topk_2d, selected_kv_buffer, stream
            )

        _wait_stream_event(stream, sparse_kv_manager.miss_done)
        if adapter is not None and not adapter.full_visible:
            plan = backend.forward_metadata.sparse_kv_lim_plan
            if plan is None:
                raise RuntimeError(
                    "Experimental prefix/tail attention requires an indexer plan."
                )
            selected_kv_buffer, topk_2d = adapter.append_tail(
                sparse_kv_manager,
                plan,
                layer.layer_id - sparse_kv_manager.start_layer,
                selected_kv_buffer,
                copy=unidex_copy_inplace,
            )
            selected_kv_length = adapter.width

        selected_k_nope, selected_k_rope = selected_kv_buffer.split(
            [nope_head_dim, rope_head_dim], dim=-1
        )

        topk_valid = (topk_2d >= 0) & (topk_2d < sparse_kv_manager.max_context_len)
        if forward_batch.seq_lens is not None:
            valid_rows = (
                (forward_batch.seq_lens[:batch_size] > 0)
                & (forward_batch.req_pool_indices[:batch_size] > 0)
                & (forward_batch.req_pool_indices[:batch_size] < sparse_kv_manager.size)
            ).view(batch_size, 1)
            topk_valid = topk_valid & valid_rows

        actual_seq_lengths_kv = (
            topk_valid.sum(dim=1)
            .clamp(min=1, max=selected_kv_length)
            .to(device=q_nope.device, dtype=torch.int32)
            .contiguous()
        )
        actual_seq_lengths_query = torch.ones(
            batch_size, dtype=torch.int32, device=q_nope.device
        ).contiguous()

        compact_indices = (
            torch.arange(selected_kv_length, device=q_nope.device, dtype=torch.int32)
            .view(1, 1, 1, selected_kv_length)
            .expand(batch_size, 1, num_kv_heads, selected_kv_length)
            .clone()
        )
        compact_valid = topk_valid.view(batch_size, 1, 1, selected_kv_length).expand(
            batch_size, 1, num_kv_heads, selected_kv_length
        )
        sparse_indices = torch.where(
            compact_valid,
            compact_indices,
            torch.full_like(compact_indices, -1),
        ).contiguous()

        empty_rows = (topk_valid.sum(dim=1) == 0).view(batch_size, 1, 1)
        sparse_indices[:, :, :, 0] = torch.where(
            empty_rows.expand(batch_size, 1, num_kv_heads),
            torch.zeros(
                (batch_size, 1, num_kv_heads),
                dtype=torch.int32,
                device=q_nope.device,
            ),
            sparse_indices[:, :, :, 0],
        )

        q_nope_sfa = q_nope.view(
            batch_size, 1, padded_query_heads, nope_head_dim
        ).contiguous()
        q_rope_sfa = q_pe.view(
            batch_size, 1, padded_query_heads, rope_head_dim
        ).contiguous()
        k_nope_sfa = selected_k_nope.contiguous()
        k_rope_sfa = selected_k_rope.contiguous()

        assert q_nope_sfa.shape == (
            batch_size,
            1,
            padded_query_heads,
            nope_head_dim,
        )
        assert q_rope_sfa.shape == (
            batch_size,
            1,
            padded_query_heads,
            rope_head_dim,
        )
        assert k_nope_sfa.shape == (
            batch_size,
            selected_kv_length,
            num_kv_heads,
            nope_head_dim,
        )
        assert k_rope_sfa.shape == (
            batch_size,
            selected_kv_length,
            num_kv_heads,
            rope_head_dim,
        )

        ret = torch_npu.npu_sparse_flash_attention(
            q_nope_sfa,
            k_nope_sfa,
            k_nope_sfa,
            sparse_indices,
            layer.scaling,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            query_rope=q_rope_sfa,
            key_rope=k_rope_sfa,
            sparse_block_size=1,
            layout_query="BSND",
            layout_kv="BSND",
            sparse_mode=0,
            attention_mode=2,
            return_softmax_lse=False,
        )

        _wait_stream_event(stream, sparse_kv_manager.refill_done)
        if sparse_kv_manager.copy_sfa_enabled and adapter is None:
            # The materialization fallback rewrites payload residency without
            # maintaining the native planner's inverse map. Reenter Copy-SFA
            # with a cold cache instead of publishing stale hits.
            layer_index = layer.layer_id - sparse_kv_manager.start_layer
            rows = forward_batch.req_pool_indices.long()
            sparse_kv_manager.device_slot_map[layer_index].index_fill_(0, rows, -1)
            sparse_kv_manager.device_slot_owner[layer_index].index_fill_(0, rows, -1)

        attn_out = ret[0] if isinstance(ret, tuple) else ret
        attn_out = attn_out[:, :, :num_query_heads, :].reshape(
            batch_size, num_query_heads * nope_head_dim
        )
        # Empty/padded rows use a dummy KV entry to satisfy the SFA contract.
        # The vendor kernel may leave their output unwritten; never publish it.
        attn_out.masked_fill_(empty_rows.view(batch_size, 1), 0)
    else:
        if is_prefill:
            k_nope_sfa, k_pe_sfa = sparse_kv_manager.get_forward_kv(
                layer, forward_batch, stream
            )
            forward_actual_seq_lengths_kv = torch.cumsum(forward_batch.seq_lens, dim=0)
        else:
            k_nope_sfa, k_pe_sfa = k_nope, k_pe
            forward_actual_seq_lengths_kv = actual_seq_lengths_kv

        topk_indices = _expand_dsa_sparse_indices(topk_indices)
        attn_out, _, _ = torch_npu.npu_sparse_flash_attention(
            query=q_nope,
            key=k_nope_sfa,
            value=k_nope_sfa,
            query_rope=q_pe,
            key_rope=k_pe_sfa,
            sparse_indices=topk_indices,
            scale_value=layer.scaling,
            actual_seq_lengths_query=actual_seq_qlen.to(
                device=q_nope.device, dtype=torch.int32
            ),
            actual_seq_lengths_kv=forward_actual_seq_lengths_kv.to(
                device=q_nope.device, dtype=torch.int32
            ),
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="TND",
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=False,
        )

    return attn_out
