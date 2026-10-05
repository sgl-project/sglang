# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""dLLM FA4 compute and graph workspaces over Triton paged-cache metadata."""

from dataclasses import dataclass
from functools import cache
from typing import ClassVar

import torch

from sglang.kernels.ops.attention.dllm_kv_pack import pack_prefix_current
from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd_unified
from sglang.kernels.ops.attention.flash_attention_v4 import flash_attn_gqa_512
from sglang.kernels.ops.attention.flash_attn.cute.interface import (
    flash_attn_varlen_func,
)
from sglang.srt.layers.attention.triton_backend import (
    ForwardMetadata,
    update_sliding_window_buffer,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner_utils.capture_mode import (
    get_capture_attention_variant,
)
from sglang.srt.utils import get_device_capability


class _DenseKVWorkspace:
    def __init__(self, capacity, heads, dim, batch_size, device, dtype):
        # TMA loads can include the masked tail beyond the packed sequences.
        self.key = torch.zeros((capacity, heads, dim), device=device, dtype=dtype)
        self.value = torch.zeros_like(self.key)
        self.cu_seqlens = torch.empty(batch_size + 1, device=device, dtype=torch.int32)
        self.indices = torch.arange(capacity, device=device, dtype=torch.int64)
        self.window_start = torch.zeros(batch_size, device=device, dtype=torch.int32)

    def fits(self, capacity, batch_size, heads, dim):
        return (
            capacity * heads * dim <= self.key.numel()
            and capacity <= self.indices.numel()
            and batch_size < self.cu_seqlens.numel()
        )

    def view(self, capacity, batch_size, heads, dim):
        shape = (capacity, heads, dim)
        elements = capacity * heads * dim
        return (
            self.key.view(-1)[:elements].view(shape),
            self.value.view(-1)[:elements].view(shape),
            self.cu_seqlens[: batch_size + 1],
            self.indices[:capacity],
            self.window_start[:batch_size],
        )


DLLM_VARLEN = "dllm_varlen"
DLLM_FULL_WINDOW = "dllm_full_window"


@dataclass(frozen=True)
class DllmWindowGraphVariants:
    window_size: int
    block_size: int
    capture_labels: ClassVar[tuple[str, ...]] = (DLLM_VARLEN, DLLM_FULL_WINDOW)

    def select(self, forward_batch: ForwardBatch, capture_batch_size: int) -> str:
        lengths = forward_batch.seq_lens_cpu
        if (
            forward_batch.batch_size > 0
            and capture_batch_size == forward_batch.batch_size
            and forward_batch.forward_mode.is_dllm_extend()
            and forward_batch.input_ids.numel()
            == forward_batch.batch_size * self.block_size
            and lengths is not None
            and int(lengths[: forward_batch.batch_size].min()) - self.block_size
            >= self.window_size
        ):
            return DLLM_FULL_WINDOW
        return DLLM_VARLEN


class DllmFlashAttention:
    """FA4 computation and context-encoding graphs for DiffusionGemma."""

    def __init__(self, backend):
        assert get_device_capability()[0] == 10, (
            "dLLM FA4 D256/D512 requires an SM100-family GPU."
        )
        self.backend = backend
        self.graph_mode = False
        self._graph_workspaces = {}
        self._eager_workspaces = {}
        self._prefill_graph_metadata = {}
        self.decode_graph_metadata = {}
        self._graph_slots = self._graph_tokens = 0
        self._prefill_capture_sizes = set()
        self._prefill_capture_max_requests = 0

    def init_graph_state(self, max_requests, max_tokens):
        self._graph_slots = max(self._graph_slots, max_requests)
        self._graph_tokens = max(self._graph_tokens, max_tokens)

    def init_prefill_graph_state(self, max_requests, capture_tokens):
        self.init_graph_state(max_requests, max(capture_tokens))
        self._prefill_capture_sizes = set(capture_tokens)
        self._prefill_capture_max_requests = max_requests

    def graph_variants(self, block_size):
        window = self.backend.sliding_window_size
        if window is not None and window > 0:
            return DllmWindowGraphVariants(window, block_size)
        return None

    def _workspace(self, key, capacity, batch_size, heads, dim, query):
        if self.graph_mode:
            capacity = self._graph_slots * key[0] + self._graph_tokens
            batch_size = self._graph_slots
        captured = self._graph_workspaces.setdefault(key, [])
        candidates = (
            captured
            if self.graph_mode
            else (w for group in self._graph_workspaces.values() for w in group)
        )
        workspace = next(
            (
                w
                for w in candidates
                if w.key.dtype == query.dtype
                and w.fits(capacity, batch_size, heads, dim)
            ),
            None,
        )
        if workspace is None:
            workspace = self._eager_workspaces.get(key)
            if workspace is None or not workspace.fits(
                capacity, batch_size, heads, dim
            ):
                workspace = _DenseKVWorkspace(
                    capacity, heads, dim, batch_size, query.device, query.dtype
                )
            if self.graph_mode:
                # Keep earlier captures' allocations alive when a later runner needs more capacity.
                captured.append(workspace)
                self._eager_workspaces.pop(key, None)
            else:
                self._eager_workspaces[key] = workspace
        return workspace

    def init_forward_metadata_out_graph(self, batch, in_capture):
        self._eager_workspaces.clear()
        self.graph_mode = True
        if batch.forward_mode == ForwardMode.EXTEND:
            return self.init_prefill_metadata(batch, in_capture)
        bs = batch.batch_size
        if not in_capture:
            self.backend.forward_metadata = self.decode_graph_metadata[bs]
        self.backend._init_forward_metadata_out_graph(batch, in_capture)
        if in_capture:
            self.decode_graph_metadata[bs] = self.backend.forward_metadata

    def init_prefill_metadata(self, batch, in_capture):
        backend = self.backend
        bs, tokens = batch.batch_size, batch.input_ids.numel()
        key = (bs, tokens)
        if in_capture:
            self.init_graph_state(bs, tokens)
            capacity = batch.max_seq_len_override or backend.max_context_len
            swa = (
                backend.sliding_window_size is not None
                and backend.sliding_window_size > 0
            )
            self._prefill_graph_metadata[key] = ForwardMetadata(
                attn_logits=None,
                attn_lse=None,
                num_kv_splits=None,
                max_extend_len=max(batch.extend_seq_lens_cpu),
                kv_indptr=torch.zeros(bs + 1, dtype=torch.int32, device=backend.device),
                kv_indices=torch.empty(
                    bs * capacity, dtype=torch.int64, device=backend.device
                ),
                qo_indptr=torch.zeros(bs + 1, dtype=torch.int32, device=backend.device),
                custom_mask=None,
                mask_indptr=None,
                window_kv_indptr=torch.zeros(
                    bs + 1, dtype=torch.int32, device=backend.device
                )
                if swa
                else None,
                window_kv_indices=torch.empty(
                    bs * backend.sliding_window_size,
                    dtype=torch.int64,
                    device=backend.device,
                )
                if swa
                else None,
                window_num_kv_splits=None,
                window_kv_offsets=torch.empty(
                    bs, dtype=torch.int64, device=backend.device
                )
                if swa
                else None,
                swa_out_cache_loc=torch.empty_like(batch.out_cache_loc)
                if backend.use_sliding_window_kv_pool
                else None,
                out_cache_loc_full_physical=torch.empty_like(batch.out_cache_loc)
                if backend.kv_index_translator.is_translating
                else None,
            )
        metadata = self._prefill_graph_metadata[key]
        backend._fill_kv_indptr_and_indices(
            bs,
            batch.extend_prefix_lens,
            batch.req_pool_indices,
            metadata.kv_indices,
            kv_indptr=metadata.kv_indptr,
        )
        metadata.qo_indptr[0].zero_()
        torch.cumsum(batch.extend_seq_lens, dim=0, out=metadata.qo_indptr[1:])
        if metadata.window_kv_indices is not None:
            _, _, _, offsets = update_sliding_window_buffer(
                metadata.window_kv_indptr,
                backend.kv_index_translator,
                batch.req_pool_indices,
                backend.sliding_window_size,
                batch.extend_prefix_lens,
                bs,
                backend.device,
                backend.token_to_kv_pool,
                window_kv_indices=metadata.window_kv_indices,
            )
            metadata.window_kv_offsets.copy_(offsets)
        if metadata.swa_out_cache_loc is not None:
            metadata.swa_out_cache_loc.copy_(
                backend.kv_index_translator.sliding_window_write_loc_for(
                    batch.out_cache_loc
                )
            )
        if metadata.out_cache_loc_full_physical is not None:
            backend.kv_index_translator.fill_capture_write_loc(
                out=metadata.out_cache_loc_full_physical,
                forward_batch=batch,
                width=metadata.out_cache_loc_full_physical.numel(),
            )
        backend.forward_metadata = metadata

    def can_run_prefill_cuda_graph(self, batch):
        query_limit = self.get_prefill_cuda_graph_max_query_len(
            batch.input_ids.numel(), self._prefill_capture_max_requests
        )
        return (
            batch.forward_mode == ForwardMode.EXTEND
            and batch.input_ids.numel() in self._prefill_capture_sizes
            and self.backend.dcp_size == 1
            and not batch.contains_image_inputs()
            and (
                query_limit is None
                or max(batch.extend_seq_lens_cpu, default=0) <= query_limit
            )
        )

    def get_prefill_cuda_graph_max_query_len(self, num_tokens, max_requests):
        if num_tokens <= max_requests * self.backend.dllm_block_size:
            return self.backend.dllm_block_size
        return None

    @cache
    def _supports_layer(self, layer, dtype):
        pool = self.backend.token_to_kv_pool
        buffers = (
            pool.get_key_buffer(layer.layer_id),
            pool.get_value_buffer(layer.layer_id),
        )
        return (
            layer.qk_head_dim == layer.v_head_dim
            and layer.qk_head_dim in (256, 512)
            and dtype in (torch.float16, torch.bfloat16)
            and all(
                buffer.ndim == 3 and buffer.dtype == dtype and buffer.stride(-1) == 1
                for buffer in buffers
            )
        )

    def forward_extend(
        self,
        layer,
        q,
        k,
        v,
        out,
        kb,
        vb,
        qo,
        ki,
        ids,
        mask,
        causal,
        mask_ptr,
        max_q,
        k_scale,
        v_scale,
        sm_scale=None,
        **options,
    ):
        if (
            not self._supports_layer(layer, q.dtype)
            or mask is not None
            or options.get("sinks") is not None
            or options.get("score_mod") is not None
            or options.get("logit_cap", 0.0)
        ):
            return self.backend.extend_attention_fwd(
                q,
                k.contiguous(),
                v.contiguous(),
                out,
                kb,
                vb,
                qo,
                ki,
                ids,
                mask,
                causal,
                mask_ptr,
                max_q,
                k_scale,
                v_scale,
                sm_scale=sm_scale,
                **options,
            )
        bs, heads, dim = qo.numel() - 1, k.shape[1], k.shape[2]
        is_window_layer = (
            layer.sliding_window_size is not None and layer.sliding_window_size >= 0
        )
        prefix_capacity = (
            self.backend.sliding_window_size
            if is_window_layer
            else self.backend.max_context_len
        )
        prefix_tokens = bs * prefix_capacity if self.graph_mode else ids.numel()
        capacity = prefix_tokens + q.shape[0]
        key = (prefix_capacity, heads, dim, q.dtype)
        workspace = self._workspace(key, capacity, bs, heads, dim, q)
        dense_k, dense_v, cu_seqlens, dense_ids, window_start = workspace.view(
            capacity, bs, heads, dim
        )
        pack_prefix_current(
            k,
            v,
            kb,
            vb,
            qo,
            ki,
            ids,
            dense_k,
            dense_v,
            cu_seqlens,
            prefix_capacity + max_q,
        )
        window = options.get("sliding_window_size", -1)
        if window > 0:
            prefix_lens = ki[1:] - ki[:-1]
            return extend_attention_fwd_unified(
                q,
                out,
                dense_k,
                dense_v,
                k_scale,
                v_scale,
                qo,
                cu_seqlens,
                dense_ids,
                prefix_lens,
                max_q,
                sm_scale=sm_scale,
                is_causal=causal,
                sliding_window_size=window,
                window_start_pos=window_start,
                page_size=1,
            )
        if dim == 512:
            return flash_attn_gqa_512(
                q,
                dense_k,
                dense_v,
                out,
                cu_seqlens_q=qo,
                cu_seqlens_k=cu_seqlens,
                softmax_scale=sm_scale,
                causal=causal,
            )
        if (
            is_window_layer
            and get_capture_attention_variant() == DLLM_FULL_WINDOW
            and not causal
        ):
            return flash_attn_varlen_func(
                q.view(bs, -1, *q.shape[1:]),
                dense_k.view(bs, -1, heads, dim),
                dense_v.view(bs, -1, heads, dim),
                out=out.view(bs, -1, *out.shape[1:]),
                softmax_scale=sm_scale,
                causal=False,
            )
        return flash_attn_varlen_func(
            q,
            dense_k,
            dense_v,
            out=out,
            cu_seqlens_q=qo,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_q,
            max_seqlen_k=prefix_capacity + max_q,
            softmax_scale=sm_scale,
            causal=causal,
        )
