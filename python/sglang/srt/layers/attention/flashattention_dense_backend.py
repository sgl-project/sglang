# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""FA4 dense attention using Triton's paged-cache metadata and KV writes."""

import torch

from sglang.kernels.ops.attention.dense_kv import DenseKVWorkspace, pack_prefix_current
from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd_unified
from sglang.kernels.ops.attention.flash_attention_v4 import flash_attn_gqa_512
from sglang.kernels.ops.attention.flash_attn.cute.interface import (
    flash_attn_varlen_func,
)
from sglang.srt.layers.attention.graph_variants import (
    DLLM_FULL_WINDOW,
    DllmWindowGraphVariants,
)
from sglang.srt.layers.attention.triton_backend import (
    ForwardMetadata,
    TritonAttnBackend,
    update_sliding_window_buffer,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_utils.capture_mode import (
    get_capture_attention_variant,
)
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import get_device_capability


class FlashAttentionDenseBackend(TritonAttnBackend):
    """FA4 D256/D512 attention over Triton's paged-prefix metadata."""

    requires_contiguous_current_kv = False
    full_cuda_graph_uses_chunked_prefix = False

    @staticmethod
    def supports_model(model_config):
        """Models validated with this adapter's paged-prefix metadata."""
        return "DiffusionGemmaForBlockDiffusion" in model_config.hf_config.architectures

    def __init__(self, model_runner):
        assert get_device_capability()[0] == 10, (
            "Dense FA4 D256/D512 requires an SM100-family GPU."
        )
        super().__init__(model_runner)
        self.qo_indptr = self.qo_indptr.to(torch.int32)
        self._dense_workspaces = {}
        self._prefill_graph_metadata = {}
        self._prefill_capture_sizes = set(
            get_exec().graph.cuda_graph_config.prefill.bs or ()
        )
        self.supports_prefill_cuda_graph_max_context_size = check_cuda_graph_backend(
            Phase.PREFILL, Backend.FULL
        )

    def get_cuda_graph_variants(self, model_runner, forward_mode, captured_req_width):
        if (
            forward_mode.is_dllm_extend()
            and self.sliding_window_size is not None
            and self.sliding_window_size > 0
        ):
            return DllmWindowGraphVariants(self.sliding_window_size, captured_req_width)
        return super().get_cuda_graph_variants(
            model_runner, forward_mode, captured_req_width
        )

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            self._init_full_prefill_metadata(forward_batch, in_capture)
        else:
            super().init_forward_metadata_out_graph(forward_batch, in_capture)

    def _init_full_prefill_metadata(self, batch, in_capture):
        bs, tokens = batch.batch_size, batch.input_ids.numel()
        key = (bs, tokens)
        if in_capture:
            capacity = batch.max_seq_len_override or self.max_context_len
            swa = self.sliding_window_size is not None and self.sliding_window_size > 0
            self._prefill_graph_metadata[key] = ForwardMetadata(
                attn_logits=None,
                attn_lse=None,
                num_kv_splits=None,
                max_extend_len=tokens,
                kv_indptr=torch.zeros(bs + 1, dtype=torch.int32, device=self.device),
                kv_indices=torch.empty(
                    bs * capacity, dtype=torch.int64, device=self.device
                ),
                qo_indptr=torch.zeros(bs + 1, dtype=torch.int32, device=self.device),
                custom_mask=None,
                mask_indptr=None,
                window_kv_indptr=torch.zeros(
                    bs + 1, dtype=torch.int32, device=self.device
                )
                if swa
                else None,
                window_kv_indices=torch.empty(
                    bs * self.sliding_window_size, dtype=torch.int64, device=self.device
                )
                if swa
                else None,
                window_num_kv_splits=None,
                window_kv_offsets=torch.empty(bs, dtype=torch.int64, device=self.device)
                if swa
                else None,
                swa_out_cache_loc=torch.empty_like(batch.out_cache_loc)
                if self.use_sliding_window_kv_pool
                else None,
                out_cache_loc_full_physical=torch.empty_like(batch.out_cache_loc)
                if self.kv_index_translator.is_translating
                else None,
            )
        metadata = self._prefill_graph_metadata[key]
        self._fill_kv_indptr_and_indices(
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
                self.kv_index_translator,
                batch.req_pool_indices,
                self.sliding_window_size,
                batch.extend_prefix_lens,
                bs,
                self.device,
                self.token_to_kv_pool,
                window_kv_indices=metadata.window_kv_indices,
            )
            metadata.window_kv_offsets.copy_(offsets)
        if metadata.swa_out_cache_loc is not None:
            metadata.swa_out_cache_loc.copy_(
                self.kv_index_translator.sliding_window_write_loc_for(
                    batch.out_cache_loc
                )
            )
        if metadata.out_cache_loc_full_physical is not None:
            self.kv_index_translator.fill_capture_write_loc(
                out=metadata.out_cache_loc_full_physical,
                forward_batch=batch,
                width=metadata.out_cache_loc_full_physical.numel(),
            )
        self.forward_metadata = metadata

    def can_run_prefill_cuda_graph(self, batch):
        if not check_cuda_graph_backend(Phase.PREFILL, Backend.FULL):
            return True
        return (
            batch.forward_mode == ForwardMode.EXTEND
            and batch.input_ids.numel() in self._prefill_capture_sizes
            and self.dcp_size == 1
            and not batch.contains_image_inputs()
        )

    def _forward_extend_kernel(
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
            q.shape[-1] not in (256, 512)
            or q.dtype not in (torch.float16, torch.bfloat16)
            or k.shape[-1] != q.shape[-1]
            or v.shape[-1] != q.shape[-1]
            or kb.ndim != 3
            or vb.ndim != 3
            or kb.dtype != q.dtype
            or vb.dtype != q.dtype
            or kb.stride(-1) != 1
            or vb.stride(-1) != 1
            or mask is not None
            or options.get("sinks") is not None
            or options.get("score_mod") is not None
            or options.get("logit_cap", 0.0)
            or options.get("skip_prefix", False)
            or options.get("skip_extend", False)
        ):
            return super()._forward_extend_kernel(
                layer,
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
            self.sliding_window_size if is_window_layer else self.max_context_len
        )
        capacity = bs * prefix_capacity + q.shape[0]
        key = (capacity, bs, heads, dim, q.dtype)
        if key not in self._dense_workspaces:
            self._dense_workspaces[key] = DenseKVWorkspace(
                capacity, heads, dim, bs, q.device, q.dtype
            )
        workspace = self._dense_workspaces[key]
        pack_prefix_current(
            k,
            v,
            kb,
            vb,
            qo,
            ki,
            ids,
            workspace.key,
            workspace.value,
            workspace.cu_seqlens,
            prefix_capacity + max_q,
        )
        window = options.get("sliding_window_size", -1)
        if window > 0:
            prefix_lens = ki[1:] - ki[:-1]
            return extend_attention_fwd_unified(
                q,
                out,
                workspace.key,
                workspace.value,
                k_scale,
                v_scale,
                qo,
                workspace.cu_seqlens,
                workspace.indices,
                prefix_lens,
                max_q,
                sm_scale=sm_scale,
                is_causal=causal,
                sliding_window_size=window,
                window_start_pos=workspace.window_start,
                page_size=1,
            )
        if dim == 512:
            return flash_attn_gqa_512(
                q,
                workspace.key,
                workspace.value,
                out,
                cu_seqlens_q=qo,
                cu_seqlens_k=workspace.cu_seqlens,
                softmax_scale=sm_scale,
                causal=causal,
            )
        if (
            is_window_layer
            and get_capture_attention_variant() == DLLM_FULL_WINDOW
            and not causal
        ):
            return flash_attn_varlen_func(
                q.unsqueeze(0),
                workspace.key.unsqueeze(0),
                workspace.value.unsqueeze(0),
                out=out.unsqueeze(0),
                softmax_scale=sm_scale,
                causal=False,
            )
        return flash_attn_varlen_func(
            q,
            workspace.key,
            workspace.value,
            out=out,
            cu_seqlens_q=qo,
            cu_seqlens_k=workspace.cu_seqlens,
            max_seqlen_q=max_q,
            max_seqlen_k=prefix_capacity + max_q,
            softmax_scale=sm_scale,
            causal=causal,
        )
