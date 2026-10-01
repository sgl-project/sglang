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
from sglang.srt.layers.attention.graph_variants import DLLM_FULL_WINDOW
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.model_executor.runner_utils.capture_mode import (
    get_capture_attention_variant,
)


class FlashAttentionDenseBackend(TritonAttnBackend):
    """Dense D256 window and D512 full attention over a paged cache."""

    requires_contiguous_current_kv = False

    def __init__(self, model_runner):
        super().__init__(model_runner)
        self.qo_indptr = self.qo_indptr.to(torch.int32)
        self._native_extend = self.extend_attention_fwd
        self.extend_attention_fwd = self._dense_extend
        self._dense_workspaces = {}

    def _dense_extend(
        self,
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
            or mask is not None
            or options.get("sinks") is not None
            or options.get("score_mod") is not None
            or options.get("logit_cap", 0.0)
            or options.get("skip_prefix", False)
            or options.get("skip_extend", False)
        ):
            return self._native_extend(
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
        prefix_capacity = (
            self.sliding_window_size if dim == 256 else self.max_context_len
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
        if dim == 256 and window > 0:
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
        if get_capture_attention_variant() == DLLM_FULL_WINDOW and not causal:
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
