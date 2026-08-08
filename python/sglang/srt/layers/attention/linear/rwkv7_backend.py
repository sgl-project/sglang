# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""RWKV-7 (Goose) linear-attention backend for sglang.

Unlike GDN, the RWKV-7 model module does ALL projections / LoRAs / gating in plain
torch and hands the backend already-projected per-head tensors. This backend owns
the two pieces of recurrent state in the MambaPool:

  * conv[0] / conv[1] : the two width-2 (prev-token) token-shift states
                        (attn / ffn), shape (size+1, hidden, 1), fp32.
  * temporal          : the WKV recurrent state S, shape (size+1, H, K, V), fp32.

The model calls `token_shift(...)` (before projections) and `recurrence(...)`
(after projections) directly on this backend instance, which it obtains via
`forward_batch.attn_backend.linear_attn_backend`. We therefore bypass the
RadixLinearAttention / HybridLinearAttnBackend.forward dispatch (whose fixed
mixed_qkv/a/b signature does not fit RWKV-7's r/w/k/v/kk/a). `init_forward_metadata`
is still driven through HybridLinearAttnBackend, so `self.forward_metadata`
(query_start_loc + mamba_cache_indices) is populated normally.
"""

from typing import Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)

# The WKV recurrence is a self-contained triton kernel for BOTH the decode
# (T==1) AND the extend/prefill (packed varlen via cu_seqlens) path, with no
# dependency on the flash-linear-attention package.
from sglang.srt.layers.attention.rwkv7_kernels import wkv_recurrent
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class Rwkv7NoOpFullAttnBackend(AttentionBackend):
    """A trivial full-attention backend for the all-linear RWKV-7 case.

    RWKV-7 has ZERO full-attention layers, so HybridLinearAttnBackend never routes
    any layer to the full backend. But it still calls `init_forward_metadata` (and a
    few cuda-graph hooks) on the full backend each step. Real backends
    (triton/flashinfer) either probe the empty full KV pool at construction or reject
    fp32 planning, so we substitute this no-op instead.
    """

    # HybridLinearAttnBackend (sglang main) copies these off the full backend at
    # construction; there is no full-attn KV pool here, and the req/token pools
    # come from the runner when one is given.
    token_to_kv_pool = None
    req_to_token_pool = None
    needs_cpu_seq_lens = False

    def __init__(self, model_runner: Optional[ModelRunner] = None):
        if model_runner is not None:
            self.req_to_token_pool = model_runner.req_to_token_pool
            self.token_to_kv_pool = getattr(model_runner, "token_to_kv_pool", None)
            self.max_context_len = model_runner.model_config.context_len

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        pass

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        pass

    def init_cpu_graph_state(self, *args, **kwargs):
        pass

    def init_forward_metadata_capture_cpu_graph(self, *args, **kwargs):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def get_cpu_graph_seq_len_fill_value(self):
        return 1

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError("RWKV-7 has no full-attention layers.")

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError("RWKV-7 has no full-attention layers.")


class Rwkv7AttnBackend(MambaAttnBackendBase):
    """Linear-attention backend for RWKV-7.

    Both the decode and the extend/prefill paths run through the same FLA-free
    `wkv_recurrent` triton kernel (exact, ~1e-6 vs the numpy oracle). scale=1.0 to
    match the numpy oracle, which applies no scaling to r before GroupNorm.
    """

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        # Base-class field (tuple[int, int]); RWKV-7's token-shift window is
        # (hidden, conv_kernel-1=1). The base reads [-1] as the conv window length.
        self.conv_states_shape = tuple(
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape[-2:]
        )
        self.scale = 1.0

    # ---- token-shift (width-2 causal shift via the conv state) ----
    def token_shift(
        self,
        x: torch.Tensor,
        layer_id: int,
        conv_idx: int,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Return the previous-token hidden state for each position in x.

        x: [num_tokens, hidden] (post attn_norm / ffn_norm). Updates the stored
        conv state with the last token of each sequence for the next step.
        """
        cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        conv = cache.conv[conv_idx]  # [size+1, hidden, 1]
        md = self.forward_metadata
        cache_indices = md.mamba_cache_indices

        if forward_batch.forward_mode.is_decode_or_idle():
            # one token per request: shifted = stored prev; store current.
            # Padded cuda-graph replay fills the tail of mamba_cache_indices with
            # PAD_SLOT_ID = -1; torch advanced indexing would WRAP -1 to row `size`
            # (an allocatable live slot) and corrupt a real request's shift state.
            # Route pads to row 0, the MambaPool's reserved never-allocated slot
            # (free_slots = arange(1, size+1)); pad outputs are discarded anyway.
            safe_idx = torch.clamp_min(cache_indices, 0)
            # advanced indexing already materializes a fresh tensor (no aliasing)
            prev = conv[safe_idx, :, 0]  # [n, hidden]
            conv[safe_idx, :, 0] = x.to(conv.dtype)
            return prev.to(x.dtype)

        # extend (packed B=1, varlen via query_start_loc)
        qsl = md.query_start_loc.to(torch.long)
        starts = qsl[:-1]
        ends = qsl[1:]
        shifted = torch.empty_like(x)
        if x.shape[0] > 1:
            shifted[1:] = x[:-1]
        # Same pad convention as the decode path: route PAD_SLOT_ID = -1 to the
        # pool's reserved row 0. Fresh sequences' slots are zeroed upstream on
        # the forward stream (`mamba_needs_clear` is set at slot alloc,
        # collected in prepare_for_extend, and cleared by the ModelRunner
        # before any layer reads the pool), so this backend only handles pads.
        safe_idx = torch.clamp_min(cache_indices, 0)
        # first token of each sequence reads the stored prev-token (0 for fresh
        # reqs; the correct carry-in for chunked prefill / prefix continuation).
        shifted[starts] = conv[safe_idx, :, 0].to(x.dtype)
        # store last token of each sequence for the next chunk / decode.
        conv[safe_idx, :, 0] = x[ends - 1].to(conv.dtype)
        return shifted

    # ---- WKV recurrence (decode + extend both -> the wkv_recurrent kernel) ----
    def recurrence(
        self,
        r: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kk: torch.Tensor,
        a: torch.Tensor,
        layer_id: int,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """r,w,k,kk,a: [num_tokens, H, K]; v: [num_tokens, H, V]. Returns [num_tokens, H, V]."""
        cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        temporal = cache.temporal  # [size+1, H, K, V] fp32
        md = self.forward_metadata
        cache_indices = md.mamba_cache_indices

        if forward_batch.forward_mode.is_decode_or_idle():
            # [bs, H, *] -> [bs, 1, H, *]
            r4 = r.unsqueeze(1).contiguous()
            w4 = w.unsqueeze(1).contiguous()
            k4 = k.unsqueeze(1).contiguous()
            v4 = v.unsqueeze(1).contiguous()
            kk4 = kk.unsqueeze(1).contiguous()
            a4 = a.unsqueeze(1).contiguous()
            # In-place indexed state: the kernel reads/writes temporal[cache_indices]
            # directly (skips the gather+scatter copies; ~3x less state traffic at large
            # bsz). Same reduction math + bits -> greedy decoding stays token-exact.
            o, _ = wkv_recurrent(
                r4,
                w4,
                k4,
                v4,
                kk4,
                a4,
                scale=self.scale,
                state_pool=temporal,
                cache_indices=cache_indices,
            )
            return o.squeeze(1)  # [bs, H, V]

        # extend: packed B=1, varlen -> the same recurrent triton kernel.
        # Same pad convention as token_shift (fresh-slot zeroing likewise
        # happens upstream before the layers run).
        safe_idx = torch.clamp_min(cache_indices, 0)
        init_state = temporal[safe_idx].contiguous().float()  # [N, H, K, V]
        cu = md.query_start_loc.to(torch.int64)
        r1 = r.unsqueeze(0).contiguous()
        w1 = w.unsqueeze(0).contiguous()
        k1 = k.unsqueeze(0).contiguous()
        v1 = v.unsqueeze(0).contiguous()
        kk1 = kk.unsqueeze(0).contiguous()
        a1 = a.unsqueeze(0).contiguous()
        o, final_state = wkv_recurrent(
            r1,
            w1,
            k1,
            v1,
            kk1,
            a1,
            scale=self.scale,
            initial_state=init_state,
            output_final_state=True,
            cu_seqlens=cu,
        )
        temporal[safe_idx] = final_state.to(temporal.dtype)
        return o.squeeze(0)  # [total_T, H, V]

    # The model calls token_shift/recurrence directly; these are not used.
    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError(
            "Rwkv7AttnBackend.recurrence/token_shift are called directly by the model."
        )

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError(
            "Rwkv7AttnBackend.recurrence/token_shift are called directly by the model."
        )
