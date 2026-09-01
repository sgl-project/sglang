"""Moonmath MLA decode backend for DeepSeek-V3 on CDNA3 (MI300X / gfx942).

Subclasses AiterAttnBackend: decode with fp8 KV and H<=16 routes to the
moonmath_attention A16W8 kernel (bf16 Q / fp8 KV, device-driven paged,
cuda-graph-safe, reads sglang's existing fused-576 MLATokenToKVPool directly).
Everything else (bf16 KV, H>16, prefill, spec-verify) falls back to aiter.
"""

from __future__ import annotations

import os

import torch

from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

KV_LORA_RANK = 512
KV_CACHE_DIM = 576  # 512 latent + 64 rope


class MoonmathMLABackend(AiterAttnBackend):
    """MLA decode backend: A16W8 kernel for H<=16 fp8 KV, else aiter."""

    def __init__(self, model_runner):
        super().__init__(model_runner)
        import moonmath_attention.mla as mla  # fail fast if missing

        self._mla = mla
        self._mla_ok = bool(self.use_mla)
        self._disabled = os.environ.get("SGLANG_MOONMATH_MLA_DISABLE") == "1"
        self._fp8_dtype = torch.float8_e4m3fnuz
        self._fp8_kv = self._mla_ok and (self.kv_cache_dtype == self._fp8_dtype)

        # Per-bs FIXED kv-split (frozen at first call / capture; reused on replay).
        self._dec_parts: dict[int, int] = {}
        model_config = getattr(model_runner, "model_config", None)
        self._mla_max_ctx = (
            model_config.context_len if model_config is not None else 131072
        )
        # int32 staging for device seq_lens (sglang gives int64 in eager mode).
        self._mla_seqlen_i32 = torch.zeros(
            8192, dtype=torch.int32, device=model_runner.device
        )

    # A16W8 kernel serves H<=16. H=128 (DSV3) falls back to aiter.
    _A16W8_DECODE_MAX_HEADS = 16

    def _decode_eligible(self, q, layer: RadixAttention, fb: ForwardBatch) -> bool:
        return (
            self._mla_ok
            and not self._disabled
            and self._fp8_kv
            and q.dtype == torch.bfloat16
            and layer.tp_q_head_num <= self._A16W8_DECODE_MAX_HEADS
            and layer.qk_head_dim == KV_CACHE_DIM
            and layer.v_head_dim == KV_LORA_RANK
            and layer.tp_k_head_num == 1
            and layer.logit_cap == 0
            and fb.forward_mode.is_decode()
            and fb.spec_info is None
            and fb.batch_size <= self._mla_seqlen_i32.numel()
            and self.forward_metadata is not None
            and self.forward_metadata.kv_indices is not None
            and self.forward_metadata.kv_indptr is not None
        )

    # ── decode ───────────────────────────────────────────────────────────────
    def forward_decode(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, sinks=None
    ):
        if sinks is not None or not self._decode_eligible(q, layer, forward_batch):
            return super().forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, sinks
            )

        mla = self._mla
        fb = forward_batch
        B = fb.batch_size
        H = layer.tp_q_head_num

        if save_kv_cache and k is not None:
            self.token_to_kv_pool.set_kv_buffer(layer, fb.out_cache_loc, k, v)

        kv_pool = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_indices = self.forward_metadata.kv_indices.to(torch.int32)
        kv_indptr = self.forward_metadata.kv_indptr.to(torch.int32)
        seq_lens = self._mla_seqlen_i32[:B]
        seq_lens.copy_(fb.seq_lens)
        k_descale = (
            float(layer.k_scale) if getattr(layer, "k_scale", None) is not None else 1.0
        )

        parts = self._dec_parts.get(B)
        if parts is None:
            parts = mla.mla_decode_a16w8_plan_parts_capped(
                B, H, self._mla_max_ctx, KV_LORA_RANK
            )
            self._dec_parts[B] = parts

        q = q.reshape(B, H, layer.qk_head_dim)
        q_lat = q[..., :KV_LORA_RANK].contiguous()
        q_pe = q[..., KV_LORA_RANK:].contiguous()
        out = torch.empty(B, H, KV_LORA_RANK, dtype=torch.bfloat16, device=q.device)
        mla.mla_decode_a16w8_paged_dev(
            q_lat,
            q_pe,
            kv_pool,
            out,
            seq_lens,
            None,
            kv_indices,
            kv_indptr,
            parts,
            layer.scaling,
            k_descale,
            1.0,
        )
        return out.reshape(B, H * KV_LORA_RANK)
