# SPDX-License-Identifier: Apache-2.0
"""Ascend backend for LLaDA-Image conditioning masks.

Mirrors ``LLaDA2CFGFlashInferAttnBackend`` but computes the custom text/query
block mask with the NPU-native SDPA path instead of a FlashInfer ragged wrapper.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend


class AscendLLaDA2CFGAttnBackend(AscendAttnBackend):
    """Stock Ascend backend plus LLaDA-Image text/query block attention masks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._llada_image_conditioning_mask_active = False

    @property
    def conditioning_mask_active(self) -> bool:
        return self._llada_image_conditioning_mask_active

    def init_forward_metadata(self, forward_batch):
        self._llada_image_conditioning_mask_active = False
        text_lens = forward_batch.llada_image_conditioning_text_lens_cpu
        if text_lens is None:
            return super().init_forward_metadata(forward_batch)

        if not forward_batch.forward_mode.is_extend_without_speculative():
            raise RuntimeError(
                "LLaDA-Image conditioning mask requires a one-shot extend"
            )
        if bool(torch.any(forward_batch.extend_prefix_lens != 0).item()):
            raise RuntimeError(
                "LLaDA-Image conditioning does not support cached prefixes"
            )

        self._llada_image_conditioning_mask_active = True

    def forward_extend(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        save_kv_cache=True,
        **kwargs,
    ):
        if not self._llada_image_conditioning_mask_active:
            return super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )
        return self._forward_llada_image_conditioning(q, k, v, layer, forward_batch)

    def _forward_llada_image_conditioning(self, q, k, v, layer, forward_batch):
        if k is None or v is None:
            raise RuntimeError("LLaDA-Image conditioning requires explicit K/V")

        head_dim = layer.head_dim
        num_q_heads = layer.tp_q_head_num
        num_kv_heads = layer.tp_k_head_num
        num_v_heads = layer.tp_v_head_num

        q = q.contiguous().view(-1, num_q_heads, head_dim)
        k = k.contiguous().view(-1, num_kv_heads, head_dim)
        v = v.contiguous().view(-1, num_v_heads, head_dim)

        # Broadcast K/V heads up to the query head count for GQA models.
        if num_q_heads != num_kv_heads:
            k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
        if num_q_heads != num_v_heads:
            v = v.repeat_interleave(num_q_heads // num_v_heads, dim=1)

        text_lens = [
            int(t) for t in forward_batch.llada_image_conditioning_text_lens_cpu
        ]
        seq_lens = [int(s) for s in forward_batch.seq_lens.tolist()]

        outputs = []
        offset = 0
        for text_len, seq_len in zip(text_lens, seq_lens):
            # [1, num_q_heads, seq_len, head_dim]
            q_i = q[offset : offset + seq_len].transpose(0, 1).unsqueeze(0)
            k_i = k[offset : offset + seq_len].transpose(0, 1).unsqueeze(0)
            v_i = v[offset : offset + seq_len].transpose(0, 1).unsqueeze(0)

            # text tokens attend to text only; query tokens attend to everything.
            mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=q.device)
            mask[:text_len, text_len:] = False
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

            if layer.logit_cap > 0:
                out = self.native_attn.scaled_dot_product_attention_with_softcapping(
                    q_i,
                    k_i,
                    v_i,
                    attn_mask=mask,
                    scale=layer.scaling,
                    logit_cap=layer.logit_cap,
                    logit_capping_method=layer.logit_capping_method,
                )
            else:
                out = F.scaled_dot_product_attention(
                    q_i,
                    k_i,
                    v_i,
                    attn_mask=mask,
                    scale=layer.scaling,
                    is_causal=False,
                )

            # [1, num_q_heads, seq_len, head_dim] -> [seq_len, num_q_heads, head_dim]
            outputs.append(out.squeeze(0).transpose(0, 1))
            offset += seq_len

        return torch.cat(outputs, dim=0).reshape(-1, num_q_heads * head_dim)
