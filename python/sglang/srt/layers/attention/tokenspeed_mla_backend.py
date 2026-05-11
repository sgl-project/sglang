from __future__ import annotations

"""
Support attention backend for the tokenspeed-mla CuTe DSL kernel on Blackwell.

Subclasses :class:`TRTLLMMLABackend` and overrides only ``_run_decode_kernel``
to dispatch decode / spec-verify / draft-extend through
``tokenspeed_mla.tokenspeed_mla_decode``. All metadata, KV-cache layout,
CUDA-graph plumbing, FP8 quantize/rope, and draft-extend padding logic are
inherited unchanged. Pure prefill (extend without speculative) stays on the
parent's TRT-LLM ragged path.
"""

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    TRTLLMMLAMultiStepDraftBackend,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# Upper bound for tokenspeed_mla_decode's workspace:
#   num_sms * num_heads * max_q_len * (kv_lora_rank + 1) * sizeof(float32)
# MAX_Q_LEN=8 covers q_len up to 8 (e.g. EAGLE3 num_draft_tokens=4 + headroom).
_TOKENSPEED_MAX_Q_LEN = 8

# Shared across backend instances on the same device (multi-step draft
# creates one TokenspeedMLABackend per draft step).
_g_tokenspeed_workspace: dict[torch.device, torch.Tensor] = {}


def _get_tokenspeed_workspace(
    device: torch.device, num_heads: int, kv_lora_rank: int
) -> torch.Tensor:
    from tokenspeed_mla import get_num_sm

    needed = (
        get_num_sm(device) * num_heads * _TOKENSPEED_MAX_Q_LEN * (kv_lora_rank + 1) * 4
    )
    existing = _g_tokenspeed_workspace.get(device)
    if existing is None or existing.numel() < needed:
        _g_tokenspeed_workspace[device] = torch.empty(
            needed, dtype=torch.int8, device=device
        )
    return _g_tokenspeed_workspace[device]


class TokenspeedMLABackend(TRTLLMMLABackend):
    """tokenspeed-mla CuTe DSL attention backend (Blackwell SM100, FP8 KV)."""

    def __init__(
        self,
        model_runner: "ModelRunner",
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        q_indptr_decode_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            model_runner,
            skip_prefill,
            kv_indptr_buf,
            q_indptr_decode_buf,
        )

        if self.data_type != torch.float8_e4m3fn:
            raise ValueError(
                "tokenspeed_mla backend requires --kv-cache-dtype fp8_e4m3, "
                f"got data_type={self.data_type}."
            )
        if self.page_size not in (32, 64):
            raise ValueError(
                "tokenspeed_mla backend requires page_size in {32, 64}, "
                f"got page_size={self.page_size}."
            )

        self._tokenspeed_workspace: Optional[torch.Tensor] = None

    def _ensure_workspace(self, device: torch.device) -> torch.Tensor:
        if (
            self._tokenspeed_workspace is None
            or self._tokenspeed_workspace.device != device
        ):
            self._tokenspeed_workspace = _get_tokenspeed_workspace(
                device, self.num_q_heads, self.kv_lora_rank
            )
        return self._tokenspeed_workspace

    def _run_decode_kernel(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        layer: "RadixAttention",
    ) -> torch.Tensor:
        from tokenspeed_mla import tokenspeed_mla_decode

        # tokenspeed splits the trtllm-gen ``bmm1_scale`` into two:
        # softmax_scale applied at QK^T, output_scale applied at attn @ V.
        # Both pick up the model's k_scale because the FP8 KV is stored without
        # explicit rescaling — K and V share the same per-tensor k_scale via
        # the kv_lora_rank prefix.
        k_scale = getattr(layer, "k_scale_float", None) or 1.0
        softmax_scale = float(layer.scaling) * float(k_scale)
        output_scale = float(k_scale)

        seq_lens_i32 = (
            seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
        )
        return tokenspeed_mla_decode(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=self._ensure_workspace(query.device),
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens_i32,
            max_seq_len=int(max_seq_len),
            softmax_scale=softmax_scale,
            output_scale=output_scale,
        )


class TokenspeedMLAMultiStepDraftBackend(TRTLLMMLAMultiStepDraftBackend):
    """Multi-step draft backend for tokenspeed_mla used by EAGLE."""

    def __init__(
        self, model_runner: "ModelRunner", topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner, topk, speculative_num_steps)
        # The parent constructor populates self.attn_backends with TRT-LLM
        # instances; replace them with tokenspeed instances sharing the
        # parent's kv_indptr / q_indptr buffers.
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i] = TokenspeedMLABackend(
                model_runner,
                skip_prefill=True,
                kv_indptr_buf=self.kv_indptr[i],
                q_indptr_decode_buf=self.q_indptr_decode,
            )
