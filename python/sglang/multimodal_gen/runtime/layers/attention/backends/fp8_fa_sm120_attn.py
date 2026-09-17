# SPDX-License-Identifier: Apache-2.0
"""FP8 attention backend for SM120 GPUs (GeForce RTX 50, RTX PRO Blackwell).

Opt in with ``--attention-backend fp8_fa_sm120``. Q, K and V are quantized per head to
E4M3 on the fly (two fused Triton passes), attention runs in FP8 with FP32
accumulation, and the output is BF16. Expect about 5% relative RMS against cuDNN BF16
per call on normally distributed inputs.

Scope: dense, non-causal, batch 1, head_dim 128, BF16 inputs on an SM120 device.
Everything else goes to cuDNN SDPA. Each distinct sequence length compiles once
(about 10 s); later calls with the same shape and strides reuse the plan.
"""

import torch

from sglang.kernels.ops.attention.fp8_fa_sm120 import (
    HEAD_DIM,
    FP8AttentionPlan,
    plan_key,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    trailing_padding_used_len,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import CudnnSDPAImpl
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Shared by every impl instance: each DiT layer owns an impl, and per-layer plans would
# hold one copy of the E4M3 and output buffers per layer. Layers run in sequence on one stream.
_PLAN_CACHE: dict[tuple, FP8AttentionPlan] = {}


class FP8FlashAttentionSM120Backend(AttentionBackend):
    accept_output_buffer: bool = False

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [HEAD_DIM]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.FP8_FA_SM120

    @staticmethod
    def get_impl_cls() -> type["FP8FlashAttentionSM120Impl"]:
        return FP8FlashAttentionSM120Impl


class FP8FlashAttentionSM120Impl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.fallback = CudnnSDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            **extra_impl_args,
        )
        self.plans = _PLAN_CACHE
        self._reported_fallbacks: set[str] = set()

    # --- dispatch -------------------------------------------------------------

    def _fallback_reason(self, query, key, value) -> str | None:
        """None when the kernel applies to these [S, H, D] tensors, else why not."""
        if self.causal:
            return "causal attention"
        if query.shape != key.shape or key.shape != value.shape:
            return "Q/K/V shapes differ"
        if query.ndim != 3 or query.shape[-1] != HEAD_DIM:
            return f"head_dim {query.shape[-1]} (kernel is head_dim {HEAD_DIM})"
        if query.dtype != torch.bfloat16:
            return f"dtype {query.dtype} (kernel takes BF16)"
        if not query.is_cuda:
            return "non-CUDA tensors"
        if torch.cuda.get_device_capability(query.device) != (12, 0):
            return "device is not SM120"
        return None

    def _report_fallback(self, reason: str) -> None:
        if reason in self._reported_fallbacks:
            return
        self._reported_fallbacks.add(reason)
        logger.warning(
            "fp8_fa_sm120 attention: %s; using cuDNN SDPA for these calls.", reason
        )

    # --- kernel path ----------------------------------------------------------

    def _get_plan(self, query, key, value) -> FP8AttentionPlan:
        key_ = plan_key(query, self.softmax_scale)
        plan = self.plans.get(key_)
        if plan is None:
            logger.info(
                "fp8_fa_sm120 attention: compiling for S=%d H=%d (once per shape)",
                query.shape[0],
                query.shape[1],
            )
            plan = FP8AttentionPlan(query, key, value, self.softmax_scale)
            self.plans[key_] = plan
        else:
            plan.bind_inputs(query, key, value)
        return plan

    def _run(self, query, key, value):
        """[S, H, 128] strided BF16 in, the plan's contiguous [S, H, 128] BF16 out."""
        plan = self._get_plan(query, key, value)
        plan.prepare()
        output, _ = plan.launch_prepared()
        return output

    # --- AttentionImpl --------------------------------------------------------

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # [B, S, H, D]; the kernel handles one dense sequence at a time.
        if query.shape[0] != 1:
            self._report_fallback(f"batch {query.shape[0]}")
            return self.fallback.forward(query, key, value, attn_metadata)
        reason = self._fallback_reason(query[0], key[0], value[0])
        if reason is not None:
            self._report_fallback(reason)
            return self.fallback.forward(query, key, value, attn_metadata)
        try:
            output = self._run(query[0], key[0], value[0])
        except ValueError as error:
            self._report_fallback(str(error))
            return self.fallback.forward(query, key, value, attn_metadata)
        return output.unsqueeze(0)

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        bounds = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(item) for item in cu_seqlens.tolist())
        )
        reason = self._fallback_reason(query, key, value)
        if reason is not None:
            self._report_fallback(reason)
            return self.fallback.forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=bounds,
            )

        # MiniMax-H3 packs one live document as (0, used, total); the tail is
        # 64-aligned padding that downstream masks, so it stays zero.
        used = trailing_padding_used_len(
            total_tokens=query.shape[0],
            max_seqlen=max_seqlen,
            bounds=bounds,
        )
        try:
            if used is not None:
                live_output = self._run(query[:used], key[:used], value[:used])
                if used == query.shape[0]:
                    return live_output
                output = torch.zeros_like(query)
                output[:used].copy_(live_output)
                return output

            output = torch.empty_like(query)
            for start, stop in zip(bounds[:-1], bounds[1:]):
                if start == stop:
                    continue
                segment_output = self._run(
                    query[start:stop], key[start:stop], value[start:stop]
                )
                output[start:stop].copy_(segment_output)
            return output
        except ValueError as error:
            self._report_fallback(str(error))
            return self.fallback.forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=bounds,
            )
