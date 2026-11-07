"""Prepared FlashInfer consumer for the Kimi T=1 fused KDA handoff.

Only the explicit FlashInfer decode selection prepares this adapter. Import and
API probing, as well as convolution-weight packing, happen after weight loading
and before KV sizing/capture. A rejected layout returns to the existing native
fused path; a launched stateful FlashInfer call must never be retried there.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import Callable, Optional

import torch

# Kimi K3/K3.1 KDA geometry after TP8 sharding: 12 local heads x 128.
# Keep the admission explicit; this consumer is not a generic KDA adapter.
_KIMI_KDA_HEADS = 12
_KIMI_KDA_HEAD_DIM = 128


@lru_cache(maxsize=1)
def _get_fused_kda_decode() -> Optional[Callable]:
    try:
        module = import_module("flashinfer.kda_decode")
        # Some FI builds keep the public wrapper even when its optional
        # implementation failed to import. Reject that before any state update.
        if getattr(module, "_FUSED_KDA_DECODE_AVAILABLE", True) is False:
            return None
        fused_kda_decode = getattr(module, "fused_kda_decode", None)
        parameters = inspect.signature(fused_kda_decode).parameters
    except (ImportError, TypeError, ValueError):
        return None
    required = {
        "x",
        "weight",
        "conv_state",
        "raw_gate",
        "raw_beta",
        "A_log",
        "dt_bias",
        "state_indices",
        "state",
        "output_gate",
        "norm_weight",
        "lower_bound",
        "norm_eps",
        "backend",
        "state_indices_mode",
    }
    return fused_kda_decode if required.issubset(parameters) else None


@dataclass(frozen=True)
class PreparedFlashInferKDADecode:
    kernel: Callable
    weight: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    norm_weight: torch.Tensor
    norm_eps: float
    lower_bound: Optional[float]
    heads: int
    head_dim: int

    @property
    def seg(self) -> int:
        return self.heads * self.head_dim

    def unsupported_layout_reason(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        onorm_gate: torch.Tensor,
    ) -> Optional[str]:
        """Return why this layout is not covered, or ``None`` when it is.

        Metadata only; do not read indices or copy strided inputs. This is the
        single source of truth for ``covered`` and the one-shot provider log, so
        an unexpected layout fallback is diagnosable instead of silent.

        Ordinary decode owns unique positive Mamba slots. Slot zero is reserved
        for dummy/idle traffic, and graph padding is -1. FlashInfer's
        ``unique_or_null`` assertion therefore allows both kinds of null rows.
        Verify and ReplaySSM do not enter this consumer.
        """
        heads, width, seg = self.heads, self.head_dim, self.seg
        if mixed_qkv.ndim != 2:
            return f"mixed_qkv is {mixed_qkv.ndim}-D, expected 2-D"
        batch = mixed_qkv.shape[0]
        if batch <= 0:
            return "empty batch"
        tensors = (mixed_qkv, a, b, conv_states, ssm_states, cache_indices, onorm_gate)
        if any(t.device != self.weight.device for t in tensors):
            return "input tensors are on a different device than the prepared weights"

        if tuple(mixed_qkv.shape) != (batch, 3 * seg):
            return f"mixed_qkv shape {tuple(mixed_qkv.shape)} != {(batch, 3 * seg)}"
        if mixed_qkv.dtype != torch.bfloat16:
            return f"mixed_qkv dtype {mixed_qkv.dtype} != torch.bfloat16"
        if mixed_qkv.stride(1) != 1 or mixed_qkv.data_ptr() % 16 != 0:
            return "mixed_qkv is not a 16-byte-aligned row view"

        if (
            tuple(a.shape) != (batch, seg)
            or a.dtype != torch.bfloat16
            or not a.is_contiguous()
            or a.data_ptr() % 16 != 0
        ):
            return "raw_gate is not a 16-byte-aligned contiguous BF16 [batch, seg]"
        if (
            tuple(b.shape) != (batch, heads)
            or b.dtype != torch.bfloat16
            or b.stride(1) != 1
            or b.data_ptr() % 16 != 0
        ):
            return "raw_beta is not a 16-byte-aligned BF16 [batch, heads] row view"
        if (
            tuple(onorm_gate.shape) != (batch, seg)
            or onorm_gate.dtype != torch.bfloat16
            or onorm_gate.stride(1) != 1
            or onorm_gate.data_ptr() % 16 != 0
        ):
            return "output gate is not a 16-byte-aligned BF16 [batch, seg] row view"

        if (
            conv_states.ndim != 3
            or tuple(conv_states.shape[1:]) != (3, 3 * seg)
            or conv_states.dtype != torch.bfloat16
            or conv_states.stride(1) != 3 * seg
            or conv_states.stride(2) != 1
            or conv_states.stride(0) < 9 * seg
            or conv_states.stride(0) % 4 != 0
            or conv_states.data_ptr() % 8 != 0
        ):
            return "conv state is not the expected BF16 envelope-strided view"
        if (
            ssm_states.ndim != 4
            or tuple(ssm_states.shape[1:]) != (heads, width, width)
            or ssm_states.shape[0] != conv_states.shape[0]
            or ssm_states.dtype != torch.float32
            or ssm_states.stride()[1:] != (width * width, width, 1)
            or ssm_states.stride(0) < heads * width * width
            or ssm_states.stride(0) % 4 != 0
            or ssm_states.data_ptr() % 16 != 0
        ):
            return "SSM state is not the expected FP32 envelope-strided view"
        if (
            cache_indices.ndim != 1
            or cache_indices.shape[0] != batch
            or cache_indices.dtype != torch.int32
            or not cache_indices.is_contiguous()
            or cache_indices.data_ptr() % 16 != 0
        ):
            return "cache indices are not a 16-byte-aligned contiguous int32 [batch]"
        return None

    def covered(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        onorm_gate: torch.Tensor,
    ) -> bool:
        """Check metadata only; ``unsupported_layout_reason`` owns the rules."""
        return (
            self.unsupported_layout_reason(
                mixed_qkv,
                a,
                b,
                conv_states,
                ssm_states,
                cache_indices,
                onorm_gate,
            )
            is None
        )

    def run(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        onorm_gate: torch.Tensor,
    ) -> torch.Tensor:
        batch = mixed_qkv.shape[0]
        # These are views, including the SD convolution-cache transpose. Keep
        # the original envelope slot strides and the caller-owned state pools.
        return self.kernel(
            x=mixed_qkv,
            weight=self.weight,
            conv_state=conv_states.transpose(-1, -2),
            raw_gate=a.view(1, batch, self.heads, self.head_dim),
            raw_beta=b.unsqueeze(0),
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            state_indices=cache_indices,
            state=ssm_states,
            output_gate=onorm_gate.unflatten(-1, (self.heads, self.head_dim)),
            norm_weight=self.norm_weight,
            lower_bound=self.lower_bound,
            norm_eps=self.norm_eps,
            backend="auto",
            state_indices_mode="unique_or_null",
        )


def prepare_fused_kda_decode(
    fused_static: tuple,
    dt_bias: torch.Tensor,
    *,
    has_conv_bias: bool,
    lower_bound: Optional[float],
) -> Optional[PreparedFlashInferKDADecode]:
    """Prepare once; old FlashInfer APIs and unsupported static inputs opt out."""
    if has_conv_bias:
        return None
    try:
        kernel = _get_fused_kda_decode()
    except RuntimeError:
        # A transient probe failure keeps this layer on native decode without
        # caching process-wide unavailability for later layer preparation.
        return None
    if kernel is None:
        return None
    w_q, w_k, w_v, _conv_bias, A_log, norm_weight, norm_eps = fused_static
    if (
        any(
            tuple(w.shape) != (4, _KIMI_KDA_HEADS * _KIMI_KDA_HEAD_DIM)
            or w.dtype != torch.float32
            for w in (w_q, w_k, w_v)
        )
        or A_log.dtype != torch.float32
        or A_log.numel() != _KIMI_KDA_HEADS
        or dt_bias.dtype != torch.float32
        or tuple(dt_bias.shape) != (_KIMI_KDA_HEADS * _KIMI_KDA_HEAD_DIM,)
        or norm_weight.dtype != torch.float32
        or tuple(norm_weight.shape) != (_KIMI_KDA_HEAD_DIM,)
        or any(
            not t.is_contiguous() or t.data_ptr() % 16 != 0
            for t in (A_log, dt_bias, norm_weight)
        )
        or not math.isfinite(norm_eps)
        or norm_eps < 0
        or (
            lower_bound is not None
            and (not math.isfinite(lower_bound) or lower_bound >= 0)
        )
        or any(t.device != w_q.device for t in (w_k, w_v, A_log, dt_bias, norm_weight))
    ):
        return None
    return PreparedFlashInferKDADecode(
        kernel=kernel,
        weight=torch.stack((w_q, w_k, w_v)),
        A_log=A_log.reshape(-1).contiguous(),
        dt_bias=dt_bias.contiguous(),
        norm_weight=norm_weight.contiguous(),
        norm_eps=float(norm_eps),
        lower_bound=lower_bound,
        heads=_KIMI_KDA_HEADS,
        head_dim=_KIMI_KDA_HEAD_DIM,
    )
