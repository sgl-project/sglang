# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

try:
    from lite_attention import LiteAttention as _LiteAttention

    _lite_attention_available = True
except Exception:
    _LiteAttention = None
    _lite_attention_available = False


def _get_threshold_from_server_args(default: float = -6.0) -> float:
    sargs = get_global_server_args()
    thr = getattr(sargs, "lite_attn_threshold", None)
    if thr is not None:
        try:
            return float(thr)
        except Exception:
            pass
    cfg = getattr(sargs, "attention_backend_config", None)
    if cfg is None:
        return default
    try:
        thr_cfg = getattr(cfg, "threshold", None)
        if thr_cfg is None and isinstance(cfg, dict):
            thr_cfg = cfg.get("threshold")
        return float(default if thr_cfg is None else thr_cfg)
    except Exception:
        return default


def _get_lite_attn_params_from_server_args() -> dict[str, Any]:
    sargs = get_global_server_args()
    return {
        "threshold": getattr(sargs, "lite_attn_threshold", -6.0),
        "enable_skipping": getattr(sargs, "lite_attn_enable_skipping", True),
        "max_batch_size": getattr(sargs, "lite_attn_max_batch_size", 2),
        "reverse_skip_list": getattr(sargs, "lite_attn_reverse_skip_list", True),
        "use_int8": getattr(sargs, "lite_attn_use_int8", False),
    }


class LiteAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # LiteAttention builds on FA3; keep the same head size set as FlashAttention backend.
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.LITE_ATTN

    @staticmethod
    def get_impl_cls() -> type["LiteAttentionImpl"]:
        return LiteAttentionImpl


class LiteAttentionImpl(AttentionImpl):
    """LiteAttention wrapper backend.

    Notes:
    - LiteAttention maintains internal skip state; each AttentionImpl instance maps to one
      attention layer, so state is not shared across layers (desired).
    - LiteAttention's skipping assumes self-attention-like shapes; for cross-attention
      (q_len != k_len), we automatically disable skipping for correctness.
    """

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
        if not _lite_attention_available:
            raise ImportError(
                "LiteAttention backend is not installed. "
                "Install it (Hopper-only) and retry."
            )
        self.causal = causal
        self.softmax_scale = softmax_scale

        # Defaults come from ServerArgs.lite_attn_*; allow per-layer overrides via extra_impl_args.
        params = _get_lite_attn_params_from_server_args()
        threshold = extra_impl_args.pop("threshold", None)
        if threshold is None:
            threshold = params.get("threshold", None)
        if threshold is None:
            threshold = _get_threshold_from_server_args(default=-6.0)

        enable_skipping = extra_impl_args.pop(
            "enable_skipping", params.get("enable_skipping", True)
        )
        max_batch_size = extra_impl_args.pop(
            "max_batch_size", params.get("max_batch_size", 2)
        )
        reverse_skip_list = extra_impl_args.pop(
            "reverse_skip_list", params.get("reverse_skip_list", True)
        )
        use_int8 = extra_impl_args.pop("use_int8", params.get("use_int8", False))

        self._enable_skipping_default = bool(enable_skipping)
        self._lite = _LiteAttention(  # type: ignore[misc]
            enable_skipping=self._enable_skipping_default,
            threshold=float(threshold),
            max_batch_size=int(max_batch_size),
            reverse_skip_list=bool(reverse_skip_list),
            use_int8=bool(use_int8),
        )
        if prefix:
            logger.debug("Initialized LiteAttentionImpl for %s", prefix)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        *,
        return_softmax_lse: bool = False,
    ):
        # LiteAttention supports non-causal attention; enforce correctness if causal requested.
        if self.causal:
            raise ValueError("LiteAttention backend does not support causal attention.")

        # Ring attention calls attention kernels with attn_metadata=None and may invoke
        # the kernel multiple times with different K/V segments. Disable skipping in
        # this mode for correctness; this still uses FA3 under the hood.
        if attn_metadata is None:
            if self._lite.enable_skipping:
                self._lite.enable_skip_optimization(enable=False)

        # LiteAttention's skipping path assumes square-ish attention / shared seq_len.
        # For cross-attn (q_len != k_len), disable skipping but still run attention.
        q_len = query.shape[1]
        k_len = key.shape[1]
        skipping = self._enable_skipping_default and (q_len == k_len)
        if skipping != self._lite.enable_skipping:
            self._lite.enable_skip_optimization(enable=skipping)

        # LiteAttention expects BSHD and uses scale as softmax scale.
        out = self._lite(
            query,
            key,
            value,
            scale=self.softmax_scale,
            return_softmax_lse=return_softmax_lse,
        )
        return out

