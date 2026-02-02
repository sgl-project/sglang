from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.layers.utils import register_custom_op
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

try:
    from lite_attention import LiteAttention as _LiteAttention

    _lite_attention_available = True
except ImportError:
    _LiteAttention = None
    _lite_attention_available = False

_LITE_ATTN_REGISTRY: dict[int, Any] = {}


def lite_attn_fake_out(
    handle: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    is_ring: bool = False,
) -> torch.Tensor:
    b, s, h = q.shape[:3]
    dv = v.shape[-1]
    return q.new_empty((b, s, h, dv))


def lite_attn_fake_out_lse(
    handle: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    is_ring: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = lite_attn_fake_out(handle, q, k, v, softmax_scale, is_ring)
    b, s, h = q.shape[:3]
    lse = q.new_empty((b, h, s), dtype=torch.float32)
    return out, lse


@register_custom_op(fake_impl=lite_attn_fake_out)
def lite_attn_func_op(
    handle: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    is_ring: bool = False,
) -> torch.Tensor:
    lite = _LITE_ATTN_REGISTRY.get(int(handle))
    if lite is None:
        raise RuntimeError(f"Invalid LiteAttention handle: {handle}")

    skipping = (q.shape[1] == k.shape[1]) and (not is_ring)
    if skipping != bool(getattr(lite, "enable_skipping", False)):
        lite.enable_skip_optimization(enable=skipping)
    return lite(q, k, v, scale=softmax_scale, return_softmax_lse=False)


@register_custom_op(fake_impl=lite_attn_fake_out_lse)
def lite_attn_func_op_lse(
    handle: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    is_ring: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    lite = _LITE_ATTN_REGISTRY.get(int(handle))
    if lite is None:
        raise RuntimeError(f"Invalid LiteAttention handle: {handle}")
    skipping = (q.shape[1] == k.shape[1]) and (not is_ring)
    if skipping != bool(getattr(lite, "enable_skipping", False)):
        lite.enable_skip_optimization(enable=skipping)
    out, lse = lite(q, k, v, scale=softmax_scale, return_softmax_lse=True)
    return out, lse


def _get_lite_attn_params_from_server_args() -> dict[str, Any]:
    sargs = get_global_server_args()
    return {
        "threshold": getattr(sargs, "lite_attn_threshold", -6.0),
        "max_batch_size": getattr(sargs, "lite_attn_max_batch_size", 2),
        "reverse_skip_list": getattr(sargs, "lite_attn_reverse_skip_list", True),
        "use_int8": getattr(sargs, "lite_attn_use_int8", False),
    }


class LiteAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.LITE_ATTN

    @staticmethod
    def get_impl_cls() -> type["LiteAttentionImpl"]:
        return LiteAttentionImpl


class LiteAttentionImpl(AttentionImpl):
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
                "LiteAttention backend is not installed. Install it (Hopper-only) and retry."
            )

        self.causal = causal
        self.softmax_scale = softmax_scale

        params = _get_lite_attn_params_from_server_args()
        threshold = extra_impl_args.pop("threshold", params["threshold"])
        max_batch_size = extra_impl_args.pop("max_batch_size", params["max_batch_size"])
        reverse_skip_list = extra_impl_args.pop(
            "reverse_skip_list", params["reverse_skip_list"]
        )
        use_int8 = extra_impl_args.pop("use_int8", params["use_int8"])

        self._lite = _LiteAttention(
            enable_skipping=True,
            threshold=float(threshold),
            max_batch_size=int(max_batch_size),
            reverse_skip_list=bool(reverse_skip_list),
            use_int8=bool(use_int8),
        )
        self._lite_handle = id(self._lite)
        _LITE_ATTN_REGISTRY[self._lite_handle] = self._lite
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        *,
        return_softmax_lse: bool = False,
    ):
        if self.causal:
            raise ValueError("LiteAttention backend does not support causal attention.")
        is_ring = attn_metadata is None
        if return_softmax_lse:
            return lite_attn_func_op_lse(
                self._lite_handle, query, key, value, self.softmax_scale, is_ring
            )
        return lite_attn_func_op(
            self._lite_handle, query, key, value, self.softmax_scale, is_ring
        )
