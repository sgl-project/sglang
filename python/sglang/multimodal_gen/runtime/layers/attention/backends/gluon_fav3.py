# SPDX-License-Identifier: Apache-2.0
"""SGLang diffusion backend for the gfx1250 Gluon FAv3 kernel."""

from __future__ import annotations

import functools
import logging

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

logger = logging.getLogger(__name__)

_logged_fallback_reasons: set[str] = set()


@functools.lru_cache(maxsize=1)
def _load_gluon_fav3():
    try:
        from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import (
            gluon_fav3_attention,
        )
    except (AttributeError, ImportError) as exc:
        return None, str(exc)
    return gluon_fav3_attention, None


class GluonFAv3Backend(AttentionBackend):
    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.GLUON_FAV3

    @staticmethod
    def get_impl_cls() -> type[GluonFAv3Impl]:
        return GluonFAv3Impl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        raise NotImplementedError(
            "Gluon FAv3 backend does not have a metadata builder."
        )


class GluonFAv3Impl(AttentionImpl):
    """Dense BF16 D128 MHA on gfx1250, with AITER as the safe fallback."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        dropout_p: float = 0.0,
        **extra_impl_args,
    ) -> None:
        del prefix, extra_impl_args
        if num_kv_heads is not None and num_heads % num_kv_heads != 0:
            raise ValueError(
                f"Gluon FAv3 requires num_heads ({num_heads}) to be a "
                f"multiple of num_kv_heads ({num_kv_heads})."
            )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.dropout_p = dropout_p
        self._aiter_fallback: AttentionImpl | None = None

    def _unsupported_reason(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> str | None:
        if self.causal:
            return "causal attention"
        if self.dropout_p != 0.0:
            return "nonzero dropout"
        if self.head_size != 128:
            return f"head_size={self.head_size}"
        if self.num_kv_heads != self.num_heads:
            return (
                f"grouped-query attention ({self.num_heads} Q heads, "
                f"{self.num_kv_heads} KV heads)"
            )
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            return "non-BSHD tensors"
        if query.dtype != torch.bfloat16:
            return f"dtype={query.dtype}"
        if key.dtype != query.dtype or value.dtype != query.dtype:
            return "mismatched Q/K/V dtypes"
        if (
            not query.is_cuda
            or key.device != query.device
            or value.device != query.device
        ):
            return "Q/K/V are not on one ROCm device"
        if query.shape[0] != key.shape[0] or key.shape != value.shape:
            return "incompatible Q/K/V batch or KV shapes"
        if query.shape[3] != self.head_size:
            return f"query shape={tuple(query.shape)}"
        if key.shape[3] != self.head_size:
            return f"key shape={tuple(key.shape)}"
        if query.shape[2] != key.shape[2]:
            return (
                f"runtime grouped-query attention ({query.shape[2]} Q heads, "
                f"{key.shape[2]} KV heads)"
            )
        if min(query.shape[1], key.shape[1]) <= 0:
            return "empty sequence"
        if query.stride(-1) != 1 or key.stride(-1) != 1 or value.stride(-1) != 1:
            return "non-contiguous head dimension"

        arch = getattr(
            torch.cuda.get_device_properties(query.device), "gcnArchName", ""
        )
        if "gfx1250" not in arch:
            return f"GPU architecture {arch or 'unknown'}"

        kernel, import_error = _load_gluon_fav3()
        if kernel is None:
            return f"Gluon FAv3 unavailable: {import_error}"
        return None

    def _fallback(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        reason: str,
    ) -> torch.Tensor:
        if reason not in _logged_fallback_reasons:
            logger.warning("Gluon FAv3 cannot serve %s; falling back to AITER.", reason)
            _logged_fallback_reasons.add(reason)

        if self._aiter_fallback is None:
            from sglang.multimodal_gen.runtime.layers.attention.backends.aiter import (
                AITerImpl,
            )

            self._aiter_fallback = AITerImpl(
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
                dropout_p=self.dropout_p,
            )
        return self._aiter_fallback.forward(query, key, value, attn_metadata)

    @torch.compiler.disable
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        reason = self._unsupported_reason(query, key, value)
        if reason is not None:
            return self._fallback(
                query, key, value, attn_metadata=attn_metadata, reason=reason
            )

        kernel, _ = _load_gluon_fav3()
        assert kernel is not None
        return kernel(
            query,
            key,
            value,
            softmax_scale=self.softmax_scale,
        )


__all__ = ["GluonFAv3Backend", "GluonFAv3Impl"]
