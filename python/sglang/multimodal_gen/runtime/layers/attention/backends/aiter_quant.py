# SPDX-License-Identifier: Apache-2.0
#
# Quantized AITER attention family backend (ROCm / gfx950, gfx942).
#
# One backend, several quant formats selected via --attention-backend-config
# (e.g. `format=f4f4`).
#
# Every format is a thin call into `aiter.ops.mha_v4.mha_v4`, which takes raw
# BF16 BSHD operands, applies the canonical quantizer for the requested
# (Q/K, V) format pair, and launches the matching ASM row.

from typing import TYPE_CHECKING

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.utils import is_gfx95_supported, is_gfx942_supported

if TYPE_CHECKING:
    from aiter.ops.mha_v4 import AttentionFormat

logger = init_logger(__name__)

# Every mha_v4 row targets full-MHA head_dim==128 models.
# Selecting this backend is explicit, so unmet constraints raise
# rather than silently falling back.
_REQUIRED_HEAD_DIM = 128

_DEFAULT_FORMAT = "fp8"

# aiter's gfx942 fmha_v4 manifest ships only the FP8 and INT8 Q/K rows,
# every MX row is gfx950-only.
_GFX942_FORMATS = frozenset({"fp8", "i8fp8"})

# Imported at module load so torch.compile sees stable symbols. If the installed
# aiter lacks mha_v4, the names resolve to None and construction raises a clear
# error.
try:
    from aiter.ops.mha_v4 import AttentionFormat as _AiterAttentionFormat
    from aiter.ops.mha_v4 import mha_v4 as _aiter_mha_v4
    from aiter.ops.mha_v4 import native_fp8_format as _aiter_native_fp8_format

    _AITER_MHA_V4_AVAILABLE = True
except ImportError:
    # Keep the names defined (as None) so they remain patchable and referencing
    # them yields a clear message via the construction-time availability check.
    _AiterAttentionFormat = None
    _aiter_mha_v4 = None
    _aiter_native_fp8_format = None

    _AITER_MHA_V4_AVAILABLE = False


def _resolve_format() -> str:
    """Read the `format` key from --attention-backend-config (default fp8)."""
    cfg = get_global_server_args().attention_backend_config or {}
    return str(cfg.get("format", _DEFAULT_FORMAT)).lower()


def _validate_arch(name: str) -> None:
    """Raise unless the active arch has an mha_v4 ASM row for this format."""
    if is_gfx95_supported():
        return
    if not is_gfx942_supported():
        raise RuntimeError(
            "AITER quant backend requires a gfx950- or gfx942-class arch."
        )
    if name not in _GFX942_FORMATS:
        raise NotImplementedError(
            f"aiter_quant format {name!r} has no gfx942 kernel row. gfx942 "
            f"supports only: {', '.join(sorted(_GFX942_FORMATS))}."
        )


def _resolve_format_pair(name: str) -> tuple["AttentionFormat", "AttentionFormat"]:
    """Map an aiter_quant format name to its (Q/K, V) mha_v4 format pair.

    mha_v4 requires Q and K to share a format, so only V varies. The pair is all
    aiter needs: it derives the quantizers, scale modes, and V packing from it.
    `native_fp8_format()` probes the active arch, so this runs on the worker
    rather than at import.
    """
    fp8 = _aiter_native_fp8_format()
    bf16 = _AiterAttentionFormat.BF16
    int8 = _AiterAttentionFormat.INT8
    mxfp4 = _AiterAttentionFormat.MXFP4
    mxfp6 = _AiterAttentionFormat.MXFP6
    # Grouped by Q/K format, which is what selects the recipe family.
    pairs = {
        "bf16fp8": (bf16, fp8),
        "fp8": (fp8, fp8),
        "f8f6": (fp8, mxfp6),
        "i8fp8": (int8, fp8),
        "mxfp6": (mxfp6, fp8),
        "f6f4": (mxfp6, mxfp4),
        "f4f4": (mxfp4, mxfp4),
    }
    if name not in pairs:
        raise ValueError(
            f"Unknown aiter_quant format {name!r}. Set "
            "--attention-backend-config format=<name> to one of: "
            f"{', '.join(sorted(pairs))}."
        )
    return pairs[name]


class AITERQuantBackend(AttentionBackend):
    """AITER quantized attention family backend (ROCm)."""

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.AITER_QUANT

    @staticmethod
    def get_impl_cls() -> type["AITERQuantImpl"]:
        return AITERQuantImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        # AITER quant backend does not require special metadata.
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError(
            "AITER quant backend does not have a metadata builder."
        )


class AITERQuantImpl(AttentionImpl):
    """Quantized attention via aiter's mha_v4, with the variant selected by the
    `format` key of --attention-backend-config (bf16fp8, fp8, f8f6, i8fp8,
    mxfp6, f6f4, f4f4)."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        if not _AITER_MHA_V4_AVAILABLE:
            raise RuntimeError(
                "AITER quant backend requires aiter.ops.mha_v4, which is not "
                "available in the installed aiter build."
            )
        if head_size != _REQUIRED_HEAD_DIM:
            raise NotImplementedError(
                f"AITER quant backend requires head_dim == {_REQUIRED_HEAD_DIM}, "
                f"got {head_size}."
            )
        if causal:
            raise NotImplementedError(
                "AITER quant backend does not support causal masking; mha_v4 has "
                "no causal ASM row."
            )

        self.format = _resolve_format()
        self.qk_format, self.v_format = _resolve_format_pair(self.format)
        _validate_arch(self.format)
        self.softmax_scale = softmax_scale

        # Deduped per message, so this logs once per format for the whole run.
        logger.info_once(f"aiter_quant attention backend using format={self.format}.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """
        Quantizes Q/K/V to the configured format and runs non-causal attention.

        Args:
            query: Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            key: Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
            value: Value tensor of shape [batch_size, seq_len, num_heads, head_dim]
            attn_metadata: Metadata for the attention operation (unused).

        Key/value may carry a different seq_len than query (cross-attention) and
        fewer heads than query (GQA, power-of-two ratios up to 16).

        Returns:
            Output tensor of shape [batch_size, seq_len, num_heads, head_dim]
        """
        return _aiter_mha_v4(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            self.qk_format,
            self.qk_format,
            self.v_format,
            softmax_scale=self.softmax_scale,
        )

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
        raise NotImplementedError(
            "AITER quant backend does not support varlen attention."
        )
