# SPDX-License-Identifier: Apache-2.0
#
# Quantized AITER attention family backend (ROCm / gfx950, gfx942).
#
# One backend, several quant formats selected via --attention-backend-config
# (e.g. `format=mxfp4`).
#
# Every format is a thin call into `aiter.ops.mha_v4.mha_v4`, which takes raw
# BF16 BSHD operands, applies the canonical quantizer for the requested
# (Q/K, V) format pair, and launches the matching ASM row.

import enum
from typing import TYPE_CHECKING, Any

import msgspec
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

# Imported at module load so torch.compile sees stable symbols. If the installed
# aiter lacks mha_v4, the names resolve to None and construction raises a clear
# error.
try:
    from aiter.ops.mha_v4 import AttentionFormat as _AiterAttentionFormat
    from aiter.ops.mha_v4 import AttentionScaleMode as _AiterAttentionScaleMode
    from aiter.ops.mha_v4 import mha_v4 as _aiter_mha_v4
    from aiter.ops.mha_v4 import native_fp8_format as _aiter_native_fp8_format

    _AITER_MHA_V4_AVAILABLE = True
except ImportError:
    # Keep the names defined (as None) so they remain patchable and referencing
    # them yields a clear message via the construction-time availability check.
    _AiterAttentionFormat = None
    _AiterAttentionScaleMode = None
    _aiter_mha_v4 = None
    _aiter_native_fp8_format = None

    _AITER_MHA_V4_AVAILABLE = False


class _Fmt(enum.Enum):
    """Operand encodings, named so the table below is plain data.

    Members are resolved against aiter's own `AttentionFormat` by name, so each
    one must match a member or alias of it. MXFP6 and MXFP4 are aliases (of
    FP6_E2M3 and FP4_E2M1) and do not appear when iterating that enum.

    NATIVE_FP8 is the exception: aiter exposes it as a function rather than a
    member, because the concrete type is architecture-dependent (FP8_E4M3 on
    gfx950 vs FP8_E4M3_FNUZ on gfx942).
    """

    BF16 = enum.auto()
    INT8 = enum.auto()
    NATIVE_FP8 = enum.auto()
    MXFP6 = enum.auto()
    MXFP4 = enum.auto()


class _QuantFormat(msgspec.Struct, frozen=True):
    name: str
    qk: _Fmt
    v: _Fmt
    # MXFP8 is the one row that needs block-scaled Q/K, which aiter expresses
    # through scale modes rather than through a format: FP8 operands carrying
    # E8M0 1x32 Q/K scales and a per-tensor V scale. Every other row takes
    # aiter's canonical scale modes for its format pair.
    block_scaled: bool = False
    # gfx942's fmha_v4 manifest ships only the FP8 and INT8 Q/K rows; the BF16
    # Q/K row and every MX row are gfx950-only.
    gfx942: bool = False


# fmt: off
# Hand-aligned: a row reads across and a column reads down, against the header.
_FORMATS = (
    #             name         Q/K              V
    _QuantFormat("bf16fp8",   _Fmt.BF16,       _Fmt.NATIVE_FP8),
    _QuantFormat("i8fp8",     _Fmt.INT8,       _Fmt.NATIVE_FP8, gfx942=True),
    _QuantFormat("fp8",       _Fmt.NATIVE_FP8, _Fmt.NATIVE_FP8, gfx942=True),
    _QuantFormat("mxfp8",     _Fmt.NATIVE_FP8, _Fmt.NATIVE_FP8, block_scaled=True),
    _QuantFormat("mxfp6",     _Fmt.MXFP6,      _Fmt.NATIVE_FP8),
    # MXFP4 Q/K with FP8 V is not a row aiter has; it rejects the combination.
    _QuantFormat("mxfp4",     _Fmt.MXFP4,      _Fmt.MXFP4),
)
# fmt: on

_FORMATS_BY_NAME = {fmt.name: fmt for fmt in _FORMATS}


def _resolve_format() -> str:
    """Read the `format` key from --attention-backend-config (default fp8)."""
    cfg = get_global_server_args().attention_backend_config or {}
    return str(cfg.get("format", _DEFAULT_FORMAT)).lower()


def _lookup_format(name: str) -> _QuantFormat:
    try:
        return _FORMATS_BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"Unknown aiter_quant format {name!r}. Set "
            "--attention-backend-config format=<name> to one of: "
            f"{', '.join(fmt.name for fmt in _FORMATS)}."
        ) from None


def _validate_arch(fmt: _QuantFormat) -> None:
    """Raise unless the active arch has an mha_v4 ASM row for this format."""
    if is_gfx95_supported():
        return
    if not is_gfx942_supported():
        raise RuntimeError(
            "AITER quant backend requires a gfx950- or gfx942-class arch."
        )
    if not fmt.gfx942:
        gfx942_names = ", ".join(f.name for f in _FORMATS if f.gfx942)
        raise NotImplementedError(
            f"aiter_quant format {fmt.name!r} has no gfx942 kernel row. gfx942 "
            f"supports only: {gfx942_names}."
        )


def _aiter_format(fmt: _Fmt) -> "AttentionFormat":
    """Translate one local format name to aiter's enum.

    `native_fp8_format()` probes the active arch, so this runs on the worker
    rather than at import.
    """
    if fmt is _Fmt.NATIVE_FP8:
        return _aiter_native_fp8_format()
    return getattr(_AiterAttentionFormat, fmt.name)


def _scale_mode_kwargs(fmt: _QuantFormat) -> dict[str, Any]:
    """The scale modes to pass, which is nothing unless the row is block-scaled.

    aiter derives the canonical modes from the format pair, and rejects any
    triple that is neither canonical nor the MXFP8 recipe, so passing them
    everywhere would only restate what it already knows.
    """
    if not fmt.block_scaled:
        return {}
    block = _AiterAttentionScaleMode.E8M0_PER_1X32
    return {
        "q_scale_mode": block,
        "k_scale_mode": block,
        "v_scale_mode": _AiterAttentionScaleMode.F32_PER_TENSOR,
    }


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
    `format` key of --attention-backend-config (bf16fp8, i8fp8, fp8, mxfp8,
    mxfp6, mxfp4; default fp8)."""

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
        fmt = _lookup_format(self.format)
        _validate_arch(fmt)
        # mha_v4 requires Q and K to share a format, so only V varies.
        self.qk_format = _aiter_format(fmt.qk)
        self.v_format = _aiter_format(fmt.v)
        self.scale_mode_kwargs = _scale_mode_kwargs(fmt)
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
            **self.scale_mode_kwargs,
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
