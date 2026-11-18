# SPDX-License-Identifier: Apache-2.0
"""MindIE-SD quantized attention adapter for Ascend FA.

The 4-bit/8-bit attention families (FP8, MXFP8, MXFP4) run through the
public ``mindiesd.quant_attention`` API; the float16/BF16 paths stay in
``ascend_fa.py``.
"""

from collections.abc import Callable
from numbers import Integral
from typing import ClassVar

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionMetadata,
    trailing_padding_used_len,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def load_mindiesd_quant_attention() -> Callable[..., torch.Tensor]:
    """Load the optional SDK and check its public quantized attention API."""
    try:
        from mindiesd import quant_attention
    except ImportError as error:
        raise ImportError(
            "Quantized Ascend FA requires a MindIE-SD build exposing the public "
            "quant_attention API and its native operators. Install a "
            "compatible MindIE-SD, torch_npu and CANN build."
        ) from error
    if not callable(quant_attention):
        raise ImportError("mindiesd.quant_attention must be callable")
    return quant_attention


class AscendQuantAttention:
    """Internal MindIE-SD FP8/MXFP8/MXFP4 implementation for Ascend FA."""

    _rot_matrices: ClassVar[dict[int, torch.Tensor]] = {}

    def __init__(
        self,
        config: dict,
        *,
        head_size: int,
        causal: bool,
        softmax_scale: float | None,
        offline_qk_rotation: bool | None = None,
        **extra_impl_args,
    ) -> None:
        if causal:
            raise ValueError("Quantized Ascend FA does not support causal attention")
        if extra_impl_args.get("dropout_p", 0.0):
            raise ValueError("Quantized Ascend FA does not support attention dropout")
        self.precision = config["precision"]
        if not isinstance(self.precision, str) or not self.precision:
            raise ValueError("precision must be a non-empty string")
        # SD owns supported precisions, mode normalization and mode/head-size
        # constraints. Only reject options that this adapter would not forward.
        self.fp8_fa_mode = None
        if self.precision == "fp8":
            self.fp8_fa_mode = config.get("fp8_fa_mode", "HIGH_PRECISION")
        elif "fp8_fa_mode" in config:
            raise ValueError("fp8_fa_mode is only supported with precision=fp8")
        self.head_chunk_size = config.get("head_chunk_size", 4)
        if (
            isinstance(self.head_chunk_size, bool)
            or not isinstance(self.head_chunk_size, Integral)
            or self.head_chunk_size < 0
        ):
            raise ValueError("head_chunk_size must be a non-negative integer")
        quant_config = extra_impl_args.get("quant_config")
        offline_rotation = (
            getattr(quant_config, "use_offline_qk_rotation", False)
            if offline_qk_rotation is None
            else offline_qk_rotation
        )
        self.qk_rotation = config.get("qk_rotation", not offline_rotation)
        if not isinstance(self.qk_rotation, bool):
            raise ValueError("qk_rotation must be a boolean")
        if offline_rotation and self.qk_rotation:
            raise ValueError("qk_rotation=true conflicts with offline Q/K rotation")

        self.head_size = head_size
        self.softmax_scale = head_size**-0.5 if softmax_scale is None else softmax_scale
        self.packed_trailing_padding = bool(
            extra_impl_args.get("packed_trailing_padding", False)
        )
        self._rot_device: torch.Tensor | None = None
        if self.qk_rotation:
            self._ensure_rot_matrix(head_size)
        self._quant_attention = load_mindiesd_quant_attention()
        logger.info_once(
            f"Using Ascend FA: provider=mindiesd, precision={self.precision}, "
            f"fp8_fa_mode={self.fp8_fa_mode}, qk_rotation={self.qk_rotation}, "
            f"head_chunk_size={self.head_chunk_size}"
        )

    @classmethod
    def _ensure_rot_matrix(cls, head_size: int) -> None:
        if head_size not in cls._rot_matrices:
            # Match AscendFAImpl without changing its implementation or global RNG.
            generator = torch.Generator(device="cpu").manual_seed(42)
            rotation, _ = torch.linalg.qr(
                torch.randn(
                    head_size,
                    head_size,
                    generator=generator,
                    device="cpu",
                    dtype=torch.float32,
                )
            )
            cls._rot_matrices[head_size] = rotation

    def _get_rotation(self, query: torch.Tensor) -> torch.Tensor | None:
        if not self.qk_rotation:
            return None
        if (
            self._rot_device is None
            or self._rot_device.device != query.device
            or self._rot_device.dtype != query.dtype
        ):
            self._rot_device = self._rot_matrices[self.head_size].to(
                device=query.device, dtype=query.dtype
            )
        return self._rot_device

    def _validate_adapter_inputs(self, query, key, value, *, ndim: int) -> None:
        # Validate only the adapter's layout/configuration and slicing contract.
        # SD owns dtype, device, GQA and K/V shape validation for each native call.
        layout = "BSND" if ndim == 4 else "TND"
        if any(t.ndim != ndim for t in (query, key, value)):
            raise ValueError(f"MindIE-SD attention requires {layout} Q/K/V")
        if query.shape[-1] != self.head_size:
            raise ValueError("Query head dimension must match configured head_size")
        # Check before slicing: extra K/V batches or tokens would otherwise be
        # discarded, hiding the mismatch from the public SD entry point.
        if any(t.shape[0] != query.shape[0] for t in (key, value)):
            raise ValueError("Q/K/V batch sizes or packed token counts must match")

    def _forward_dense(self, query, key, value) -> torch.Tensor:
        # Match native Ascend FA's MHA head chunking. Preserve SDK handling of
        # GQA and invalid K/V head counts instead of slicing away mismatches.
        if (
            0 < self.head_chunk_size < query.shape[2]
            and query.shape[2] == key.shape[2] == value.shape[2]
        ):
            return torch.cat(
                [
                    self._run_attention(q, k, v)
                    for q, k, v in zip(
                        query.split(self.head_chunk_size, dim=2),
                        key.split(self.head_chunk_size, dim=2),
                        value.split(self.head_chunk_size, dim=2),
                    )
                ],
                dim=2,
            )
        return self._run_attention(query, key, value)

    def _run_attention(self, query, key, value) -> torch.Tensor:
        rotation = self._get_rotation(query)
        options = {}
        if self.precision == "fp8":
            options["fp8_fa_mode"] = self.fp8_fa_mode
        # SD applies rotation exactly once, before quantization. Preserve its
        # output dtype, including the current MXFP4 kernel's BF16 output.
        return self._quant_attention(
            query,
            key,
            value,
            precision=self.precision,
            layout="BSND",
            scale=self.softmax_scale,
            q_rot=rotation,
            k_rot=rotation,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            **options,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        *,
        return_softmax_lse: bool = False,
    ) -> torch.Tensor:
        if return_softmax_lse:
            raise NotImplementedError("MindIE-SD attention does not return softmax LSE")
        self._validate_adapter_inputs(query, key, value, ndim=4)
        # FP8 requires B=1. MXFP8 groups V over flattened B*S, so isolate
        # batches unless their sequence boundaries are aligned to 64 tokens.
        split_batch = self.precision == "fp8" or (
            self.precision == "mxfp8" and key.shape[1] % 64 != 0
        )
        # Send zero-batch inputs to SD validation instead of torch.cat([]).
        if not split_batch or query.shape[0] <= 1:
            return self._forward_dense(query, key, value)
        return torch.cat(
            [
                self._forward_dense(query[i : i + 1], key[i : i + 1], value[i : i + 1])
                for i in range(query.shape[0])
            ],
            dim=0,
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
        self._validate_adapter_inputs(query, key, value, ndim=3)
        if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("cu_seqlens must be a 1D int32 or int64 tensor")
        bounds = (
            tuple(cu_seqlens.tolist()) if cu_seqlens_host is None else cu_seqlens_host
        )
        if (
            len(bounds) < 2
            or len(bounds) != cu_seqlens.numel()
            or any(isinstance(x, bool) or not isinstance(x, Integral) for x in bounds)
            or bounds[0] != 0
            or bounds[-1] != query.shape[0]
            or any(start > stop for start, stop in zip(bounds[:-1], bounds[1:]))
        ):
            raise ValueError("cu_seqlens must monotonically cover all packed tokens")
        if (
            isinstance(max_seqlen, bool)
            or not isinstance(max_seqlen, Integral)
            or max_seqlen < 0
        ):
            raise ValueError("max_seqlen must be a non-negative integer")
        if self.packed_trailing_padding:
            used = trailing_padding_used_len(query.shape[0], max_seqlen, bounds)
            if used is not None:
                bounds = (0, used)
        if any(
            stop - start > max_seqlen for start, stop in zip(bounds[:-1], bounds[1:])
        ):
            raise ValueError("Packed sequence length exceeds max_seqlen")

        output = None
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if start == stop:
                continue
            segment = self._forward_dense(
                query[start:stop].unsqueeze(0),
                key[start:stop].unsqueeze(0),
                value[start:stop].unsqueeze(0),
            )
            if output is None:
                output = segment.new_zeros(query.shape)
            output[start:stop].copy_(segment[0])
        return torch.zeros_like(query) if output is None else output
