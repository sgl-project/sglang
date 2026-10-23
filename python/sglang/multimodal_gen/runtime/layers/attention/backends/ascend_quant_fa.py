# SPDX-License-Identifier: Apache-2.0
"""Ascend quantized attention through the public MindIE-SD API."""

from __future__ import annotations

from numbers import Integral
from typing import ClassVar

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    trailing_padding_used_len,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.platforms.npu import load_mindiesd_quant_attention
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class AscendQuantFABackend(AttentionBackend):
    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.ASCEND_QUANT_FA

    @staticmethod
    def get_impl_cls() -> type[AscendQuantFAImpl]:
        return AscendQuantFAImpl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls():
        return None


class AscendQuantFAImpl(AttentionImpl):
    _rot_matrices: ClassVar[dict[int, torch.Tensor]] = {}

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float | None,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        from sglang.multimodal_gen.runtime.server_args import get_global_server_args

        if causal:
            raise ValueError("ascend_quant_fa does not support causal attention")
        if extra_impl_args.get("dropout_p", 0.0):
            raise ValueError("ascend_quant_fa does not support attention dropout")
        config = get_global_server_args().attention_backend_config or {}
        self.precision = config.get("precision", "fp8")
        # SD owns supported precisions, mode normalization and mode/head-size
        # constraints. Only reject options that this adapter would not forward.
        self.fp8_fa_mode = None
        if self.precision == "fp8":
            self.fp8_fa_mode = config.get("fp8_fa_mode", "HIGH_PRECISION")
        elif "fp8_fa_mode" in config:
            raise ValueError("fp8_fa_mode is only supported with precision=fp8")
        self.qk_rotation = config.get("qk_rotation", True)
        if not isinstance(self.qk_rotation, bool):
            raise ValueError("qk_rotation must be a boolean")

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
            f"Using MindIE-SD ascend_quant_fa: precision={self.precision}, "
            f"fp8_fa_mode={self.fp8_fa_mode}, qk_rotation={self.qk_rotation}"
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
            raise ValueError(f"ascend_quant_fa requires {layout} Q/K/V")
        if query.shape[-1] != self.head_size:
            raise ValueError("Query head dimension must match configured head_size")
        # Check before slicing: extra K/V batches or tokens would otherwise be
        # discarded, hiding the mismatch from the public SD entry point.
        if any(t.shape[0] != query.shape[0] for t in (key, value)):
            raise ValueError("Q/K/V batch sizes or packed token counts must match")

    def _forward_dense(self, query, key, value) -> torch.Tensor:
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
            raise NotImplementedError("ascend_quant_fa does not return softmax LSE")
        self._validate_adapter_inputs(query, key, value, ndim=4)
        # Send zero-batch inputs to SD validation instead of torch.cat([]).
        if self.precision != "fp8" or query.shape[0] <= 1:
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
