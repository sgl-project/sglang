# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import torch

from sglang.multimodal_gen.runtime.layers.quantization.configs.nunchaku_config import (
    is_nunchaku_available,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import StoreBoolean

logger = init_logger(__name__)


@dataclass
class NunchakuSVDQuantArgs:
    """CLI-facing configuration for Nunchaku (SVDQuant) inference.

    This is intentionally lightweight and only contains arguments needed to
    construct `runtime.layers.quantization.nunchaku_config.NunchakuConfig`.
    """

    enable_svdquant: bool = False
    transformer_weights_path: str | None = None
    quantization_precision: str | None = None  # "int4" or "nvfp4"
    quantization_rank: int | None = None
    quantization_act_unsigned: bool = False

    def _adjust_config(self) -> None:
        """infer precision and rank from filename if not provided"""
        if self.transformer_weights_path and not self.enable_svdquant:
            filename = os.path.basename(self.transformer_weights_path)
            if re.search(r"svdq-(int4|fp4)_r(\d+)", filename):
                self.enable_svdquant = True

        if not self.enable_svdquant or not self.transformer_weights_path:
            return

        inferred_precision = None
        inferred_rank = None

        filename = os.path.basename(self.transformer_weights_path)
        # Expected pattern: svdq-{precision}_r{rank}-...
        # e.g., svdq-int4_r32-qwen-image.safetensors
        match = re.search(r"svdq-(int4|fp4)_r(\d+)", filename)

        if match:
            p_str, r_str = match.groups()
            inferred_precision = "nvfp4" if p_str == "fp4" else "int4"
            inferred_rank = int(r_str)

        if self.quantization_precision is None:
            self.quantization_precision = inferred_precision or "int4"
            if inferred_precision:
                logger.info(
                    f"inferred --quantization-precision: {self.quantization_precision} "
                    f"from --transformer-weights-path: {self.transformer_weights_path}"
                )

        if self.quantization_rank is None:
            self.quantization_rank = inferred_rank or 32
            if inferred_rank:
                logger.info(
                    f"inferred --quantization-rank: {self.quantization_rank} "
                    f"from --transformer-weights-path: {self.transformer_weights_path}"
                )

    def validate(self) -> None:
        # TODO: warn if the served model doesn't support nunchaku
        self._adjust_config()

        if not self.enable_svdquant:
            return

        if not current_platform.is_cuda():
            raise ValueError(
                "Nunchaku SVDQuant is only supported on NVIDIA CUDA GPUs "
                "(Ampere SM8x or SM12x)."
            )

        device_count = torch.cuda.device_count()

        unsupported: list[str] = []
        for i in range(device_count):
            major, minor = torch.cuda.get_device_capability(i)
            if major == 9:
                unsupported.append(f"cuda:{i} (SM{major}{minor}, Hopper)")
            elif major not in (8, 12):
                unsupported.append(f"cuda:{i} (SM{major}{minor})")

        if unsupported:
            raise ValueError(
                "Nunchaku SVDQuant is currently only supported on Ampere (SM8x) or SM12x GPUs; "
                "Hopper (SM90) is not supported. "
                f"Unsupported devices: {', '.join(unsupported)}. "
                "Disable it with --enable-svdquant false."
            )

        if not self.transformer_weights_path:
            raise ValueError(
                "--enable-svdquant requires --transformer-weights-path to be set"
            )

        if not is_nunchaku_available():
            raise ValueError(
                "Nunchaku is enabled, but not installed. Please refer to https://nunchaku.tech/docs/nunchaku/installation/installation.html for detailed installation methods."
            )

        if self.quantization_precision not in ("int4", "nvfp4"):
            raise ValueError(
                f"Invalid --quantization-precision: {self.quantization_precision}. "
                "Must be one of: int4, nvfp4"
            )

        if self.quantization_rank <= 0:
            raise ValueError(
                f"Invalid --quantization-rank: {self.quantization_rank}. Must be > 0"
            )

    @staticmethod
    def add_cli_args(parser) -> None:
        parser.add_argument(
            "--enable-svdquant",
            action=StoreBoolean,
            default=NunchakuSVDQuantArgs.enable_svdquant,
            help="Enable Nunchaku SVDQuant (W4A4-style) inference.",
        )
        parser.add_argument(
            "--transformer-weights-path",
            type=str,
            default=NunchakuSVDQuantArgs.transformer_weights_path,
            help=(
                "Path to pre-quantized transformer weights. Can be a single .safetensors "
                "file, a directory, or a HuggingFace repo ID. Used by Nunchaku (SVDQuant) and quantized single-file checkpoints."
            ),
        )
        parser.add_argument(
            "--quantization-precision",
            type=str,
            default=None,
            help="Quantization precision: int4 or nvfp4. If not specified, inferred from model path or defaults to int4.",
        )
        parser.add_argument(
            "--quantization-rank",
            type=int,
            default=None,
            help="SVD low-rank dimension (e.g., 32). If not specified, inferred from model path or defaults to 32.",
        )
        parser.add_argument(
            "--quantization-act-unsigned",
            action=StoreBoolean,
            default=NunchakuSVDQuantArgs.quantization_act_unsigned,
            help="Use unsigned activation quantization (if supported).",
        )

    @classmethod
    def from_dict(cls, kwargs: dict[str, Any]) -> "NunchakuSVDQuantArgs":
        # Map CLI/config keys to dataclass fields (keep backwards compatibility).
        path = (
            kwargs.get("transformer_weights_path")
            or kwargs.get("transformer_quantized_path")
            or kwargs.get("quantized_model_path")
        )
        return cls(
            enable_svdquant=bool(kwargs.get("enable_svdquant", cls.enable_svdquant)),
            transformer_weights_path=path,
            quantization_precision=kwargs.get("quantization_precision"),
            quantization_rank=kwargs.get("quantization_rank"),
            quantization_act_unsigned=bool(
                kwargs.get("quantization_act_unsigned", cls.quantization_act_unsigned)
            ),
        )
