# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from typing import Any

import torch

from sglang.multimodal_gen.runtime.layers.quantization.configs.nunchaku_config import (
    NunchakuConfig,
    is_nunchaku_available,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import StoreBoolean

logger = init_logger(__name__)


@dataclass
class NunchakuArgsResolution:
    """Normalized runtime settings derived from Nunchaku CLI-facing args."""

    transformer_weights_path: str | None = None
    nunchaku_config: NunchakuConfig | None = None


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

    def _infer_from_weights_path(self) -> tuple[bool, str | None, int | None]:
        """Infer whether SVDQuant is enabled and parse precision/rank from filename."""
        inferred_precision = None
        inferred_rank = None
        enable_svdquant = self.enable_svdquant

        if not self.transformer_weights_path:
            return enable_svdquant, inferred_precision, inferred_rank

        filename = os.path.basename(self.transformer_weights_path)
        if not enable_svdquant and re.search(r"svdq-(int4|fp4)_r(\d+)", filename):
            enable_svdquant = True

        if not enable_svdquant:
            return enable_svdquant, inferred_precision, inferred_rank

        # Expected pattern: svdq-{precision}_r{rank}-...
        # e.g., svdq-int4_r32-qwen-image.safetensors
        match = re.search(r"svdq-(int4|fp4)_r(\d+)", filename)

        if match:
            p_str, r_str = match.groups()
            inferred_precision = "nvfp4" if p_str == "fp4" else "int4"
            inferred_rank = int(r_str)

        return enable_svdquant, inferred_precision, inferred_rank

    def _normalized(self) -> "NunchakuSVDQuantArgs":
        enable_svdquant, inferred_precision, inferred_rank = (
            self._infer_from_weights_path()
        )
        normalized = replace(
            self,
            enable_svdquant=enable_svdquant,
            quantization_precision=(
                self.quantization_precision or inferred_precision or "int4"
            ),
            quantization_rank=self.quantization_rank or inferred_rank or 32,
        )

        if self.quantization_precision is None and inferred_precision:
            if inferred_precision:
                logger.info(
                    f"inferred --quantization-precision: {normalized.quantization_precision} "
                    f"from --transformer-weights-path: {self.transformer_weights_path}"
                )

        if self.quantization_rank is None and inferred_rank:
            if inferred_rank:
                logger.info(
                    f"inferred --quantization-rank: {normalized.quantization_rank} "
                    f"from --transformer-weights-path: {self.transformer_weights_path}"
                )

        return normalized

    def _validate(self) -> None:
        # TODO: warn if the served model doesn't support nunchaku
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

    def resolve_runtime_config(self) -> NunchakuArgsResolution:
        normalized = self._normalized()
        normalized._validate()

        if not normalized.enable_svdquant or not normalized.transformer_weights_path:
            return NunchakuArgsResolution(
                transformer_weights_path=normalized.transformer_weights_path,
                nunchaku_config=None,
            )

        return NunchakuArgsResolution(
            transformer_weights_path=normalized.transformer_weights_path,
            nunchaku_config=NunchakuConfig(
                precision=normalized.quantization_precision,
                rank=normalized.quantization_rank,
                act_unsigned=normalized.quantization_act_unsigned,
                transformer_weights_path=normalized.transformer_weights_path,
            ),
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
