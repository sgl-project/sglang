# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang.multimodal_gen.utils import StoreBoolean


@dataclass
class NunchakuSVDQuantArgs:
    """CLI-facing configuration for Nunchaku (SVDQuant) inference.

    This is intentionally lightweight and only contains arguments needed to
    construct `runtime.layers.quantization.nunchaku_config.NunchakuConfig`.
    """

    enable_svdquant: bool = False
    quantized_model_path: str | None = None
    quantization_precision: str = "int4"  # "int4" or "nvfp4"
    quantization_rank: int = 32
    quantization_act_unsigned: bool = False

    def validate(self) -> None:
        if not self.enable_svdquant:
            return

        if not self.quantized_model_path:
            raise ValueError(
                "--enable-svdquant requires --quantized-model-path to be set"
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
            "--quantized-model-path",
            type=str,
            default=NunchakuSVDQuantArgs.quantized_model_path,
            help=(
                "Path to pre-quantized Nunchaku weights. Can be a single .safetensors "
                "file or a directory containing .safetensors."
            ),
        )
        parser.add_argument(
            "--quantization-precision",
            type=str,
            default=NunchakuSVDQuantArgs.quantization_precision,
            help="Quantization precision: int4 or nvfp4.",
        )
        parser.add_argument(
            "--quantization-rank",
            type=int,
            default=NunchakuSVDQuantArgs.quantization_rank,
            help="SVD low-rank dimension (e.g., 32).",
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
        return cls(
            enable_svdquant=bool(kwargs.get("enable_svdquant", cls.enable_svdquant)),
            quantized_model_path=kwargs.get("quantized_model_path", cls.quantized_model_path),
            quantization_precision=str(
                kwargs.get("quantization_precision", cls.quantization_precision)
            ),
            quantization_rank=int(kwargs.get("quantization_rank", cls.quantization_rank)),
            quantization_act_unsigned=bool(
                kwargs.get("quantization_act_unsigned", cls.quantization_act_unsigned)
            ),
        )

