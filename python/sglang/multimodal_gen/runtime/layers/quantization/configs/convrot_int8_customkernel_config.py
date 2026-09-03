# SPDX-License-Identifier: Apache-2.0
"""Config for online INT8 ConvRot quantization via sgl-kernel's fused ops.

Quality class: group-wise (default 256) Hadamard rotation followed by per-row
dynamic INT8 quantization of both activations and weights (ConvRot,
arXiv:2512.03673). Same-seed outputs are visually equivalent to BF16 but not
bit-exact; this is not a consistency ground-truth mode.

The rotation is data-free, so weights load in their source dtype from a stock
BF16 checkpoint and are rotated and quantized in
``process_weights_after_loading``. A no-arg ``ConvRotInt8CustomKernelConfig()``
is the only supported form; there is no serialized checkpoint format.

Recommended ``--quantization-ignored-layers`` for Qwen-Image:
``img_mod txt_mod txt_mlp.net.2``. Those GEMMs see only a handful of rows
(the AdaLN modulation runs on one row per batch element, the text-stream
FFN down-projection on a few dozen), so the rotate-quantize launch plus a
CTA-starved INT8 GEMM measured slower than the BF16 GEMM they replace.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.layers.quantization.utils import is_layer_skipped

logger = init_logger(__name__)

# Group widths the rotate-quantize kernel is instantiated for.
_SUPPORTED_GROUP_SIZES = (64, 128, 256, 512)


# Parts the ops deliberately leave out, with the reason shown at load time. The
# supported table itself lives in sgl-kernel (convrot_int8_supported_sm_versions);
# nothing here duplicates it.
REFUSED_CAPABILITY_REASONS: dict[tuple[int, int], str] = {
    (10, 3): (
        "Blackwell Ultra cuts INT8 tensor-core throughput to a fraction of its "
        "BF16 rate, so W8A8 INT8 would be a slowdown there; use FP8 or NVFP4 "
        "quantization on this GPU"
    ),
}


def convrot_int8_supported_capabilities() -> frozenset[tuple[int, int]]:
    """(major, minor) pairs the installed sgl_kernel carries convrot code for."""
    import sgl_kernel  # noqa: F401 -- registers torch.ops.sgl_kernel.*

    versions = torch.ops.sgl_kernel.convrot_int8_supported_sm_versions()
    return frozenset((int(v) // 10, int(v) % 10) for v in versions)


def check_convrot_int8_capability(capability: tuple[int, int]) -> None:
    """Raise with the specific reason when `capability` is not in the kernel's table."""
    supported = convrot_int8_supported_capabilities()
    if capability in supported:
        return
    major, minor = capability
    reason = REFUSED_CAPABILITY_REASONS.get(
        capability, "the convrot_int8_* ops carry no code for it"
    )
    supported_text = ", ".join(f"{a}.{b}" for a, b in sorted(supported))
    raise RuntimeError(
        f"convrot_int8_customkernel does not support CC {major}.{minor}: {reason}. "
        f"Supported compute capabilities: {supported_text}"
    )


class ConvRotInt8CustomKernelConfig(QuantizationConfig):
    """Online ConvRot INT8 for every divisible linear not listed as ignored."""

    def __init__(
        self,
        group_size: int = 256,
        ignored_layers: list[str] | None = None,
        packed_modules_mapping: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__()
        if group_size not in _SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"convrot_int8_customkernel group_size must be one of "
                f"{_SUPPORTED_GROUP_SIZES}, got {group_size}"
            )
        self.group_size = group_size
        self.ignored_layers = ignored_layers or []
        self.packed_modules_mapping = packed_modules_mapping or {}
        # Which layers actually got quantized is worth stating plainly in the
        # log: a silent fallback to BF16 looks exactly like a slow kernel.
        self.selected: list[str] = []
        self.skipped: list[str] = []
        self._processed = 0
        self._quantized_bytes = 0

    @classmethod
    def get_name(cls) -> str:
        return "convrot_int8_customkernel"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Lowest entry of sgl-kernel's supported table (see
        # convrot_int8_supported_capabilities); the exact check runs at load
        # time in check_convrot_int8_capability.
        return 90

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ConvRotInt8CustomKernelConfig:
        return cls(
            group_size=cls.get_from_keys_or(config, ["group_size"], 256),
            ignored_layers=cls.get_from_keys_or(config, ["ignored_layers"], None),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase
        from sglang.multimodal_gen.runtime.layers.quantization.convrot_int8_customkernel import (
            ConvRotInt8CustomKernelLinearMethod,
        )

        if not isinstance(layer, LinearBase):
            return None
        if is_layer_skipped(
            prefix, self.ignored_layers, fused_mapping=self.packed_modules_mapping
        ):
            self.skipped.append(prefix)
            return UnquantizedLinearMethod()
        # The rotation partitions the input dim into fixed-size groups, so a
        # layer whose input does not divide evenly simply stays in BF16 rather
        # than failing the whole model.
        if layer.input_size % self.group_size:
            self.skipped.append(f"{prefix}(in={layer.input_size})")
            return UnquantizedLinearMethod()
        self.selected.append(prefix)
        return ConvRotInt8CustomKernelLinearMethod(self)

    def note_quantized(self, saved_bytes: int) -> None:
        self._processed += 1
        self._quantized_bytes += saved_bytes
        if self._processed == len(self.selected):
            logger.info(
                "convrot_int8_customkernel: quantized %d linear layers (%.2f GiB "
                "of BF16 weights -> %.2f GiB INT8), left %d in BF16",
                self._processed,
                self._quantized_bytes / 1024**3,
                self._quantized_bytes / 2 / 1024**3,
                len(self.skipped),
            )
            logger.debug(
                "convrot_int8_customkernel: layers left in BF16: %s", self.skipped
            )

    def get_scaled_act_names(self) -> list[str]:
        return []

    def supports_input_partition(
        self, prefix: str, input_size_per_partition: int
    ) -> bool:
        return input_size_per_partition % self.group_size == 0
