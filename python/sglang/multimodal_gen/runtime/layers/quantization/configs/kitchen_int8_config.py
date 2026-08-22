# SPDX-License-Identifier: Apache-2.0
"""Config for online or serialized INT8 ConvRot via comfy_kitchen."""

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

_SUPPORTED_GROUP_SIZES = (16, 64, 256)


class KitchenInt8Config(QuantizationConfig):
    """Dispatch online quantization or serialized Comfy ConvRot layers."""

    def __init__(
        self,
        group_size: int = 256,
        ignored_layers: list[str] | None = None,
        packed_modules_mapping: dict[str, list[str]] | None = None,
        layer_markers: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        if group_size not in _SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"kitchen_int8 group_size must be one of {_SUPPORTED_GROUP_SIZES}, "
                f"got {group_size}"
            )
        self.group_size = group_size
        self.ignored_layers = ignored_layers or []
        self.packed_modules_mapping = packed_modules_mapping or {}
        self.layer_markers = layer_markers
        self.is_checkpoint_int8_serialized = layer_markers is not None
        self.checkpoint_uses_native_qkv_layout = self.is_checkpoint_int8_serialized
        self._serialized_group_sizes: dict[str, int] = {}
        if layer_markers is not None:
            for prefix, marker in layer_markers.items():
                if marker.get("format") != "int8_tensorwise":
                    raise ValueError(
                        f"Unsupported Comfy INT8 format for {prefix!r}: "
                        f"{marker.get('format')!r}"
                    )
                if marker.get("convrot") is not True:
                    raise ValueError(
                        f"Serialized kitchen_int8 layer {prefix!r} must set "
                        "convrot=true"
                    )
                marker_group_size = marker.get("convrot_groupsize")
                if marker_group_size not in _SUPPORTED_GROUP_SIZES:
                    raise ValueError(
                        f"Serialized kitchen_int8 layer {prefix!r} must declare "
                        f"convrot_groupsize in {_SUPPORTED_GROUP_SIZES}, got "
                        f"{marker_group_size!r}"
                    )
                self._serialized_group_sizes[prefix] = marker_group_size
        # Which layers actually got quantized is worth stating plainly in the
        # log: a silent fallback to BF16 looks exactly like a slow kernel.
        self.selected: list[str] = []
        self.skipped: list[str] = []
        self._processed = 0
        self._quantized_bytes = 0

    @classmethod
    def get_name(cls) -> str:
        return "kitchen_int8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # INT8 tensor cores land on Turing.
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> KitchenInt8Config:
        return cls(
            group_size=cls.get_from_keys_or(config, ["group_size"], 256),
            ignored_layers=cls.get_from_keys_or(config, ["ignored_layers"], None),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase
        from sglang.multimodal_gen.runtime.layers.quantization.kitchen_int8 import (
            KitchenInt8LinearMethod,
        )

        if not isinstance(layer, LinearBase):
            return None
        if self.layer_markers is not None:
            marker_group_size = self._serialized_group_sizes.get(prefix)
            if marker_group_size is None:
                return UnquantizedLinearMethod()
            if layer.input_size % marker_group_size:
                raise ValueError(
                    f"Serialized kitchen_int8 layer {prefix!r} has input size "
                    f"{layer.input_size}, which is not divisible by its "
                    f"ConvRot group size {marker_group_size}"
                )
            self.selected.append(prefix)
            return KitchenInt8LinearMethod(
                self,
                group_size=marker_group_size,
                is_checkpoint_serialized=True,
            )
        if is_layer_skipped(
            prefix, self.ignored_layers, fused_mapping=self.packed_modules_mapping
        ):
            self.skipped.append(prefix)
            return UnquantizedLinearMethod()
        # The rotation partitions the input dim into fixed-size groups, so a
        # layer whose input does not divide evenly simply stays in BF16 rather
        # than failing the whole model. H3's adaln projections (in=2688) are
        # the case this exists for, and they cost 0.2% of a step anyway.
        if layer.input_size % self.group_size:
            self.skipped.append(f"{prefix}(in={layer.input_size})")
            return UnquantizedLinearMethod()
        self.selected.append(prefix)
        return KitchenInt8LinearMethod(
            self,
            group_size=self.group_size,
            is_checkpoint_serialized=False,
        )

    def note_quantized(self, saved_bytes: int) -> None:
        self._processed += 1
        self._quantized_bytes += saved_bytes
        if self._processed == len(self.selected):
            logger.info(
                "kitchen_int8: quantized %d linear layers (%.2f GiB of BF16 weights "
                "-> %.2f GiB INT8), left %d in BF16",
                self._processed,
                self._quantized_bytes / 1024**3,
                self._quantized_bytes / 2 / 1024**3,
                len(self.skipped),
            )
            logger.debug("kitchen_int8: layers left in BF16: %s", self.skipped)

    def get_scaled_act_names(self) -> list[str]:
        return []

    def supports_input_partition(
        self, prefix: str, input_size_per_partition: int
    ) -> bool:
        group_size = self.group_size
        if self.layer_markers is not None:
            marker_group_size = self._serialized_group_sizes.get(prefix)
            if marker_group_size is None:
                return True
            group_size = marker_group_size
        return input_size_per_partition % group_size == 0
