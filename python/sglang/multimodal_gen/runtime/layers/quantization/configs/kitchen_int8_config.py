# SPDX-License-Identifier: Apache-2.0
"""Config for online INT8 ConvRot quantization via comfy_kitchen.

A no-arg ``KitchenInt8Config()`` is the only supported form: weights load in
their source dtype and are quantized in ``process_weights_after_loading``.

Registered CLI name: ``kitchen_int8``.
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

_SUPPORTED_GROUP_SIZES = (16, 64, 256)


class KitchenInt8Config(QuantizationConfig):
    """Config for online INT8 ConvRot quantization via comfy_kitchen.

    A no-arg ``KitchenInt8Config()`` is the only supported form: weights load in
    their source dtype and are quantized in ``process_weights_after_loading``.
    """

    def __init__(
        self,
        group_size: int = 256,
        ignored_layers: list[str] | None = None,
        packed_modules_mapping: dict[str, list[str]] | None = None,
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
        return KitchenInt8LinearMethod(self)

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
