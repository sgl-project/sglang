# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from ..weight_layout import WeightLayoutPlan
from ..weight_runtime_manifest import WeightManifestError


@dataclass(frozen=True)
class WeightAdapterContext:
    config: Any


class WeightLayoutAdapter(ABC):
    def validate(self, model, context: WeightAdapterContext) -> None:
        if getattr(model, "quant_config", None) is not None:
            raise WeightManifestError("model adapters support only unquantized weights")
        if torch.version.hip is not None or any(
            p.device.type != "cuda" for p in model.parameters()
        ):
            raise WeightManifestError(
                "model adapters currently support NVIDIA CUDA storage only"
            )
        for name, param in model.named_parameters():
            if param.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise WeightManifestError(
                    f"unsupported weight dtype: {name}: {param.dtype}"
                )

    @abstractmethod
    def build_layout(self, model, context: WeightAdapterContext) -> WeightLayoutPlan:
        """Describe final storage without reading or modifying weight values."""

    def finalize_target(self, model, context: WeightAdapterContext) -> None:
        from sglang.srt.model_loader.loader import PreshardedModelLoader

        PreshardedModelLoader._rebind_parameter_aliases(model)
