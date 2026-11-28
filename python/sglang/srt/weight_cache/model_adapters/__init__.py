# SPDX-License-Identifier: Apache-2.0
"""Explicit model registrations: subclasses never inherit reshard support."""

from importlib import import_module

from ..weight_runtime_manifest import WeightManifestError
from .base import WeightAdapterContext, WeightLayoutAdapter

_ADAPTERS = {
    "Qwen3ForCausalLM": ("qwen3", "Qwen3WeightAdapter"),
    "DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2WeightAdapter"),
}


def validate_adapter_config(config):
    architectures = getattr(config, "architectures", ()) or ()
    if not architectures or architectures[0] not in _ADAPTERS:
        raise WeightManifestError(
            f"no weight-layout adapter registered for {architectures}"
        )
    if getattr(config, "quantization_config", None):
        raise WeightManifestError(
            "weight-layout adapters do not support quantized checkpoints"
        )
    if architectures[0] == "DeepseekV2ForCausalLM" and (
        getattr(config, "q_lora_rank", None) is not None
        or getattr(config, "num_nextn_predict_layers", 0)
    ):
        raise WeightManifestError(
            "DeepSeek adapter currently supports the V2-Lite MLA layout without NextN"
        )


def get_weight_layout_adapter(
    model, context: WeightAdapterContext
) -> WeightLayoutAdapter:
    validate_adapter_config(context.config)
    architecture = context.config.architectures[0]
    module_name, adapter_name = _ADAPTERS[architecture]
    model_type = getattr(
        import_module(f"sglang.srt.models.{module_name}"), architecture
    )
    if type(model) is not model_type:
        raise WeightManifestError(
            f"weight-layout adapter requires exact type {model_type.__name__}, got {type(model).__name__}"
        )
    adapter = getattr(import_module(f"{__name__}.{module_name}"), adapter_name)()
    adapter.validate(model, context)
    return adapter
