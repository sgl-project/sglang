# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)


def _is_runtime_option_requested(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (dict, list, tuple, set)):
        return bool(value)
    return True


@dataclass
class SenseNovaU1PipelineConfig(PipelineConfig):
    """Native SenseNova-U1 text-to-image pipeline configuration."""

    task_type: ModelTaskType = ModelTaskType.T2I
    model_precision: str = "bf16"
    should_use_guidance: bool = True
    supports_cfg_parallel: bool = False

    def supports_dynamic_batching(self):
        return False

    def supports_disaggregation(self) -> bool:
        return False

    def supports_sequential_multi_output_inference(self):
        return True

    def validate_server_args(self, server_args) -> None:
        if server_args.num_gpus != 1:
            raise ValueError(
                "SenseNovaU1Pipeline currently supports num_gpus=1. "
                "Native tensor/pipeline parallelism is not implemented yet."
            )
        if getattr(server_args, "enable_torch_compile", False):
            raise ValueError(
                "SenseNovaU1Pipeline does not support torch.compile yet. "
                "Please omit --enable-torch-compile."
            )
        if getattr(server_args, "lora_path", None):
            raise ValueError(
                "SenseNovaU1Pipeline does not support LoRA adapters yet. "
                "Please omit --lora-path."
            )
        unsupported_runtime_options = {
            "component_residency": "component residency",
            "cpu_offload_components": "CPU offload",
            "dit_cpu_offload": "DiT CPU offload",
            "text_encoder_cpu_offload": "text encoder CPU offload",
            "image_encoder_cpu_offload": "image encoder CPU offload",
            "vae_cpu_offload": "VAE CPU offload",
            "dit_layerwise_offload": "DiT layerwise offload",
            "layerwise_offload_components": "layerwise offload",
            "quantization": "quantization",
            "transformer_weights_path": "pre-quantized transformer weights",
            "component_quantizations": "component quantization",
            "component_precisions": "component precision overrides",
        }
        for option, description in unsupported_runtime_options.items():
            if _is_runtime_option_requested(getattr(server_args, option, None)):
                raise ValueError(
                    f"SenseNovaU1Pipeline does not support {description} yet. "
                    f"Please omit --{option.replace('_', '-')}."
                )
        if getattr(server_args, "attention_backend", None) is not None:
            raise ValueError(
                "SenseNovaU1Pipeline does not support custom attention backends yet. "
                "Please omit --attention-backend."
            )
        if getattr(server_args, "component_attention_backends", None):
            raise ValueError(
                "SenseNovaU1Pipeline does not support component attention backends yet. "
                "Please omit --component-attention-backends."
            )
        if getattr(server_args, "attention_backend_config", None):
            raise ValueError(
                "SenseNovaU1Pipeline does not support attention backend config yet. "
                "Please omit --attention-backend-config."
            )

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            speed_mode_enable_torch_compile_by_default=False,
            keep_resident_min_available_gb=80,
            auto_enable_cfg_parallel=False,
            supports_cfg_parallel=False,
        )
