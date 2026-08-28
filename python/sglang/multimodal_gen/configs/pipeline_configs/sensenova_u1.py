# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)


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
