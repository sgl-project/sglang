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

    def supports_disaggregation(self) -> bool:
        return False

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            speed_mode_enable_torch_compile_by_default=False,
            keep_resident_min_available_gb=80,
            auto_enable_cfg_parallel=False,
            supports_cfg_parallel=False,
        )
