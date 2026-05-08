# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)


@dataclass
class SenseNovaU1PipelineConfig(PipelineConfig):
    task_type: ModelTaskType = ModelTaskType.TI2I
    enable_autocast: bool = True
    vae_tiling: bool = False
    vae_sp: bool = False

    default_height: int = 1024
    default_width: int = 1024

    def validate_runtime(
        self,
        *,
        num_gpus: int,
        enable_cfg_parallel: bool | None,
        disagg_mode: bool,
    ) -> list[str]:
        unsupported = []
        if num_gpus != 1:
            unsupported.append("num_gpus")
        if enable_cfg_parallel:
            unsupported.append("enable_cfg_parallel")
        if disagg_mode:
            unsupported.append("disagg_mode")
        return unsupported
