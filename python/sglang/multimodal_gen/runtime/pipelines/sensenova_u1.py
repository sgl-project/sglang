# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.configs.pipeline_configs.sensenova_u1 import (
    SenseNovaU1PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sensenova_u1 import (
    SenseNovaU1SamplingParams,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.models.sensenova_u1.loader import (
    load_model_and_tokenizer,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import InputValidationStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1 import (
    SenseNovaU1GenerationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class SenseNovaU1Pipeline(ComposedPipelineBase):
    pipeline_name = "SenseNovaU1Pipeline"
    pipeline_config_cls = SenseNovaU1PipelineConfig
    sampling_params_cls = SenseNovaU1SamplingParams
    _required_config_modules: list[str] = []

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError(
                "SenseNovaU1Pipeline only supports monolithic deployment; "
                f"disaggregation role {role.value!r} is not supported"
            )

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        if loaded_modules is not None and {"model", "tokenizer"} <= set(loaded_modules):
            return loaded_modules

        if server_args.num_gpus != 1:
            raise ValueError(
                "SenseNovaU1Pipeline currently supports num_gpus=1. "
                "Native tensor/pipeline parallelism is not implemented yet."
            )
        modules = load_model_and_tokenizer(self.model_path, server_args)
        logger.info("Loaded SenseNova-U1 model from %s", self.model_path)
        return modules

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        del server_args
        self.add_stage(InputValidationStage())
        self.add_stage(
            SenseNovaU1GenerationStage(
                model=self.get_module("model"),
                tokenizer=self.get_module("tokenizer"),
            ),
            "sensenova_u1_generation_stage",
        )


EntryClass = SenseNovaU1Pipeline
