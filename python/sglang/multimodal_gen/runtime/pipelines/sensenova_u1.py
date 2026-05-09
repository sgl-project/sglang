# SPDX-License-Identifier: Apache-2.0
"""SenseNova U1 pixel-flow pipeline used as an omni generation backend."""

from typing import Any

from sglang.multimodal_gen.runtime.pipelines_core import ComposedPipelineBase
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1 import (
    SenseNovaU1PixelFlowStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class SenseNovaU1Pipeline(ComposedPipelineBase):
    """Stateless SenseNova U1 pixel-flow generation pipeline.

    Omni interleave orchestration is intentionally outside multimodal_gen. This
    pipeline owns only the model-private pixel-flow generation stage.
    """

    pipeline_name = "SenseNovaU1Pipeline"
    _required_config_modules: list[str] = []

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del server_args
        return dict(loaded_modules or {})

    def create_pipeline_stages(self, server_args: ServerArgs):
        del server_args
        self.add_stage(InputValidationStage())
        self.add_stage(
            SenseNovaU1PixelFlowStage(context_ops_key="sensenova_u1_context_ops")
        )


EntryClass = SenseNovaU1Pipeline
