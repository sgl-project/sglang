# SPDX-License-Identifier: Apache-2.0

from typing import Any

from sglang.multimodal_gen.runtime.pipelines_core import ComposedPipelineBase
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1 import (
    SenseNovaU1PixelFlowGSegmentExecutor,
    SenseNovaU1PixelFlowGSegmentStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class SenseNovaU1Pipeline(ComposedPipelineBase):
    """Stateless SenseNova U1 G segment generator.

    UG interleave orchestration is intentionally outside multimodal_gen. This
    pipeline owns only the model-private pixel-flow generation call.
    """

    pipeline_name = "SenseNovaU1Pipeline"
    _required_config_modules: list[str] = []

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del server_args
        modules = dict(loaded_modules or {})
        if "g_segment_executor" not in modules:
            modules["g_segment_executor"] = SenseNovaU1PixelFlowGSegmentExecutor()
        return modules

    def create_pipeline_stages(self, server_args: ServerArgs):
        del server_args
        self.add_stage(InputValidationStage())
        self.add_stage(
            SenseNovaU1PixelFlowGSegmentStage(
                self.get_module("g_segment_executor"),
                context_ops_key="sensenova_u1_context_ops",
                output_extra_key="sensenova_u1_generated_segment",
            )
        )


EntryClass = SenseNovaU1Pipeline
