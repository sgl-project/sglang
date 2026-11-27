# SPDX-License-Identifier: Apache-2.0

"""Motubrain2 robot action pipeline."""

from __future__ import annotations

from typing import Any

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.vla import (
    VLAActionDenoisingStage,
    VLAActionPostprocessStage,
    VLAObservationPreprocessStage,
    VLAPrefixEncodingStage,
    vla_state,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.vla.prefix_cache import VLAPrefixCacheManager

from ...configs.pipeline_configs.motubrain2 import Motubrain2PipelineConfig
from ...configs.sample.motubrain2 import Motubrain2SamplingParams
from ..models.vlas.motubrain2_policy import Motubrain2PolicyModel
from ..pipelines_core.stages.model_specific_stages.motubrain2_preprocess import (
    Motubrain2Preprocessor,
)

logger = init_logger(__name__)


class Motubrain2ActionPostprocessStage(VLAActionPostprocessStage):
    def forward(self, batch, server_args):
        state = vla_state(batch)
        observation = state.get("observation_batch")
        if observation is not None and "native_outputs" in observation.metadata:
            state.setdefault("extras", {})["native_outputs"] = observation.metadata.pop(
                "native_outputs"
            )
        return super().forward(batch, server_args)


class Motubrain2Pipeline(ComposedPipelineBase):
    """SGLang Diffusion pipeline for Motubrain2 robot policies."""

    pipeline_name = "Motubrain2Pipeline"
    pipeline_config_cls = Motubrain2PipelineConfig
    sampling_params_cls = Motubrain2SamplingParams
    _required_config_modules: list[str] = []

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError("Motubrain2Pipeline requires monolithic execution")

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if loaded_modules is not None:
            return loaded_modules
        config: Motubrain2PipelineConfig = server_args.pipeline_config
        model = Motubrain2PolicyModel.from_pretrained(self.model_path, config)
        logger.info("Loaded Motubrain2 policy backend: %r", model)
        return {"policy_model": model}

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        config: Motubrain2PipelineConfig = server_args.pipeline_config
        self.preprocessor = Motubrain2Preprocessor(config)
        self.prefix_cache = VLAPrefixCacheManager(
            max_entries=config.prefix_cache_max_entries
        )

    def release_action_session(self, session_id: str) -> bool:
        return self.get_module("policy_model").release_session(session_id)

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(
            VLAObservationPreprocessStage(self.preprocessor),
            "motubrain2_preprocess",
        )
        self.add_stage(
            VLAPrefixEncodingStage(
                self.get_module("policy_model"),
                self.prefix_cache,
            ),
            "motubrain2_prefix",
        )
        self.add_stage(
            VLAActionDenoisingStage(self.get_module("policy_model")),
            "motubrain2_action_denoise",
        )
        self.add_stage(
            Motubrain2ActionPostprocessStage(),
            "motubrain2_postprocess",
        )


EntryClass = Motubrain2Pipeline
