# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sglang.multimodal_gen.configs.pipeline_configs.dm05 import DM05PipelineConfig
from sglang.multimodal_gen.configs.sample.dm05 import DM05SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.models.vlas.dm05_policy import DM05PolicyModel
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    RequestValidationError,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.dm05_preprocess import (
    DM05Preprocessor,
    collate_dm05_observations,
    dm05_observation_signature,
    postprocess_dm05_actions,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.vla import (
    VLAActionDenoisingStage,
    VLAActionPostprocessStage,
    VLAObservationPreprocessStage,
    VLAPrefixEncodingStage,
    vla_state,
)
from sglang.multimodal_gen.runtime.vla.prefix_cache import VLAPrefixCacheManager


class DM05ObservationPreprocessStage(VLAObservationPreprocessStage):
    def __init__(self, preprocessor, policy_model):
        super().__init__(preprocessor)
        self.policy_model = policy_model

    def forward(self, batch, server_args):
        if batch.is_warmup:
            return batch
        try:
            batch = super().forward(batch, server_args)
        except ValueError as exc:
            raise RequestValidationError(str(exc)) from exc
        observation = vla_state(batch)["observation_batch"]
        if observation.noise is None and not batch.is_warmup:
            InputValidationStage()._generate_seeds(batch, server_args)
            if len(batch.generator) != observation.batch_size:
                raise ValueError("DM05 requires one seed per observation")
            observation.noise = torch.cat(
                [
                    self.policy_model.sample_noise(1, generator=generator)
                    for generator in batch.generator
                ]
            )
            # Noise is already request-owned; the grouped scheduler can collate
            # it without sharing or reseeding a process-global generator.
            batch.generator = None
        return batch


class DM05ActionPostprocessStage(VLAActionPostprocessStage):
    def forward(self, batch, server_args):
        state = vla_state(batch)
        if not batch.is_warmup and state.get("actions_output") is None:
            state["actions"] = postprocess_dm05_actions(
                state["actions"],
                state["observation_batch"],
                server_args.pipeline_config,
            )
        return super().forward(batch, server_args)


class DM05Pipeline(ComposedPipelineBase):
    pipeline_name = "DM05Pipeline"
    pipeline_config_cls = DM05PipelineConfig
    sampling_params_cls = DM05SamplingParams
    _required_config_modules: list[str] = []

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError("DM05 prefix and action inference run in the same worker")

    def load_modules(self, server_args, loaded_modules=None):
        if loaded_modules is not None:
            return loaded_modules
        if server_args.pipeline_config.precision_policy == "fp32_mixed":
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        return {
            "policy_model": DM05PolicyModel.from_pretrained(
                self.model_path,
                server_args.pipeline_config,
                revision=getattr(server_args, "revision", None),
            ),
        }

    def initialize_pipeline(self, server_args) -> None:
        config = server_args.pipeline_config
        model_path = self.get_module("policy_model").model_path
        self.preprocessor = DM05Preprocessor(config, model_path)
        self.prefix_cache = VLAPrefixCacheManager(
            max_entries=config.prefix_cache_max_entries
        )

    def create_pipeline_stages(self, server_args):
        self.add_stage(
            DM05ObservationPreprocessStage(
                self.preprocessor, self.get_module("policy_model")
            ),
            "dm05_preprocess",
        )
        self.add_stage(
            VLAPrefixEncodingStage(
                self.get_module("policy_model"),
                self.prefix_cache,
                observation_collator=collate_dm05_observations,
                observation_signature=dm05_observation_signature,
            ),
            "dm05_prefix",
        )
        self.add_stage(
            VLAActionDenoisingStage(
                self.get_module("policy_model"),
                output_transform=postprocess_dm05_actions,
            ),
            "dm05_action_denoise",
        )
        self.add_stage(DM05ActionPostprocessStage(), "dm05_postprocess")


EntryClass = DM05Pipeline
