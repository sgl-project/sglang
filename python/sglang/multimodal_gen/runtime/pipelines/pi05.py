# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sglang.multimodal_gen.configs.pipeline_configs.pi05 import Pi05PipelineConfig
from sglang.multimodal_gen.configs.sample.pi05 import Pi05SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.models.vlas import Pi05PolicyModel
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.pi05_preprocess import (
    Pi05Preprocessor,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.vla import (
    VLAActionDenoisingStage,
    VLAActionPostprocessStage,
    VLAObservationPreprocessStage,
    VLAPrefixEncodingStage,
    VLAStageKeys,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.vla.prefix_cache import VLAPrefixCacheManager

logger = init_logger(__name__)

PI05_STAGE_KEYS = VLAStageKeys.for_namespace("pi05")


class Pi05Pipeline(ComposedPipelineBase):
    pipeline_name = "Pi05Pipeline"
    pipeline_config_cls = Pi05PipelineConfig
    sampling_params_cls = Pi05SamplingParams
    _required_config_modules: list[str] = []

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError(
                "Pi05Pipeline v1 supports same-process execution only. "
                "Use prefix/action logical groups inside one worker; cross-node "
                "multimodal_gen disaggregation is a v2 target."
            )

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, torch.nn.Module]:
        if loaded_modules is not None:
            return loaded_modules

        pipeline_config: Pi05PipelineConfig = server_args.pipeline_config
        pipeline_config.offload_prefix_image_encoder = (
            pipeline_config.offload_prefix_image_encoder
            or bool(server_args.image_encoder_cpu_offload)
        )
        pipeline_config.offload_prefix_token_embedding = (
            pipeline_config.offload_prefix_token_embedding
            or bool(server_args.text_encoder_cpu_offload)
        )
        logger.info(
            "Pi05 memory config: prefix_cache=%s/%s, action_cuda_graph=%s, "
            "offload_image=%s, offload_image_after_embed=%s, "
            "offload_tokens=%s, offload_language_layers=%s, "
            "offload_language_after_prefix=%s/%s, "
            "offload_action_after_denoise=%s, empty_cache_after_prefix=%s",
            pipeline_config.enable_global_prefix_cache,
            pipeline_config.prefix_cache_max_entries,
            pipeline_config.enable_action_cuda_graph,
            pipeline_config.offload_prefix_image_encoder,
            pipeline_config.offload_prefix_image_encoder_after_embed,
            pipeline_config.offload_prefix_token_embedding,
            pipeline_config.offload_prefix_language_layers,
            pipeline_config.offload_prefix_language_layers_after_prefix,
            pipeline_config.offload_prefix_language_layer_count_after_prefix,
            pipeline_config.offload_action_expert_after_denoise,
            pipeline_config.empty_cache_after_prefix,
        )
        policy_model = Pi05PolicyModel.from_pretrained(
            self.model_path,
            pipeline_config,
        )
        if (
            pipeline_config.prefix_parallel_strategy
            == pipeline_config.action_parallel_strategy
            == "tp"
        ):
            raise ValueError(
                "VLA action expert should not share the prefix TP layout. "
                "Use SP, Ulysses, Ring, DP, or monolithic fallback for the "
                "action path."
            )
        return {
            "policy_model": policy_model,
        }

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        pipeline_config: Pi05PipelineConfig = server_args.pipeline_config
        self.preprocessor = Pi05Preprocessor(pipeline_config)
        self.prefix_cache = VLAPrefixCacheManager(
            max_entries=pipeline_config.prefix_cache_max_entries
        )

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(
            VLAObservationPreprocessStage(
                self.preprocessor,
                keys=PI05_STAGE_KEYS,
            ),
            "pi05_preprocess",
        )
        self.add_stage(
            VLAPrefixEncodingStage(
                self.get_module("policy_model"),
                self.prefix_cache,
                keys=PI05_STAGE_KEYS,
            ),
            "pi05_prefix",
        )
        self.add_stage(
            VLAActionDenoisingStage(
                self.get_module("policy_model"),
                keys=PI05_STAGE_KEYS,
            ),
            "pi05_action_denoise",
        )
        self.add_stage(
            VLAActionPostprocessStage(keys=PI05_STAGE_KEYS),
            "pi05_postprocess",
        )


EntryClass = Pi05Pipeline
