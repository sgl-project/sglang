# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sglang.multimodal_gen.configs.pipeline_configs.pi05 import Pi05PipelineConfig
from sglang.multimodal_gen.configs.sample.pi05 import Pi05SamplingParams
from sglang.multimodal_gen.runtime.cache.vla_prefix_cache import VLAPrefixCacheManager
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed.vla_topology import (
    VLAParallelTopology,
)
from sglang.multimodal_gen.runtime.models.pi05 import Pi05PolicyModel
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.pi05 import (
    Pi05ActionDenoisingStage,
    Pi05PostprocessStage,
    Pi05PrefixStage,
    Pi05PreprocessStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.pi05_preprocess import (
    Pi05Preprocessor,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


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
    ) -> dict:
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
        parallel_topology = VLAParallelTopology.from_config(pipeline_config)
        parallel_topology.validate()
        return {
            "parallel_topology": parallel_topology,
            "policy_model": policy_model,
            "preprocessor": Pi05Preprocessor(pipeline_config),
            "prefix_cache": VLAPrefixCacheManager(
                max_entries=pipeline_config.prefix_cache_max_entries
            ),
        }

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(
            Pi05PreprocessStage(self.get_module("preprocessor")),
            "pi05_preprocess",
        )
        self.add_stage(
            Pi05PrefixStage(
                self.get_module("policy_model"),
                self.get_module("prefix_cache"),
            ),
            "pi05_prefix",
        )
        self.add_stage(
            Pi05ActionDenoisingStage(self.get_module("policy_model")),
            "pi05_action_denoise",
        )
        self.add_stage(Pi05PostprocessStage(), "pi05_postprocess")


EntryClass = Pi05Pipeline
