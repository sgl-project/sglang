# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any

import torch

from sglang.multimodal_gen.configs.pipeline_configs.dreamzero import (
    DreamZeroPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.dreamzero import DreamZeroSamplingParams
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_world_size,
    get_tp_world_size,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.dreamzero.denoising import (
    DreamZeroActionOutputStage,
    DreamZeroCausalDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.dreamzero.image_encoding import (
    DreamZeroObsPrepStage,
    DreamZeroVisualEncodingStage,
    load_dreamzero_image_encoder,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.dreamzero.session_cache import (
    DreamZeroCachePoolManager,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.dreamzero.text_encoding import (
    DreamZeroTextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.torch_compile import (
    build_torch_compile_kwargs,
    maybe_enable_inductor_compute_comm_overlap,
)

logger = init_logger(__name__)


def _compile_dreamzero_dit_blocks(transformer: Any) -> int:
    blocks = transformer.blocks
    if not isinstance(blocks, torch.nn.ModuleList):
        raise TypeError("DreamZero transformer.blocks must be a ModuleList")

    maybe_enable_inductor_compute_comm_overlap()

    torch._dynamo.config.cache_size_limit = max(
        torch._dynamo.config.cache_size_limit,
        128,
    )
    compile_kwargs = build_torch_compile_kwargs(mode="default")
    for index, block in enumerate(blocks):
        blocks[index] = torch.compile(block, **compile_kwargs)
    return len(blocks)


def _is_sglang_dreamzero_checkpoint(model_path: str) -> bool:
    return (
        os.path.isfile(os.path.join(model_path, "model_index.json"))
        and os.path.isdir(os.path.join(model_path, "tokenizer"))
        and os.path.isdir(os.path.join(model_path, "transformer"))
        and os.path.isdir(os.path.join(model_path, "text_encoder"))
        and os.path.isdir(os.path.join(model_path, "image_encoder"))
        and os.path.isdir(os.path.join(model_path, "vae"))
    )


def _validate_dreamzero_parallel_config(server_args: ServerArgs) -> None:
    configured_tp_size = int(server_args.tp_size)
    configured_sp_size = int(server_args.sp_degree)
    if model_parallel_is_initialized():
        actual_tp_size = get_tp_world_size()
        if configured_tp_size != actual_tp_size:
            raise ValueError(
                "DreamZero tensor parallel size must match the initialized TP "
                f"group: configured={configured_tp_size}, actual={actual_tp_size}"
            )
        actual_sp_size = get_sp_world_size()
        if configured_sp_size != actual_sp_size:
            raise ValueError(
                "DreamZero sp_degree must match the initialized SP "
                f"group: configured={configured_sp_size}, actual={actual_sp_size}"
            )
    elif configured_tp_size > 1 or configured_sp_size > 1:
        raise RuntimeError(
            "DreamZero TP/SP requires initialized model-parallel process groups"
        )


class DreamZeroPipeline(ComposedPipelineBase):
    """Pipeline that composes DreamZero obs prep, text encoding, DiT and action output."""

    pipeline_name = "DreamZeroPipeline"
    is_video_pipeline = False
    _required_config_modules = [
        "tokenizer",
        "text_encoder",
        "image_encoder",
        "vae",
        "transformer",
    ]
    pipeline_config_cls = DreamZeroPipelineConfig
    sampling_params_cls = DreamZeroSamplingParams

    def _build_scheduler(self, server_args: ServerArgs) -> FlowUniPCMultistepScheduler:
        return FlowUniPCMultistepScheduler(
            shift=server_args.pipeline_config.flow_shift,
        )

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        modules = dict(loaded_modules or {})
        modules.setdefault("scheduler", self._build_scheduler(server_args))
        if loaded_modules is not None:
            return modules

        if not _is_sglang_dreamzero_checkpoint(self.model_path):
            raise RuntimeError(
                "DreamZeroPipeline requires a checkpoint in SGLang component layout "
                "with model_index.json plus tokenizer/transformer/text_encoder/"
                "image_encoder/vae component directories."
            )

        server_args.pipeline_config.dit_config.arch_config.use_tensor_parallel = (
            server_args.tp_size > 1
        )
        modules["image_encoder"] = load_dreamzero_image_encoder(
            server_args,
            self._resolve_component_path(server_args, "image_encoder", "image_encoder"),
        )
        modules.update(super().load_modules(server_args, modules))
        if server_args.pipeline_config.dreamzero_compile_components:
            compiled_blocks = _compile_dreamzero_dit_blocks(modules["transformer"])
            logger.info("Compiled %d DreamZero DiT blocks", compiled_blocks)
        return modules

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        self.modules.setdefault("scheduler", self._build_scheduler(server_args))
        _validate_dreamzero_parallel_config(server_args)
        self.cache_manager = DreamZeroCachePoolManager(
            max_sessions=server_args.pipeline_config.dreamzero_max_sessions
        )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(DreamZeroObsPrepStage(), "dreamzero_obs_prep_stage")
        self.add_stage(
            DreamZeroTextEncodingStage(
                self.get_module("text_encoder"),
                self.get_module("tokenizer"),
                cache_manager=self.cache_manager,
            ),
            "dreamzero_text_encoding_stage",
        )
        self.add_stage(
            DreamZeroVisualEncodingStage(
                image_encoder=self.get_module("image_encoder"),
                vae=self.get_module("vae"),
                cache_manager=self.cache_manager,
            ),
            "dreamzero_visual_encoding_stage",
        )
        self.add_stage(
            DreamZeroCausalDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                cache_manager=self.cache_manager,
            ),
            "dreamzero_causal_denoising_stage",
        )
        self.add_stage(
            DreamZeroActionOutputStage(),
            "dreamzero_action_postproc_stage",
        )


EntryClass = DreamZeroPipeline
