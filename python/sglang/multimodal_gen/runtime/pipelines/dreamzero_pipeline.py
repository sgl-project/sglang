# SPDX-License-Identifier: Apache-2.0
"""DreamZero DROID one-shot action pipeline."""

from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.configs.pipeline_configs.dreamzero import (
    DreamZeroPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.dreamzero import DreamZeroSamplingParams
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_world_size,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.managers.dreamzero_session_cache import (
    DreamZeroCachePoolManager,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.dreamzero import (
    DreamZeroActionOutputStage,
    DreamZeroCausalDenoisingStage,
    DreamZeroObsPrepStage,
    DreamZeroTextEncodingStage,
    DreamZeroVisualEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.torch_compile import (
    build_torch_compile_kwargs,
    maybe_enable_inductor_compute_comm_overlap,
)
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


def _component_path(server_args: ServerArgs, model_path: str, *names: str) -> str:
    component_paths = getattr(server_args, "component_paths", None) or {}
    for name in names:
        path = component_paths.get(name)
        if path:
            return path
    return model_path


def _dtype_from_precision(precision: str | None) -> torch.dtype:
    return PRECISION_TO_TYPE.get(precision or "bf16", torch.bfloat16)


def _compile_dreamzero_dit_blocks(transformer: Any) -> int:
    blocks = getattr(transformer, "blocks", None)
    if not isinstance(blocks, torch.nn.ModuleList):
        logger.warning("Skipping DreamZero DiT compile; transformer.blocks not found")
        return 0

    maybe_enable_inductor_compute_comm_overlap()

    torch._dynamo.config.cache_size_limit = max(
        getattr(torch._dynamo.config, "cache_size_limit", 64), 128
    )
    compile_kwargs = build_torch_compile_kwargs(mode="default")
    for index, block in enumerate(blocks):
        blocks[index] = torch.compile(block, **compile_kwargs)
    return len(blocks)


class DreamZeroPipeline(ComposedPipelineBase):
    """Pipeline that composes DreamZero obs prep, text encoding, DiT and action output."""

    pipeline_name = "DreamZeroPipeline"
    is_video_pipeline = False
    _required_config_modules = [
        "text_encoder",
        "image_encoder",
        "vae",
        "transformer",
        "scheduler",
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
        pc = server_args.pipeline_config
        dit_path = _component_path(
            server_args, self.model_path, "dreamzero_dit", "transformer"
        )
        from sglang.multimodal_gen.runtime.loader.component_loaders.dreamzero_config import (
            materialize_arch_configs_from_checkpoint,
        )

        materialize_arch_configs_from_checkpoint(dit_path, pc)
        if loaded_modules is not None:
            return modules

        from sglang.multimodal_gen.runtime.loader.component_loaders.dreamzero_dit_loader import (
            build_dreamzero_dit_from_checkpoint,
            load_dreamzero_dit_checkpoint,
        )
        from sglang.multimodal_gen.runtime.loader.component_loaders.dreamzero_encoder_loader import (
            build_dreamzero_image_encoder_from_checkpoint,
            build_dreamzero_text_encoder_from_checkpoint,
        )
        from sglang.multimodal_gen.runtime.loader.component_loaders.dreamzero_vae_loader import (
            build_dreamzero_vae_from_checkpoint,
        )
        from sglang.multimodal_gen.runtime.platforms import current_platform

        device = torch.device(current_platform.device_type)
        if current_platform.device_type == "cuda":
            device = torch.device(f"cuda:{torch.cuda.current_device()}")

        use_tensor_parallel = bool(
            getattr(pc, "dreamzero_use_tensor_parallel", False)
            or (getattr(server_args, "tp_size", 1) or 1) > 1
        )

        transformer = build_dreamzero_dit_from_checkpoint(
            dit_path,
            device=device,
            dtype=_dtype_from_precision(pc.dit_precision),
            use_tensor_parallel=use_tensor_parallel,
        )
        dit_report = load_dreamzero_dit_checkpoint(
            transformer,
            dit_path,
            device=device,
            strict=True,
        )
        logger.info("Loaded DreamZero DiT checkpoint: %s", dit_report.as_dict())
        modules["transformer"] = transformer

        vae_path = _component_path(server_args, self.model_path, "dreamzero_vae", "vae")
        vae, vae_report = build_dreamzero_vae_from_checkpoint(
            vae_path,
            device=device,
            dtype=_dtype_from_precision(pc.vae_precision),
            strict=True,
        )
        logger.info("Loaded DreamZero VAE checkpoint: %s", vae_report.as_dict())
        modules["vae"] = vae

        text_path = _component_path(
            server_args, self.model_path, "dreamzero_text_encoder", "text_encoder"
        )
        text_encoder, text_report = build_dreamzero_text_encoder_from_checkpoint(
            text_path,
            device=device,
            dtype=_dtype_from_precision(pc.text_encoder_precisions[0]),
            strict=True,
        )
        logger.info(
            "Loaded DreamZero text encoder checkpoint: %s", text_report.as_dict()
        )
        modules["text_encoder"] = text_encoder

        image_path = _component_path(
            server_args, self.model_path, "dreamzero_image_encoder", "image_encoder"
        )
        image_encoder, image_report = build_dreamzero_image_encoder_from_checkpoint(
            image_path,
            device=device,
            dtype=_dtype_from_precision(pc.image_encoder_precision),
            strict=True,
        )
        logger.info(
            "Loaded DreamZero image encoder checkpoint: %s", image_report.as_dict()
        )
        modules["image_encoder"] = image_encoder

        if getattr(pc, "dreamzero_compile_components", True):
            compiled_blocks = _compile_dreamzero_dit_blocks(transformer)
            logger.info("Compiled %d DreamZero DiT blocks", compiled_blocks)
        return modules

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        self.modules.setdefault("scheduler", self._build_scheduler(server_args))
        configured_sp_size = int(
            server_args.pipeline_config.dreamzero_sequence_parallel_size
        )
        if model_parallel_is_initialized():
            actual_sp_size = get_sp_world_size()
            if configured_sp_size != actual_sp_size:
                raise ValueError(
                    "DreamZero sequence parallel size must match the initialized SP "
                    f"group: configured={configured_sp_size}, actual={actual_sp_size}"
                )
        elif configured_sp_size > 1:
            raise RuntimeError(
                "DreamZero SP requires initialized model-parallel process groups"
            )
        self.cache_manager = DreamZeroCachePoolManager(
            max_sessions=server_args.pipeline_config.dreamzero_max_sessions
        )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(DreamZeroObsPrepStage(), "dreamzero_obs_prep_stage")
        self.add_stage(
            DreamZeroTextEncodingStage(
                self.get_module("text_encoder"),
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
