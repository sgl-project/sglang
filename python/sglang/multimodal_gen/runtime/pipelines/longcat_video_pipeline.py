import os
from typing import Any

import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

from sglang.multimodal_gen.configs.pipeline_configs.longcat_video import (
    longcat_text_postprocess,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DecodingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.longcat_video import (
    LongCatVideoDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class LongCatComponentSpec(dict):
    def __init__(self, library: str, architecture: str):
        super().__init__({"library": library, "architecture": architecture})

    def __iter__(self):
        yield self["library"]
        yield self["architecture"]


def synthesize_longcat_model_index(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Synthesize a diffusers-compatible model_index for a LongCat-Video checkpoint.

    LongCat-Video does not ship a standard diffusers model_index.json — its on-disk
    model_index contains only ``{"model_name": "LongCat-Video"}`` and uses a ``dit/``
    directory instead of the diffusers-standard ``transformer/``.  Rather than patching
    the checkpoint or requiring users to restructure it, we build the index in code so
    the rest of the pipeline loading machinery can proceed normally.
    """
    return {
        "_class_name": LongCatVideoPipeline.pipeline_name,
        "_diffusers_version": "0.35.0",
        "tokenizer": LongCatComponentSpec("transformers", "AutoTokenizer"),
        "text_encoder": LongCatComponentSpec("transformers", "UMT5EncoderModel"),
        "vae": LongCatComponentSpec("diffusers", "AutoencoderKLWan"),
        "scheduler": LongCatComponentSpec(
            "diffusers", "FlowMatchEulerDiscreteScheduler"
        ),
        "transformer": LongCatComponentSpec(
            "diffusers", "LongCatVideoTransformer3DModel"
        ),
    }


class LongCatVideoPipeline(ComposedPipelineBase):
    pipeline_name = "LongCatVideoPipeline"
    is_video_pipeline = True

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def _load_config(self) -> dict[str, Any]:
        # Use force_diffusers_model=False because LongCat-Video uses a non-standard
        # model_index.json ({"model_name": "LongCat-Video"}) and a "dit/" directory
        # instead of "transformer/". The synthesize_longcat_model_index() call below
        # handles the non-standard format, so we skip the completeness check here.
        model_path = maybe_download_model(self.model_path, force_diffusers_model=False)
        self.model_path = model_path
        logger.info("Model path: %s", model_path)
        return synthesize_longcat_model_index(model_path)

    def _resolve_component_path(
        self, server_args: ServerArgs, module_name: str, load_module_name: str
    ) -> str:
        override_path = server_args.component_paths.get(module_name)
        if override_path is not None:
            return maybe_download_model(override_path)
        if module_name == "transformer" or load_module_name == "transformer":
            return os.path.join(self.model_path, "dit")
        return os.path.join(self.model_path, load_module_name)

    def initialize_pipeline(self, server_args: ServerArgs):
        if server_args.num_gpus != 1:
            raise NotImplementedError(
                "LongCat T2V MVP only supports single-process, single-GPU inference."
            )
        if server_args.tp_size not in (None, 1):
            raise NotImplementedError("LongCat T2V MVP does not support TP/CP.")
        if server_args.sp_degree not in (None, 1):
            raise NotImplementedError("LongCat T2V MVP does not support SP.")
        if server_args.ulysses_degree not in (None, 1):
            raise NotImplementedError("LongCat T2V MVP does not support Ulysses SP.")
        if server_args.ring_degree not in (None, 1):
            raise NotImplementedError("LongCat T2V MVP does not support Ring SP.")
        if server_args.enable_cfg_parallel:
            raise NotImplementedError("LongCat T2V MVP does not support CFG parallel.")
        if server_args.pipeline_config.task_type.name != "T2V":
            raise NotImplementedError("LongCat MVP only supports T2V generation.")

        # Reconstruct the scheduler with default parameters rather than using the
        # one loaded from the checkpoint. The checkpoint's scheduler config may
        # contain non-default sigmas incompatible with LongCatVideoPipelineConfig
        # .prepare_sigmas(). A fresh default instance guarantees a clean state.
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler()

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stages(
            [
                InputValidationStage(),
                TextEncodingStage(
                    text_encoders=[self.get_module("text_encoder")],
                    tokenizers=[self.get_module("tokenizer")],
                ),
                LatentPreparationStage(
                    scheduler=self.get_module("scheduler"),
                    transformer=self.get_module("transformer"),
                ),
                TimestepPreparationStage(scheduler=self.get_module("scheduler")),
                LongCatVideoDenoisingStage(
                    transformer=self.get_module("transformer"),
                    scheduler=self.get_module("scheduler"),
                    vae=self.get_module("vae"),
                    pipeline=self,
                ),
                DecodingStage(vae=self.get_module("vae"), pipeline=self),
            ]
        )


EntryClass = LongCatVideoPipeline

__all__ = [
    "LongCatVideoPipeline",
    "EntryClass",
    "longcat_text_postprocess",
    "synthesize_longcat_model_index",
]
