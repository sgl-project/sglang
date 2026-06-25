"""Krea-2 text-to-image pipeline (native diffusers layout).

The released repo is diffusers-style (``model_index.json`` + ``transformer/``,
``text_encoder/``, ``vae/``, ``tokenizer/``, ``scheduler/`` subfolders), so the
base ``load_modules`` loads every component from it (the MMDiT via
``Krea2Transformer2DModel``, the Qwen3-VL text encoder, the Qwen-Image VAE, and
the ``FlowMatchEulerDiscreteScheduler``). This pipeline only adds the two
K2-specific touches the base loader can't infer: dropping the unused Qwen3-VL
vision tower (K2 conditions on text only) and building the assistant-suffix
tokenizer (``processor``), which has no ``model_index.json`` entry. The stage
chain is Krea2BeforeDenoisingStage -> DenoisingStage -> DecodingStage.
"""

from typing import Any

import torch
from transformers import Qwen2TokenizerFast

from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.krea2 import (
    Krea2BeforeDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_TEXT_MAX_LENGTH = 512


class Krea2Pipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "Krea2Pipeline"

    from sglang.multimodal_gen.configs.pipeline_configs.krea2 import Krea2PipelineConfig
    from sglang.multimodal_gen.configs.sample.krea2 import Krea2SamplingParams

    pipeline_config_cls = Krea2PipelineConfig
    sampling_params_cls = Krea2SamplingParams

    # Every entry is a diffusers/transformers component declared in model_index.json,
    # so the base loader handles them. "processor" is added in load_modules (no entry).
    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        vae_config = server_args.pipeline_config.vae_config
        if hasattr(vae_config, "post_init"):
            vae_config.post_init()

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        modules = super().load_modules(server_args, loaded_modules)

        # K2 conditions on text only: drop the unused Qwen3-VL vision tower that the
        # base loader brings in with the full Qwen3VLModel (frees its weights and
        # shrinks the encoder's CPU<->GPU page). It sits on the encoder or under .model.
        text_encoder = modules.get("text_encoder")
        if text_encoder is not None:
            for owner in (text_encoder, getattr(text_encoder, "model", None)):
                if owner is not None and getattr(owner, "visual", None) is not None:
                    del owner.visual
                    break

        # The conditioner appends a fixed assistant suffix, tokenized separately;
        # model_index.json has no "processor" entry, so build one from tokenizer/.
        tok_path = self._resolve_component_path(server_args, "tokenizer", "tokenizer")
        modules["processor"] = Qwen2TokenizerFast.from_pretrained(
            tok_path, max_length=_TEXT_MAX_LENGTH
        )
        return modules

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(
            Krea2BeforeDenoisingStage(
                vae=self.get_module("vae"),
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                processor=self.get_module("processor"),
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
            "k2_before_denoising_stage",
        )
        self.add_stage(
            DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        self.add_standard_decoding_stage()


EntryClass = [Krea2Pipeline]
