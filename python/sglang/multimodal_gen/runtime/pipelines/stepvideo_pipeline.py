# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# type: ignore
# SPDX-License-Identifier: Apache-2.0
"""
Hunyuan video diffusion pipeline implementation.

This module contains an implementation of the Hunyuan video diffusion pipeline
using the modular pipeline architecture.
"""

import os
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loader import (
    PipelineComponentLoader,
)
from sglang.multimodal_gen.runtime.models.encoders.bert import (
    HunyuanClip,  # type: ignore
)
from sglang.multimodal_gen.runtime.models.encoders.stepllm import STEP1TextEncoder
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DecodingStage,
    DenoisingStage,
    InputValidationStage,
    LatentPreparationStage,
    StepvideoPromptEncodingStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class StepVideoPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "StepVideoPipeline"

    _required_config_modules = ["transformer", "scheduler", "vae"]

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(
            stage_name="input_validation_stage", stage=InputValidationStage()
        )

        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=StepvideoPromptEncodingStage(
                stepllm=self.get_module("text_encoder"),
                clip=self.get_module("text_encoder_2"),
            ),
        )

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae"))
        )

    def build_llm(self, model_dir, device) -> torch.nn.Module:
        text_encoder = (
            STEP1TextEncoder(model_dir, max_length=320).to(torch.bfloat16).eval()
        )
        return text_encoder

    def build_clip(self, model_dir, device) -> HunyuanClip:
        clip = HunyuanClip(model_dir, max_length=77).eval()
        return clip

    def initialize_pipeline(self, server_args: ServerArgs):
        """
        Initialize the pipeline.
        """
        target_device = get_local_torch_device()
        llm_dir = os.path.join(self.model_path, "step_llm")
        clip_dir = os.path.join(self.model_path, "hunyuan_clip")
        text_enc = self.build_llm(llm_dir, target_device)
        clip_enc = self.build_clip(clip_dir, target_device)
        self.add_module("text_encoder", text_enc)
        self.add_module("text_encoder_2", clip_enc)
        lib_path = (
            os.path.join(
                server_args.model_path,
                "lib/liboptimus_ths-torch2.5-cu124.cpython-310-x86_64-linux-gnu.so",
            )
            if os.path.isdir(server_args.model_path)  # local checkout
            else hf_hub_download(
                repo_id=server_args.model_path,
                filename="lib/liboptimus_ths-torch2.5-cu124.cpython-310-x86_64-linux-gnu.so",
            )
        )
        torch.ops.load_library(lib_path)

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        """
        Load the modules from the config.
        """
        model_index = self._load_config()
        logger.info("Loading pipeline modules from config: %s", model_index)

        # remove keys that are not pipeline modules
        model_index.pop("_class_name")
        model_index.pop("_diffusers_version")

        # some sanity checks
        assert (
            len(model_index) > 1
        ), "model_index.json must contain at least one pipeline module"

        required_modules = ["transformer", "scheduler", "vae"]
        for module_name in required_modules:
            if module_name not in model_index:
                raise ValueError(
                    f"model_index.json must contain a {module_name} module"
                )
        logger.info("Diffusers config passed sanity checks")

        # all the component models used by the pipeline
        modules = {}
        for module_name, (
            transformers_or_diffusers,
            architecture,
        ) in model_index.items():
            component_model_path = os.path.join(self.model_path, module_name)
            module = PipelineComponentLoader.load_module(
                module_name=module_name,
                component_model_path=component_model_path,
                transformers_or_diffusers=transformers_or_diffusers,
                server_args=server_args,
            )
            logger.info("Loaded module %s from %s", module_name, component_model_path)

            if module_name in modules:
                logger.warning("Overwriting module %s", module_name)
            modules[module_name] = module

        required_modules = self.required_config_modules
        # Check if all required modules were loaded
        for module_name in required_modules:
            if module_name not in modules or modules[module_name] is None:
                raise ValueError(
                    f"Required module {module_name} was not loaded properly"
                )

        return modules


EntryClass = StepVideoPipeline
