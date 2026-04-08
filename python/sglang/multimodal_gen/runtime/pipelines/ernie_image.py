# SPDX-License-Identifier: Apache-2.0
"""ErnieImage text-to-image pipeline."""

import json
import os

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import InputValidationStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ernie_image_pe import (
    PromptEnhancementStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import TextEncodingStage
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model_index
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class ErnieImagePipeline(LoRAPipeline, ComposedPipelineBase):

    pipeline_name = "ErnieImagePipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def _has_pe_in_model_index(self, server_args) -> bool:
        try:
            model_index = maybe_download_model_index(server_args.model_path)
            return "pe" in model_index and model_index["pe"] is not None
        except Exception:
            return False

    def _read_tokenizer_model_max_length(self, model_path: str):
        """Read model_max_length from tokenizer/tokenizer_config.json.

        Supports both local paths and HuggingFace Hub model IDs.
        Returns None if the value cannot be determined.
        """
        tokenizer_config_subpath = os.path.join("tokenizer", "tokenizer_config.json")

        # Local path
        if os.path.exists(model_path):
            config_path = os.path.join(model_path, tokenizer_config_subpath)
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                return config.get("model_max_length")
            return None

        # Remote HuggingFace Hub model ID
        try:
            from huggingface_hub import hf_hub_download
            import tempfile

            with tempfile.TemporaryDirectory() as tmp_dir:
                config_path = hf_hub_download(
                    repo_id=model_path,
                    filename=tokenizer_config_subpath,
                    local_dir=tmp_dir,
                )
                with open(config_path) as f:
                    config = json.load(f)
                return config.get("model_max_length")
        except Exception as e:
            logger.warning("Failed to read tokenizer_config.json from %s: %s", model_path, e)
            return None

    def load_modules(self, server_args, loaded_modules=None):
        if self._has_pe_in_model_index(server_args):
            if "pe" not in self._required_config_modules:
                self._required_config_modules.insert(0, "pe")
            logger.info("PE model detected in model_index.json, will load PE module.")

        model_max_length = self._read_tokenizer_model_max_length(server_args.model_path)
        if model_max_length is not None:
            pipeline_config = server_args.pipeline_config
            if (
                hasattr(pipeline_config, "text_encoder_extra_args")
                and pipeline_config.text_encoder_extra_args
            ):
                pipeline_config.text_encoder_extra_args[0]["max_length"] = model_max_length
                logger.info(
                    "Set text encoder max_length=%d from tokenizer/tokenizer_config.json",
                    model_max_length,
                )
        else:
            logger.warning(
                "Could not read model_max_length from tokenizer/tokenizer_config.json, "
                "using default max_length from pipeline config."
            )

        return super().load_modules(server_args, loaded_modules)

    def create_pipeline_stages(self, server_args):
        self.add_stage(InputValidationStage())

        pe_model = self.get_module("pe")
        if pe_model is not None:
            pe_tokenizer = getattr(pe_model, "pe_tokenizer", None)
            if pe_tokenizer is None:
                # Fallback: load tokenizer directly from PE model path
                from transformers import AutoTokenizer

                pe_component_path = os.path.join(self.model_path, "pe")
                pe_override = server_args.component_paths.get("pe")
                if pe_override is not None:
                    pe_component_path = pe_override
                logger.warning(
                    "pe_tokenizer not found on pe_model (%s), loading from %s",
                    type(pe_model).__name__,
                    pe_component_path,
                )
                pe_tokenizer = AutoTokenizer.from_pretrained(
                    pe_component_path,
                    trust_remote_code=server_args.trust_remote_code,
                )
            self.add_stage(
                PromptEnhancementStage(
                    pe_model=pe_model,
                    pe_tokenizer=pe_tokenizer,
                ),
                "prompt_enhancement_stage",
            )

        self.add_stage(
            TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
            "prompt_encoding_stage_primary",
        )

        self.add_standard_timestep_preparation_stage()
        self.add_standard_latent_preparation_stage()
        self.add_standard_denoising_stage()
        self.add_standard_decoding_stage()


EntryClass = ErnieImagePipeline
