# SPDX-License-Identifier: Apache-2.0
import os
from typing import Any

from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class BooguTextEncodingStage(TextEncodingStage):
    def _forward_text_encoder(self, text_encoder, encoder_forward_kwargs):
        inner = text_encoder.model
        with set_forward_context(current_timestep=0, attn_metadata=None):
            return inner(**encoder_forward_kwargs)


class BooguImagePipeline(ComposedPipelineBase):
    pipeline_name = "BooguImagePipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    _slot_renames = {"mllm": "text_encoder", "processor": "tokenizer"}

    def _load_config(self) -> dict[str, Any]:
        model_index = super()._load_config()
        for src, dst in self._slot_renames.items():
            if src in model_index:
                model_index[dst] = model_index.pop(src)
                override = os.path.join(self.model_path, src)
                self.server_args.component_paths.setdefault(dst, override)

        for slot in ("transformer", "scheduler"):
            entry = model_index.get(slot)
            if isinstance(entry, list) and len(entry) == 2 and entry[0] != "diffusers":
                model_index[slot] = ["diffusers", entry[1]]
        return model_index

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(InputValidationStage())
        self.add_stage(
            BooguTextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            )
        )
        self.add_standard_latent_preparation_stage()
        self.add_standard_timestep_preparation_stage()
        self.add_standard_denoising_stage()
        self.add_standard_decoding_stage()


EntryClass = BooguImagePipeline
