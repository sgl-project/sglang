# SPDX-License-Identifier: Apache-2.0
"""SANA-Video text-to-video pipeline."""

import torch

from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

SANA_VIDEO_COMPLEX_HUMAN_INSTRUCTION = (
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed "
    "visual descriptions suitable for video generation. Evaluate the level of "
    "detail in the user prompt:\n"
    "- If the prompt is simple, focus on adding specifics about colors, shapes, "
    "sizes, textures, motion, and temporal relationships to create vivid and "
    "dynamic scenes.\n"
    "- If the prompt is already detailed, refine and enhance the existing details "
    "slightly without overcomplicating.\n"
    "Here are examples of how to transform or refine prompts:\n"
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat slowly "
    "settling into a curled position, peacefully falling asleep on a warm sunny "
    "windowsill, with gentle sunlight filtering through surrounding pots of "
    "blooming red flowers.\n"
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene "
    "at dusk, featuring glowing street lamps gradually lighting up, a diverse "
    "crowd of people in colorful clothing walking past, and a double-decker bus "
    "smoothly passing by towering glass skyscrapers.\n"
    "Please generate only the enhanced description for the prompt below and avoid "
    "including any additional commentary or evaluations:\n"
    "User Prompt: "
)


def select_sana_video_prompt_window(
    tensor: torch.Tensor, max_sequence_length: int
) -> torch.Tensor:
    """Keep the BOS token and the final prompt window, matching Diffusers."""
    if tensor.shape[1] < max_sequence_length:
        raise ValueError(
            f"Encoded prompt has {tensor.shape[1]} tokens, expected at least "
            f"{max_sequence_length}"
        )
    if max_sequence_length == 1:
        return tensor[:, :1]
    return torch.cat([tensor[:, :1], tensor[:, -(max_sequence_length - 1) :]], dim=1)


class SanaVideoTextEncodingStage(TextEncodingStage):
    """Apply SANA-Video's asymmetric positive/negative prompt encoding."""

    @staticmethod
    def _normalize_text(text: str | list[str]) -> str | list[str]:
        if isinstance(text, str):
            return text.lower().strip()
        return [item.lower().strip() for item in text]

    def _encode_negative_text(self, batch, server_args, all_indices):
        cache_key = self._build_negative_text_cache_key(batch, server_args, all_indices)
        cached = self._get_cached_negative_text_embedding(cache_key)
        if cached is not None:
            return cached
        outputs = self.encode_text(
            self._normalize_text(batch.negative_prompt),
            server_args,
            encoder_index=all_indices,
            return_attention_mask=True,
            max_length=300,
        )
        self._maybe_cache_negative_text_embedding(cache_key, outputs)
        return outputs

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        assert batch.prompt is not None
        self.tokenizers[0].padding_side = "right"
        all_indices = list(range(len(self.text_encoders)))
        max_sequence_length = batch.max_sequence_length or 300
        prompt = self._normalize_text(batch.prompt)
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        enhanced_prompt = [
            SANA_VIDEO_COMPLEX_HUMAN_INSTRUCTION + item for item in prompt_list
        ]
        instruction_tokens = len(
            self.tokenizers[0].encode(SANA_VIDEO_COMPLEX_HUMAN_INSTRUCTION)
        )
        encoded_length = instruction_tokens + max_sequence_length - 2
        positive_outputs = list(
            self.encode_text(
                enhanced_prompt,
                server_args,
                encoder_index=all_indices,
                return_attention_mask=True,
                max_length=encoded_length,
            )
        )

        for output_index in (0, 1, 3):
            positive_outputs[output_index] = [
                select_sana_video_prompt_window(tensor, max_sequence_length)
                for tensor in positive_outputs[output_index]
            ]
        positive_outputs[4] = [
            [int(value) for value in mask.sum(dim=1).tolist()]
            for mask in positive_outputs[1]
        ]

        self._append_positive_text_outputs(batch, *positive_outputs)
        if batch.do_classifier_free_guidance:
            negative_outputs = self._encode_negative_text(
                batch, server_args, all_indices
            )
            self._append_negative_text_outputs(
                batch,
                positive_outputs[0],
                *negative_outputs,
            )
        return batch


class SanaVideoPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "SanaVideoPipeline"
    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(InputValidationStage())
        self.add_stage(
            SanaVideoTextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
            "prompt_encoding_stage_primary",
        )
        self.add_standard_timestep_preparation_stage()
        self.add_standard_latent_preparation_stage()
        self.add_standard_denoising_stage()
        self.add_standard_decoding_stage()


EntryClass = SanaVideoPipeline
