# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

TOKEN_LENGTH = 37698
HIDDEN_STATE_SKIP_LAYER = 0

PROMPT_TEMPLATE = (
    "<|im_start|>system\nGiven a user input that may include a text prompt alone, "
    "a text prompt with an image reference, or a text prompt with a video reference "
    'or a video reference alone, generate an "Enhanced prompt" that provides detailed '
    "visual descriptions suitable for video generation. Evaluate the level of detail "
    "in the user's input: if it is simple, enrich it by adding specifics about colors, "
    "shapes, sizes, textures, lighting, motion dynamics, camera movement, temporal "
    "progression, and spatial relationships to create vivid, concrete, and temporally "
    "coherent scenes to create vivid and concrete scenes. Please generate only the "
    "enhanced description for the prompt below and avoid including any additional "
    "commentary or evaluations:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
IMG_PROMPT_TEMPLATE = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PROMPT_TEMPLATE = "<|vision_start|><|video_pad|><|vision_end|>"


class LingBotVideoTextEncodingStage(TextEncodingStage):
    """Qwen3-VL prompt/negative encoding for LingBot-Video MoE (T2V, base)."""

    def __init__(self, text_encoders, tokenizers, transformer):
        super().__init__(text_encoders, tokenizers)
        self.transformer = transformer
        self.token_length = TOKEN_LENGTH
        self.hidden_state_skip_layer = HIDDEN_STATE_SKIP_LAYER
        self.prompt_template = PROMPT_TEMPLATE
        self._crop_start: int | None = None

    @staticmethod
    def check_inputs(height: int, width: int, num_frames: int) -> None:
        if num_frames != 1 and (num_frames - 1) % 4 != 0:
            raise ValueError(f"`num_frames` must be 1 or 4n+1, got {num_frames}.")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`height` and `width` must be multiples of 16, got {height}x{width}."
            )

    @staticmethod
    def apply_text_to_template(text: str, template: str = PROMPT_TEMPLATE) -> str:
        return template.format(text)

    def _compute_crop_start(self) -> int:
        processor = self.tokenizers[0]
        if self._crop_start is None:
            marker = "<|USER_INPUT_MARKER|>"
            marked = self.prompt_template.format(marker)
            marker_pos = marked.find(marker)
            if marker_pos < 0:
                self._crop_start = 0
            else:
                prefix = processor(
                    text=marked[:marker_pos],
                    images=None,
                    videos=None,
                    return_tensors="pt",
                )
                self._crop_start = int(prefix["input_ids"].shape[1])
        return self._crop_start

    def _build_prompt_inputs(self, prompt: str | list[str]):
        processor = self.tokenizers[0]
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        texts = [
            self.apply_text_to_template(text, self.prompt_template) for text in prompts
        ]
        return processor(
            text=texts,
            images=None,
            videos=None,
            video_metadata=None,
            do_resize=False,
            truncation=True,
            max_length=self.token_length,
            padding="longest",
            return_tensors="pt",
        )

    @torch.no_grad()
    def _encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device,
        dtype: torch.dtype,
    ):
        text_encoder = self.text_encoders[0]
        if text_encoder is None or self.tokenizers[0] is None:
            raise ValueError(
                "`text_encoder` and `processor` are required for encode_prompt()."
            )

        inputs = self._build_prompt_inputs(prompt)
        inputs = inputs.to(device)
        outputs = text_encoder(
            **inputs,
            output_hidden_states=self.hidden_state_skip_layer is not None,
        )
        if self.hidden_state_skip_layer is not None:
            prompt_embeds = outputs.hidden_states[-(self.hidden_state_skip_layer + 1)]
        else:
            prompt_embeds = outputs.last_hidden_state

        prompt_mask = inputs["attention_mask"]
        crop_start = self._compute_crop_start()
        if crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_mask = prompt_mask[:, crop_start:]

        # B=1: drop right padding before DiT inference.
        if prompt_embeds.shape[0] == 1:
            true_len = int(prompt_mask[0].sum().item())
            prompt_embeds = prompt_embeds[:, :true_len]
            prompt_mask = prompt_mask[:, :true_len]

        return prompt_embeds.to(dtype=dtype), prompt_mask

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        dtype = next(self.transformer.parameters(), torch.tensor([])).dtype
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            dtype = torch.bfloat16

        self.check_inputs(int(batch.height), int(batch.width), int(batch.num_frames))

        prompt_embeds, prompt_mask = self._encode_prompt(batch.prompt, device, dtype)
        batch.prompt_embeds = [prompt_embeds]
        batch.prompt_attention_mask = prompt_mask

        if batch.do_classifier_free_guidance:
            negative_embeds, negative_mask = self._encode_prompt(
                batch.negative_prompt, device, dtype
            )
            batch.negative_prompt_embeds = [negative_embeds]
            batch.negative_attention_mask = negative_mask
        return batch
