"""Text+image (VL) prompt encoding stage for LongCat-Image-Edit (I2I).

Mirrors diffusers ``LongCatImageEditPipeline._encode_prompt``: the condition
image (already resized to the output resolution by ``InputValidationStage``)
is downscaled by 2x and fed to the Qwen2.5-VL vision tower; the prompt body is
quote-aware tokenized to a fixed 512 tokens; the edit system prefix has its
``<|image_pad|>`` placeholder expanded to the per-image token count. The
resulting hidden states are sliced from the ``<|vision_start|>`` token through
the 512-token body, so the DiT conditioning includes the VL image tokens.
"""

import PIL.Image
import torch

from sglang.multimodal_gen.configs.pipeline_configs.longcat_image import (
    _tokenize_prompt_for_encode,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

IMAGE_TOKEN = "<|image_pad|>"

# Copied from diffusers LongCatImageEditPipeline.prompt_template_encode_prefix/suffix.
PROMPT_TEMPLATE_ENCODE_PREFIX = (
    "<|im_start|>system\nAs an image editing expert, first analyze the content and "
    "attributes of the input image(s). Then, based on the user's editing instructions, "
    "clearly and precisely determine how to modify the given image(s), ensuring that "
    "only the specified parts are altered and all other aspects remain consistent with "
    "the original(s).<|im_end|>\n<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>"
)
PROMPT_TEMPLATE_ENCODE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


class LongCatImageEditTextEncodingStage(PipelineStage):
    """Encode the edit instruction together with the reference image.

    The stage fills ``batch.prompt_embeds`` (and negative counterparts under
    CFG) with hidden states sliced as ``[vision_start ... vision_end, 512-token
    body]``. Both branches share the same VL image inputs, matching the
    diffusers reference which encodes positive and negative prompts against
    the same ``prompt_image``.
    """

    deduplicated_output_fields = (
        "prompt_embeds",
        "negative_prompt_embeds",
        "prompt_embeds_mask",
        "negative_prompt_embeds_mask",
        "prompt_seq_lens",
        "negative_prompt_seq_lens",
    )

    def __init__(
        self,
        text_encoder,
        tokenizer,
        text_processor,
        text_encoder_dtype: torch.dtype,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.text_processor = text_processor
        self.text_encoder_dtype = text_encoder_dtype
        self._suffix_ids: list[int] | None = None

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(
                stage_name,
                "text_encoder",
                target_dtype=self.text_encoder_dtype,
                memory_intensive=True,
            ),
        ]

    def _get_suffix_ids(self) -> list[int]:
        if self._suffix_ids is None:
            self._suffix_ids = self.tokenizer(
                PROMPT_TEMPLATE_ENCODE_SUFFIX, add_special_tokens=False
            )["input_ids"]
        return self._suffix_ids

    def _build_expanded_prefix_ids(
        self, image_grid_thw: torch.Tensor
    ) -> tuple[list[int], int]:
        """Expand ``<|image_pad|>`` to the per-image token count and tokenize.

        Returns (prefix_token_ids, prefix_len) where prefix_len is the index of
        ``<|vision_start|>`` — the slice start for the DiT conditioning.
        """
        merge_length = self.text_processor.image_processor.merge_size**2
        num_image_tokens = int(image_grid_thw.prod().item()) // merge_length
        text = PROMPT_TEMPLATE_ENCODE_PREFIX
        while IMAGE_TOKEN in text:
            text = text.replace(IMAGE_TOKEN, "<|placeholder|>" * num_image_tokens, 1)
        text = text.replace("<|placeholder|>", IMAGE_TOKEN)

        prefix_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        prefix_len = prefix_ids.index(vision_start_id)
        return prefix_ids, prefix_len

    def _encode(
        self,
        prompt: list[str],
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        prefix_ids: list[int],
        prefix_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        body = _tokenize_prompt_for_encode(prompt, self.tokenizer)
        suffix_ids = self._get_suffix_ids()
        suffix_len = len(suffix_ids)
        batch_size = body.input_ids.size(0)

        prefix_ids_t = (
            torch.tensor(prefix_ids, dtype=body.input_ids.dtype)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        suffix_ids_t = (
            torch.tensor(suffix_ids, dtype=body.input_ids.dtype)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        prefix_mask_t = torch.ones(
            batch_size, len(prefix_ids), dtype=body.attention_mask.dtype
        )
        suffix_mask_t = torch.ones(
            batch_size, suffix_len, dtype=body.attention_mask.dtype
        )

        input_ids = torch.cat((prefix_ids_t, body.input_ids, suffix_ids_t), dim=-1).to(
            device
        )
        attention_mask = torch.cat(
            (prefix_mask_t, body.attention_mask, suffix_mask_t), dim=-1
        ).to(device)

        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states[-1]
        # Keep [vision_start ... vision_end, 512-token body].
        return hidden_states[:, prefix_len:-suffix_len, :]

    @staticmethod
    def _all_ones_conditioning(prompt_embeds: torch.Tensor):
        batch_size, seq_len = prompt_embeds.shape[:2]
        mask = torch.ones(
            batch_size, seq_len, dtype=torch.bool, device=prompt_embeds.device
        )
        return mask, [seq_len] * batch_size

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()

        image = batch.condition_image
        if isinstance(image, list):
            if len(image) != 1:
                raise ValueError(
                    "LongCat-Image-Edit supports exactly one condition image, "
                    f"got {len(image)}."
                )
            image = image[0]
        if image is None:
            # Mirrors ImageEncodingStage: nothing to encode without an image
            # (e.g. warmup requests). Downstream stages will validate inputs.
            logger.debug(
                "LongCatImageEditTextEncodingStage skipped: no condition image."
            )
            return batch

        prompt = batch.prompt if isinstance(batch.prompt, list) else [batch.prompt]
        if len(prompt) != 1:
            raise ValueError(
                "LongCat-Image-Edit supports a single prompt per request, got "
                f"{len(prompt)}."
            )

        # The VL tower sees the condition image at half the output resolution
        # (matching diffusers: image_processor.resize(image, h // 2, w // 2)).
        prompt_image = image.resize(
            (int(batch.width) // 2, int(batch.height) // 2),
            PIL.Image.Resampling.LANCZOS,
        )
        vl_inputs = self.text_processor.image_processor(
            images=prompt_image, return_tensors="pt"
        )
        image_grid_thw = vl_inputs["image_grid_thw"].to(device)
        prefix_ids, prefix_len = self._build_expanded_prefix_ids(image_grid_thw)

        with self.use_declared_component(
            component_name="text_encoder",
            module=self.text_encoder,
        ) as text_encoder:
            assert text_encoder is not None
            self.text_encoder = text_encoder

            pixel_values = vl_inputs["pixel_values"].to(
                device=device, dtype=self.text_encoder_dtype
            )

            prompt_embeds = self._encode(
                prompt, pixel_values, image_grid_thw, prefix_ids, prefix_len, device
            )
            batch.prompt_embeds.append(prompt_embeds)
            mask, seq_lens = self._all_ones_conditioning(prompt_embeds)
            batch.prompt_embeds_mask = [mask]
            batch.prompt_seq_lens = [seq_lens]

            if batch.do_classifier_free_guidance:
                negative_prompt = batch.negative_prompt or ""
                if isinstance(negative_prompt, list):
                    negative_prompt = negative_prompt[:1]
                else:
                    negative_prompt = [negative_prompt]
                negative_prompt_embeds = self._encode(
                    negative_prompt,
                    pixel_values,
                    image_grid_thw,
                    prefix_ids,
                    prefix_len,
                    device,
                )
                batch.negative_prompt_embeds.append(negative_prompt_embeds)
                neg_mask, neg_seq_lens = self._all_ones_conditioning(
                    negative_prompt_embeds
                )
                batch.negative_prompt_embeds_mask = [neg_mask]
                batch.negative_prompt_seq_lens = [neg_seq_lens]

        return batch
