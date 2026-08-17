"""Processor class for MinerU-Diffusion."""

from typing import Union

import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import load_image
from transformers.processing_utils import ProcessorMixin


class MinerUDiffusionHFProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    optional_attributes = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    special_tokens = ["<|image_pad|>", "<|vision_start|>", "<|vision_end|>"]

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        if chat_template == "auto":
            chat_template = tokenizer.chat_template
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.downsample_size = kwargs.pop("downsample_size", 2)
        self.image_token = kwargs.pop(
            "image_token",
            getattr(tokenizer, "image_token", "<|image_pad|>"),
        )
        self.special_tokens = kwargs.pop(
            "special_tokens",
            ["<|image_pad|>", "<|vision_start|>", "<|vision_end|>"],
        )

        try:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": self.special_tokens},
                replace_additional_special_tokens=False,
            )
        except TypeError:
            self.tokenizer.add_special_tokens({"additional_special_tokens": self.special_tokens})
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None) is not None
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )

    def _process_images(self, images) -> dict:
        loaded_images = [load_image(image) for image in images]
        if loaded_images:
            return self.image_processor(images=loaded_images, return_tensors="pt")

        patch_size = getattr(self.image_processor, "patch_size", 14)
        return {
            "pixel_values": torch.zeros((0, 3 * 2 * (patch_size**2)), dtype=torch.float32),
            "image_grid_thw": torch.zeros((0, 3), dtype=torch.int64),
        }

    def _expand_image_tokens(self, text, image_grid_thw):
        image_token_lengths = (image_grid_thw.prod(dim=1) // (self.downsample_size**2)).tolist()
        image_index = 0
        new_text = []
        for t in text:
            parts = t.split(self.image_token)
            expanded_t = parts[0]
            for i in range(1, len(parts)):
                if image_index >= len(image_token_lengths):
                    raise ValueError("Wrong image token count, more image tokens than processed images.")
                expanded_t += self.image_token * image_token_lengths[image_index]
                expanded_t += parts[i]
                image_index += 1
            new_text.append(expanded_t)
        if image_index != len(image_token_lengths):
            raise ValueError(
                "Wrong image token count, "
                f"image_token_count({image_index}) != image_count({len(image_token_lengths)})"
            )
        return new_text

    @staticmethod
    def _count_image_embeds(image_grid_thw, downsample_size):
        return int((image_grid_thw.prod(dim=1) // (downsample_size**2)).sum().item())

    def _validate_image_inputs(self, input_ids, image_grid_thw):
        if isinstance(input_ids, torch.Tensor):
            image_token_count = torch.count_nonzero(input_ids == self.image_token_id).item()
        else:
            image_token_count = sum(row.count(self.image_token_id) for row in input_ids)
        image_embed_count = self._count_image_embeds(image_grid_thw, self.downsample_size)
        if image_token_count != image_embed_count:
            raise ValueError(
                "Wrong image embed token count, "
                f"image_embed_token_count({image_token_count}) != image_embed_count({image_embed_count})"
            )

    def __call__(self, images=None, text=None, **kwargs) -> BatchFeature:
        if images is None:
            images = []
        return_tensors = kwargs.pop("return_tensors", None)
        image_inputs = self._process_images(images)
        if text is None:
            return BatchFeature(data=image_inputs, tensor_type=return_tensors)

        if not isinstance(text, list):
            text = [text]
        text = self._expand_image_tokens(list(text), image_inputs["image_grid_thw"])
        text_inputs = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
        self._validate_image_inputs(text_inputs["input_ids"], image_inputs["image_grid_thw"])
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(
        self,
        generated_outputs: Union[list, torch.Tensor],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
        **kwargs,
    ):
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


# Backward-compatible alias for external imports using previous class name.
MinerUDiffusionProcessor = MinerUDiffusionHFProcessor
