from __future__ import annotations

import asyncio
from collections.abc import Mapping

import torch

from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalProcessorOutput
from sglang.srt.multimodal.cache import resolve_multimodal_item_hash
from sglang.srt.multimodal.encoder_preprocessing import EncoderPreprocessOutput
from sglang.srt.multimodal.inkling2 import Inkling2ImageProcessor
from sglang.srt.multimodal.media_io import download_remote_media
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.parser.inkling_tokenizer import IMAGE_TOKEN_ID


class Inkling2Processor:
    def __init__(self, tokenizer, image_processor: Inkling2ImageProcessor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    @classmethod
    def from_pretrained(cls, path, *, tokenizer_path=None, revision=None, **kwargs):
        from sglang.srt.hf_transformers.tokenizer import get_tokenizer

        return cls(
            get_tokenizer(tokenizer_path or path, revision=revision, **kwargs),
            Inkling2ImageProcessor.from_pretrained(path, revision=revision),
        )


def _resolve_image(image):
    if isinstance(image, Mapping):
        image = image["url"]
    elif hasattr(image, "url"):
        image = image.url
    if isinstance(image, str) and image.startswith(("http://", "https://")):
        return download_remote_media(image, timeout=30)
    return image


def build_encoder_inputs(images, modality: Modality) -> EncoderPreprocessOutput:
    """EPD encoder preprocessing: the colocated patches, one item per image."""
    if modality != Modality.IMAGE:
        raise ValueError("Inkling2 encoders support images only")
    image_processor = Inkling2ImageProcessor()
    items, grids, features = [], [], []
    for image in images:
        encoded = image_processor.preprocess([image])
        patches = encoded["vision_patches_bthwc"]
        grid = encoded["vision_grid_thw"]
        items.append(
            MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=patches,
                model_specific_data={
                    "vision_grid_thw": grid,
                    "vision_original_thw": encoded["vision_original_thw"],
                },
            )
        )
        grids.append(grid)
        features.append(patches)
    return EncoderPreprocessOutput(
        {"image_grid_thw": torch.cat(grids), "pixel_values": features},
        mm_items=items,
    )


class Inkling2MultimodalProcessor(BaseMultimodalProcessor):
    gpu_image_decode = False
    prefer_tokenized_input = True

    def __init__(self, hf_config, server_args, processor, *args, **kwargs):
        super().__init__(hf_config, server_args, processor, *args, **kwargs)
        if hf_config.vision_config is None:
            raise ValueError("Inkling2 image requests require vision_config")
        self.image_processor = processor.image_processor
        self.IMAGE_TOKEN_ID = IMAGE_TOKEN_ID

    def assemble(self, input_ids: list[int], image_data=None):
        images = list(image_data or [])
        count = input_ids.count(self.IMAGE_TOKEN_ID)
        if count != len(images):
            raise ValueError(
                f"Image placeholder count {count} does not match image count {len(images)}"
            )
        features = self.image_processor.preprocess(
            [_resolve_image(image) for image in images]
        )
        items = []
        output = []
        patch_offset = 0
        for token in input_ids:
            if token != self.IMAGE_TOKEN_ID:
                output.append(token)
                continue
            index = len(items)
            num_tokens = features["num_tokens"][index]
            start = len(output)
            output.extend([self.IMAGE_TOKEN_ID] * num_tokens)
            patches = features["vision_patches_bthwc"][
                patch_offset : patch_offset + num_tokens
            ]
            grid = features["vision_grid_thw"][index : index + 1]
            items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=patches,
                    hash=resolve_multimodal_item_hash(feature=[patches, grid]),
                    offsets=[(start, len(output) - 1)],
                    model_specific_data={
                        "vision_grid_thw": grid,
                        "vision_original_thw": features["vision_original_thw"][
                            index : index + 1
                        ],
                    },
                )
            )
            patch_offset += num_tokens
        return MultimodalProcessorOutput(
            input_ids=output, mm_items=items, im_token_id=self.IMAGE_TOKEN_ID
        )

    def get_mm_data(self, prompt, embeddings, **kwargs):
        """Language-only EPD: expand placeholders from encoder grids like ``assemble``."""
        if not isinstance(prompt, list):
            raise ValueError("Inkling2 EPD expects tokenized prompts")
        img_grid_thw = kwargs.get("img_grid_thw")
        grids = [
            torch.as_tensor(grid, dtype=torch.int64).reshape(1, 3)
            for grid in ([] if img_grid_thw is None else img_grid_thw)
        ]
        count = prompt.count(self.IMAGE_TOKEN_ID)
        if count != len(grids):
            raise ValueError(
                f"Image placeholder count {count} does not match encoded image count {len(grids)}"
            )
        image_embeddings = embeddings.get(Modality.IMAGE)
        items = []
        output = []
        row = 0
        for token in prompt:
            if token != self.IMAGE_TOKEN_ID:
                output.append(token)
                continue
            grid = grids[len(items)]
            num_tokens = int(grid.prod())
            start = len(output)
            output.extend([self.IMAGE_TOKEN_ID] * num_tokens)
            items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    offsets=[(start, len(output) - 1)],
                    precomputed_embeddings=image_embeddings[row : row + num_tokens],
                    model_specific_data={"vision_grid_thw": grid},
                )
            )
            row += num_tokens
        return MultimodalProcessorOutput(
            input_ids=output, mm_items=items, im_token_id=self.IMAGE_TOKEN_ID
        )

    async def process_mm_data_async(
        self,
        image_data=None,
        audio_data=None,
        video_data=None,
        input_text="",
        request_obj=None,
        **kwargs,
    ):
        if audio_data or video_data or getattr(request_obj, "video_data", None):
            raise ValueError("Inkling2 supports text and images only")
        input_ids = getattr(request_obj, "input_ids", None)
        if input_ids is None:
            input_ids = (
                self._tokenizer.encode(input_text, add_special_tokens=False)
                if isinstance(input_text, str)
                else input_text
            )
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.flatten().tolist()
        return await asyncio.get_running_loop().run_in_executor(
            self.io_executor, self.assemble, list(input_ids), image_data
        )
