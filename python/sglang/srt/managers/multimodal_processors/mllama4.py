from typing import List, Mapping, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import Llama4Processor
from transformers.image_utils import SizeDict
from transformers.models.llama4.image_processing_llama4 import (
    find_supported_resolutions,
    get_best_fit,
)

from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.mllama4 import Llama4ForConditionalGeneration
from sglang.srt.utils import load_image


class Mllama4ImageProcessor(BaseMultimodalProcessor):
    models = [Llama4ForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.vision_config = hf_config.vision_config
        self.text_config = hf_config.text_config
        self.multimodal_tokens = MultimodalSpecialTokens(
            image_token=_processor.image_token
        )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        max_req_input_len=None,
        *args,
        **kwargs,
    ):
        if not image_data:
            return None

        if isinstance(input_text, list):
            assert len(input_text) and isinstance(input_text[0], int)
            input_text = self._processor.tokenizer.decode(input_text)

        # Process images and text using the base processor's load_mm_data method
        processed_data = self.load_mm_data(
            prompt=input_text,
            multimodal_tokens=self.multimodal_tokens,
            max_req_input_len=max_req_input_len or 4096,
            image_data=image_data,
            return_text=True,
        )

        # Process the images using the processor
        processor = Llama4Processor.from_pretrained(
            self.server_args.model_path, **kwargs
        )

        # Process the prompt and images
        image_inputs = processor(
            text=processed_data.input_text,
            images=processed_data.images,
            return_tensors="pt",
        )

        # Handle image resolutions and aspect ratios
        if "pixel_values" in image_inputs:
            image_processor = processor.image_processor
            tokenizer = self._processor.tokenizer

            # Calculate tile size and find supported resolutions
            tile_size = self.vision_config.image_size
            max_num_tiles = getattr(self.vision_config, "max_patches", 1)

            possible_resolutions = find_supported_resolutions(
                max_num_chunks=max_num_tiles,
                patch_size=SizeDict(height=tile_size, width=tile_size),
            )

            # Find best fit for each image
            best_fit_sizes = [
                get_best_fit(
                    (image.size[1], image.size[0]),  # (height, width)
                    torch.tensor(possible_resolutions),
                    resize_to_max_canvas=image_processor.resize_to_max_canvas,
                )
                for image in processed_data.images
            ]

            # Calculate aspect ratios and patches per image
            aspect_ratios = [
                (image_size[0] // tile_size, image_size[1] // tile_size)
                for image_size in best_fit_sizes
            ]

            patches_per_image = [
                1 if r_h * r_w == 1 else 1 + r_h * r_w for (r_h, r_w) in aspect_ratios
            ]

            # Add to image_inputs
            image_inputs["aspect_ratios"] = aspect_ratios
            image_inputs["patches_per_image"] = torch.tensor(patches_per_image)

            # Process embed_is_patch
            vocab = tokenizer.get_vocab()
            patch_id = vocab.get(processor.img_patch_token, -1)
            image_end_id = vocab.get(processor.end_of_img_token, -1)

            if patch_id != -1 and image_end_id != -1:
                input_ids = image_inputs["input_ids"].view(-1)

                # Remove BOS token if present
                if input_ids.size(0) > 0 and input_ids[0] == tokenizer.bos_token_id:
                    input_ids = input_ids[1:]

                # Find image end indices and split input_ids
                image_end_indices = (input_ids == image_end_id).nonzero().view(-1)

                if image_end_indices.size(0) > 0:
                    # Split at image boundaries
                    split_indices = (image_end_indices + 1)[:-1]
                    split_input_ids = torch.tensor_split(input_ids, split_indices)
                    split_input_ids = [x for x in split_input_ids if x.numel() > 0]

                    # Create embed_is_patch for each image
                    embed_is_patch = []
                    for per_image_input_ids in split_input_ids:
                        embed_is_patch.append(per_image_input_ids == patch_id)

                    image_inputs["embed_is_patch"] = embed_is_patch

        # Convert to the format expected by SGLang
        image_inputs["input_ids"] = image_inputs["input_ids"].tolist()[0]

        # Add metadata for image processing
        image_inputs["mm_items"] = [
            MultimodalDataItem(
                pixel_values=image_inputs["pixel_values"],
                modality=Modality.IMAGE,
                # Add additional metadata needed for Llama4 vision processing
                embed_is_patch=image_inputs.get("embed_is_patch", None),
                aspect_ratios=image_inputs.get("aspect_ratios", None),
                patches_per_image=image_inputs.get("patches_per_image", None),
            )
        ]

        return image_inputs

    def get_patch_per_chunk(self):
        """Calculate patches per chunk based on vision config"""
        image_size = self.vision_config.image_size
        patch_size = self.vision_config.patch_size

        assert (
            image_size % patch_size == 0
        ), f"chunk size {image_size} should be multiple of patch_size {patch_size}"

        ds_ratio = int(round(1.0 / (self.vision_config.pixel_shuffle_ratio**2)))
        return (image_size // patch_size) ** 2 // ds_ratio
