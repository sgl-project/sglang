from typing import List, Tuple

import torch

from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalProcessorOutput
from sglang.srt.models.molmo2 import Molmo2ForConditionalGeneration
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.molmo2 import IMAGE_PLACEHOLDER
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)


class Molmo2ImageProcessor(BaseMultimodalProcessor):
    models = [Molmo2ForConditionalGeneration]
    gpu_image_decode = False

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.image_patch_id = hf_config.image_patch_id
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=IMAGE_PLACEHOLDER,
            image_token_id=self.image_patch_id,
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ):
        if (request_obj is not None and request_obj.video_data) or audio_data:
            raise ValueError("Molmo2 supports image inputs only")
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )
        mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
            base_output, self.mm_tokens
        )
        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.image_patch_id,
        )

    def process_and_combine_mm_data(
        self,
        base_output: BaseMultiModalProcessorOutput,
        mm_tokens: MultimodalSpecialTokens,
        processor=None,
        **kwargs,
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:
        if any(isinstance(image, dict) for image in base_output.images):
            raise ValueError(
                "Molmo2 takes raw images; processor-output and precomputed "
                "embedding inputs are not supported"
            )
        ret = self.process_mm_data(
            input_text=base_output.input_text,
            images=base_output.images,
            processor=processor,
            **kwargs,
        )
        input_ids = ret["input_ids"].flatten()
        if not base_output.images:
            return [], input_ids, ret
        mm_items = _split_per_image(
            processor_output=ret,
            runs=self.get_mm_items_offset(input_ids, self.image_patch_id),
        )
        return (
            self._finalize_mm_items(mm_items, images=base_output.images),
            input_ids,
            ret,
        )


def _split_per_image(
    *,
    processor_output: dict,
    runs: List[Tuple[int, int]],
) -> List[MultimodalDataItem]:
    pixel_values = processor_output["pixel_values"]
    pooling = processor_output["image_token_pooling"]
    patches_per_crop = pixel_values.shape[1]

    items = []
    run_index = crop_start = token_start = 0
    for grid, num_crops in zip(
        processor_output["image_grids"].tolist(),
        processor_output["image_num_crops"].tolist(),
    ):
        low_res_h, low_res_w, high_res_h, high_res_w = grid
        num_tokens = low_res_h * low_res_w + high_res_h * high_res_w
        image_runs = []
        covered = 0
        while covered < num_tokens and run_index < len(runs):
            start, end = runs[run_index]
            image_runs.append((start, end))
            covered += end - start + 1
            run_index += 1
        if covered != num_tokens:
            raise ValueError(
                f"Molmo2 image expects {num_tokens} patch tokens, "
                f"the prompt holds {covered}"
            )

        image_pooling = pooling[token_start : token_start + num_tokens]
        patch_offset = crop_start * patches_per_crop
        items.append(
            MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=pixel_values[crop_start : crop_start + num_crops],
                offsets=image_runs,
                model_specific_data={
                    "image_token_pooling": torch.where(
                        image_pooling >= 0, image_pooling - patch_offset, image_pooling
                    )
                },
            )
        )
        crop_start += num_crops
        token_start += num_tokens

    if run_index != len(runs) or token_start != len(pooling):
        raise ValueError(
            "Molmo2 patch tokens in the prompt do not match the processed images"
        )
    return items
