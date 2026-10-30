import copy
import math
from typing import Any, List, Optional, Tuple, Union

from PIL import Image
from transformers import PreTrainedTokenizerBase
from transformers.models.pixtral.image_processing_pixtral import (
    _num_image_tokens as _get_pixtral_hf_num_image_tokens,
)

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.pixtral import (
    PixtralForConditionalGeneration,
    PixtralVisionModel,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.utils.hf_transformers.mistral_utils import mistral_common_image_encoder


class PixtralProcessor(BaseMultimodalProcessor):
    models = [PixtralVisionModel, PixtralForConditionalGeneration]
    gpu_image_decode = False  # Pixtral processes loaded image as PIL image explicitly

    PAD_TOKEN = "<pad>"
    DEFAULT_IMAGE_TOKEN = "[IMG]"

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.IM_TOKEN_ID = getattr(
            hf_config, "image_token_index", PixtralVisionModel.DEFAULT_IMAGE_TOKEN_ID
        )

        self.vision_config = hf_config.vision_config
        self.image_size = self.vision_config.image_size
        self.patch_size = self.vision_config.patch_size

        # spatial_merge_size may live on vision_config (Mistral native) or
        # on the top-level config (HF native Mistral3Config).
        self._spatial_merge_size = getattr(
            self.vision_config,
            "spatial_merge_size",
            getattr(hf_config, "spatial_merge_size", 1),
        )
        self._effective_patch_size = self.patch_size * self._spatial_merge_size

        self._processor.patch_size = self.patch_size
        if self._spatial_merge_size > 1:
            self._processor.spatial_merge_size = self._spatial_merge_size

        tokenizer = (
            _processor
            if isinstance(_processor, PreTrainedTokenizerBase)
            else _processor.tokenizer
        )
        self.image_token = getattr(_processor, "image_token", self.DEFAULT_IMAGE_TOKEN)

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.image_token,
            image_token_id=self.IM_TOKEN_ID,
        ).build(_processor)
        self._img_break_id = tokenizer.convert_tokens_to_ids("[IMG_BREAK]")
        self._img_end_id = tokenizer.convert_tokens_to_ids("[IMG_END]")
        self._grid_token_ids = {
            self.IM_TOKEN_ID,
            self._img_break_id,
            self._img_end_id,
        } - {None, tokenizer.unk_token_id}
        self._mistral_common_image_encoder = mistral_common_image_encoder(tokenizer)
        tokenizer.add_special_tokens(
            {
                "pad_token": getattr(hf_config, "pad_token", self.PAD_TOKEN),
            }
        )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        if self._is_placeholder_prompt(input_text, image_data):
            return await self._process_placeholder_prompt(image_data, input_text)

        mm_data = await self.load_mm_data(
            prompt=input_text,
            multimodal_tokens=self.mm_tokens,
            image_data=image_data,
            return_text=True,
        )
        mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
            mm_data, self.mm_tokens
        )

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            im_token_id=self.IM_TOKEN_ID,
        )

    def _is_placeholder_prompt(self, input_text, image_data) -> bool:
        return (
            self._mistral_common_image_encoder is not None
            and isinstance(input_text, list)
            and bool(image_data)
            and not self._all_mm_data_is_preprocessed(image_data)
            and input_text.count(self.IM_TOKEN_ID) == len(image_data)
            and not any(
                token_id in self._grid_token_ids and token_id != self.IM_TOKEN_ID
                for token_id in input_text
            )
        )

    async def _process_placeholder_prompt(
        self, image_data: List[Union[str, bytes]], prompt_ids: List[int]
    ) -> MultimodalProcessorOutput:
        """mistral-common rendered the chat with one [IMG] per image; expand each into
        the grid mistral-common computes for the loaded image."""
        mm_data = await self.load_mm_data(
            prompt=self.image_token * len(image_data),
            multimodal_tokens=self.mm_tokens,
            image_data=image_data,
            return_text=True,
        )
        grids = []
        for image in mm_data.images:
            cols, rows = self._mistral_common_image_encoder._image_to_num_tokens(image)
            grids.append((rows, cols))
        mm_data.images = self._fit_images_to_grids(mm_data.images, grids)
        mm_items, grid_ids, _ = await self.process_and_combine_mm_data_async(
            mm_data, self.mm_tokens
        )
        grid_ids = grid_ids.tolist()
        actual_grids = self._prompt_grids(grid_ids)
        if actual_grids != grids:
            raise ValueError(
                f"Image grids {actual_grids} do not match the expected {grids}"
            )
        input_ids = self._expand_image_placeholders(prompt_ids, grid_ids, mm_items)
        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids,
            im_token_id=self.IM_TOKEN_ID,
        )

    def _expand_image_placeholders(
        self,
        prompt_ids: List[int],
        grid_ids: List[int],
        mm_items: List[MultimodalDataItem],
    ) -> List[int]:
        image_runs = []
        run = []
        for pos, token_id in enumerate(grid_ids):
            if token_id in self._grid_token_ids:
                run.append(pos)
                if token_id == self._img_end_id:
                    image_runs.append(run)
                    run = []
        input_ids = []
        grid_to_prompt_offset = {}
        runs = iter(image_runs)
        for token_id in prompt_ids:
            if token_id != self.IM_TOKEN_ID:
                input_ids.append(token_id)
                continue
            for pos in next(runs):
                grid_to_prompt_offset[pos] = len(input_ids)
                input_ids.append(grid_ids[pos])
        for item in mm_items:
            item.offsets = [
                (grid_to_prompt_offset[s], grid_to_prompt_offset[e])
                for s, e in item.offsets
            ]
        return input_ids

    def _postprocess_mm_items_before_transport(
        self,
        mm_items: List[MultimodalDataItem],
        *,
        images: Optional[List[Any]],
    ) -> List[MultimodalDataItem]:
        if not images or len(images) <= 1:
            return mm_items

        image_items = [item for item in mm_items if item.modality == Modality.IMAGE]
        if len(image_items) == len(images):
            return mm_items
        if len(image_items) != 1:
            raise ValueError(
                "Pixtral multi-image processing expected one bundled IMAGE item or "
                f"{len(images)} split items, but found {len(image_items)}"
            )

        old_item = image_items[0]
        all_offsets = old_item.offsets
        old_feature = old_item.feature
        old_image_sizes = old_item.model_specific_data.get("image_sizes")
        image_nrows = self._get_image_nrows(images)
        if old_feature is None or len(old_feature) != len(image_nrows):
            raise ValueError(
                "Pixtral multi-image feature count does not match the number of "
                f"images: features={0 if old_feature is None else len(old_feature)}, "
                f"images={len(image_nrows)}"
            )
        if all_offsets is None or sum(image_nrows) != len(all_offsets):
            raise ValueError(
                "Pixtral image patch rows do not match the computed offsets: "
                f"rows={sum(image_nrows)}, "
                f"offsets={0 if all_offsets is None else len(all_offsets)}"
            )

        split_items = [item for item in mm_items if item.modality != Modality.IMAGE]
        offset_idx = 0
        for image_idx, num_rows in enumerate(image_nrows):
            item_offsets = all_offsets[offset_idx : offset_idx + num_rows]
            offset_idx += num_rows
            new_item = copy.copy(old_item)
            feature = old_feature[image_idx : image_idx + 1]
            if old_image_sizes is not None:
                # The HF processor pads every image of a request to the largest one;
                # the encoder derives the patch grid from the pixel shape.
                height, width = (int(v) for v in old_image_sizes[image_idx])
                feature = feature[..., :height, :width].contiguous()
            new_item.feature = feature
            new_item.offsets = item_offsets
            new_item.model_specific_data = copy.copy(old_item.model_specific_data)
            if old_image_sizes is not None:
                new_item.model_specific_data["image_sizes"] = old_image_sizes[
                    image_idx : image_idx + 1
                ]
            new_item.hash = None
            new_item.pad_value = None
            split_items.append(new_item)
        if offset_idx != len(all_offsets):
            raise ValueError(
                "Pixtral multi-image split did not consume every offset: "
                f"consumed={offset_idx}, offsets={len(all_offsets)}"
            )
        return split_items

    def _prompt_grids(self, prompt_ids: List[int]) -> List[Tuple[int, int]]:
        grids, rows, cols, row_width = [], 0, 0, None
        for token_id in prompt_ids:
            if token_id == self.IM_TOKEN_ID:
                cols += 1
            elif token_id in (self._img_break_id, self._img_end_id):
                rows += 1
                row_width = cols if row_width is None else row_width
                cols = 0
                if token_id == self._img_end_id:
                    grids.append((rows, row_width))
                    rows, row_width = 0, None
        return grids

    def _fit_images_to_grids(
        self, images: List[Any], grids: List[Tuple[int, int]]
    ) -> List[Any]:
        """mistral-common rounds the downscaled size where the HF processor floors,
        which can drop a patch row or column; resize to mistral-common's grid."""
        patch = self._effective_patch_size
        return [
            (
                image
                if self._hf_grid(image) == grid
                else image.resize(
                    (grid[1] * patch, grid[0] * patch), Image.Resampling.BICUBIC
                )
            )
            for image, grid in zip(images, grids)
        ]

    def _hf_grid(self, image: Any) -> Tuple[int, int]:
        width, height = image.size
        ratio = max(width / self.image_size, height / self.image_size)
        if ratio > 1:
            width = int(math.floor(width / ratio))
            height = int(math.floor(height / ratio))
        return tuple(
            _get_pixtral_hf_num_image_tokens(
                (height, width),
                (self._effective_patch_size, self._effective_patch_size),
            )
        )

    def _get_image_nrows(self, images: List[Any]) -> List[int]:
        return [self._hf_grid(image)[0] for image in images]
