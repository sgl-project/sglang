import asyncio
from typing import List, Union

import torch

from sglang.srt.managers.image_processor import BaseImageProcessor
from sglang.srt.managers.image_processors.base_image_processor import (
    get_global_processor,
)
from sglang.srt.models.minicpmv import MiniCPMV


class MiniCPMVImageProcessor(BaseImageProcessor):
    models = [MiniCPMV]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.IMAGE_TOKEN = "(<image>./</image>)"

    @staticmethod
    def _process_images_task(images, input_text):
        processor = get_global_processor()
        result = processor.__call__(text=input_text, images=images, return_tensors="pt")
        return {
            "input_ids": result.input_ids,
            "pixel_values": result.pixel_values,
            "tgt_sizes": result.tgt_sizes,
        }

    async def _process_images(self, images, input_text):
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            image_inputs = await loop.run_in_executor(
                self.executor,
                MiniCPMVImageProcessor._process_images_task,
                images,
                input_text,
            )
        else:
            image_inputs = self._processor(
                images=images, text=input_text, return_tensors="pt"
            )

        return image_inputs

    async def process_images_async(
        self,
        image_data: List[Union[str, bytes]],
        input_ids,
        request_obj,
        max_req_input_len,
    ):
        if not image_data:
            return None
        if not isinstance(image_data, list):
            image_data = [image_data]

        base_output = self.load_images(
            input_ids=input_ids,
            image_data=image_data,
            image_token=self.IMAGE_TOKEN,
            max_req_input_len=max_req_input_len,
        )
        if base_output is None:
            return None

        if len(base_output.all_frames) == 0:
            return None
        res = await self._process_images(
            images=base_output.all_frames, input_text=base_output.input_text
        )

        # Collect special token ids
        tokenizer = self._processor.tokenizer
        im_start_id = tokenizer.im_start_id
        im_token_id = tokenizer.unk_token_id
        im_end_id = tokenizer.im_end_id
        if tokenizer.slice_start_id:
            slice_start_id = tokenizer.slice_start_id
            slice_end_id = tokenizer.slice_end_id

        pixel_values = res["pixel_values"]
        tgt_sizes = res["tgt_sizes"]

        if not isinstance(pixel_values, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of pixel values. " f"Got type: {type(pixel_values)}"
            )

        if not isinstance(tgt_sizes, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of target sizes. " f"Got type: {type(tgt_sizes)}"
            )

        if len(pixel_values) != len(tgt_sizes):
            raise ValueError(
                "Inconsistent batch lengths, found: "
                f"{len(pixel_values)} vs. {len(tgt_sizes)}"
            )

        # tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
        # tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)
        pixel_values_flat: List[torch.Tensor] = []
        tgt_sizes_flat: List[torch.Tensor] = []
        for pixel_b, tgt_b in zip(pixel_values, tgt_sizes):
            # per image
            if len(pixel_b) != len(tgt_b):
                raise ValueError(
                    "Inconsistent N lengths, found: " f"{len(pixel_b)} vs {len(tgt_b)}"
                )
            for pixel_n, tgt_n in zip(pixel_b, tgt_b):
                # per patch
                pixel_values_flat += [pixel_n]
                tgt_sizes_flat += [tgt_n]

        pixel_values = pixel_values_flat
        tgt_sizes = torch.stack(tgt_sizes_flat)
        return {
            "input_ids": res["input_ids"].flatten().tolist(),
            "pixel_values": pixel_values,
            "tgt_sizes": tgt_sizes,
            "image_hashes": base_output.image_hashes,
            "modalities": request_obj.modalities or ["image"],
            "im_start_id": im_start_id,
            "im_token_id": im_token_id,
            "im_end_id": im_end_id,
            "slice_start_id": slice_start_id,
            "slice_end_id": slice_end_id,
        }
