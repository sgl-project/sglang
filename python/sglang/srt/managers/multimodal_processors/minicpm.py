import asyncio
from typing import List, Union

import torch

from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
    get_global_processor,
)
from sglang.srt.models.minicpmo import MiniCPMO
from sglang.srt.models.minicpmv import MiniCPMV


# Compatible with both 'O' and 'V'
class MiniCPMMultimodalProcessor(BaseMultimodalProcessor):
    models = [MiniCPMV, MiniCPMO]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.image_token = "(<image>./</image>)"
        self.audio_token = "(<audio>./</audio>)"

    @staticmethod
    def _process_data_task(input_text, images=None, audios=None):

        if isinstance(images, list) and len(images) == 0:
            images = None
        if isinstance(audios, list) and len(audios) == 0:
            audios = None
        result = get_global_processor().__call__(
            text=input_text,
            images=images,
            audios=audios,
            return_tensors="pt",
            chunk_input=True,
        )
        return {
            "input_ids": result.input_ids,
            "pixel_values": getattr(result, "pixel_values", None),
            "tgt_sizes": getattr(result, "tgt_sizes", None),
            "audio_features": getattr(result, "audio_features", None),
            "audio_feature_lens": getattr(result, "audio_feature_lens", None),
            "audio_bounds": getattr(result, "audio_bounds", None),
        }

    async def _process_data(self, images, input_text, audios=None):
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            multimodal_data_inputs = await loop.run_in_executor(
                self.executor,
                MiniCPMMultimodalProcessor._process_data_task,
                input_text,
                images,
                audios,
            )
        else:
            multimodal_data_inputs = self._processor(
                images=images, text=input_text, audios=audios, return_tensors="pt"
            )

        return multimodal_data_inputs

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_ids,
        request_obj,
        max_req_input_len,
    ):
        audio_data = request_obj.audio_data
        if not image_data and not audio_data:
            return None
        if not isinstance(image_data, list):
            image_data = [image_data]
        if not isinstance(audio_data, list):
            audio_data = [audio_data]

        base_output = self.load_mm_data(
            input_ids=input_ids,
            max_req_input_len=max_req_input_len,
            audio_data=audio_data,
            image_data=image_data,
            multimodal_tokens=MultimodalSpecialTokens(
                image_token=self.image_token, audio_token=self.audio_token
            ),
        )
        if base_output is None:
            return None

        res = await self._process_data(
            images=base_output.images,
            input_text=base_output.input_text,
            audios=base_output.audios,
        )

        # Collect special token ids
        tokenizer = self._processor.tokenizer
        slice_start_id, slice_end_id, audio_start_id, audio_end_id = (
            None,
            None,
            None,
            None,
        )
        if tokenizer.slice_start_id:
            slice_start_id = tokenizer.slice_start_id
            slice_end_id = tokenizer.slice_end_id
        if hasattr(tokenizer, "audio_start_id"):
            audio_start_id = tokenizer.audio_start_id
            audio_end_id = tokenizer.audio_end_id

        im_token_id = tokenizer.unk_token_id
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

        pixel_values_flat: List[torch.Tensor] = []
        tgt_sizes_flat: List[torch.Tensor] = []
        for pixel_b, tgt_b in zip(pixel_values, tgt_sizes):
            # per image
            if len(pixel_b) != len(tgt_b):
                raise ValueError(
                    "Inconsistent N lengths, found: " f"{len(pixel_b)} vs {len(tgt_b)}"
                )
            for pixel_n, tgt_n in zip(pixel_b, tgt_b):
                pixel_values_flat += [pixel_n]
                tgt_sizes_flat += [tgt_n]

        pixel_values = pixel_values_flat
        if len(tgt_sizes_flat) == 0:
            tgt_sizes = None
        else:
            tgt_sizes = torch.stack(tgt_sizes_flat)
        if not isinstance(res["audio_features"], list):
            res["audio_features"] = [res["audio_features"]]
        return {
            "input_ids": res["input_ids"].flatten().tolist(),
            "pixel_values": pixel_values,
            "tgt_sizes": tgt_sizes,
            "data_hashes": base_output.mm_data_hashes,
            "modalities": request_obj.modalities or ["image"],
            "audio_start_id": audio_start_id,
            "audio_end_id": audio_end_id,
            "audio_features": res["audio_features"],
            "audio_bounds": res["audio_bounds"],
            "audio_feature_lens": res["audio_feature_lens"],
            "im_token_id": im_token_id,
            "im_start_id": tokenizer.im_start_id,
            "im_end_id": tokenizer.im_end_id,
            "slice_start_id": slice_start_id,
            "slice_end_id": slice_end_id,
        }
