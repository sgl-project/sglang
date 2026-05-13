# SPDX-License-Identifier: Apache-2.0

from typing import Any

from sglang.omni.model_adapters.sensenova_u1.context import (
    U1_IMAGE_PLACEHOLDER,
    U1_IMG_CONTEXT_TOKEN,
    U1_IMG_END_TOKEN,
    U1_IMG_START_TOKEN,
    build_u1_vlm_input_ids_and_offsets,
    load_u1_native_image,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.sensenova_u1 import NEOChatModel
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor


class SenseNovaU1MultimodalProcessor(BaseMultimodalProcessor):
    models = [NEOChatModel]
    gpu_image_decode = False

    async def process_mm_data_async(
        self,
        image_data=None,
        audio_data=None,
        input_text: str | list[int] = "",
        request_obj=None,
        **kwargs: Any,
    ) -> MultimodalProcessorOutput:
        del request_obj, kwargs
        if audio_data:
            raise ValueError("SenseNova U1 processor does not support audio input")
        if image_data:
            return self._process_image_input(image_data, input_text)
        if isinstance(input_text, list):
            input_ids = input_text
        else:
            input_ids = (
                self._tokenizer(
                    input_text,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
                .input_ids.flatten()
                .tolist()
            )
        return MultimodalProcessorOutput(mm_items=[], input_ids=input_ids)

    def _process_image_input(
        self,
        image_data: Any,
        input_text: str | list[int],
    ) -> MultimodalProcessorOutput:
        images = image_data if isinstance(image_data, list) else [image_data]
        images = [image for image in images if image is not None]
        if len(images) != 1:
            raise ValueError(
                "SenseNova U1 standard VLM path currently supports exactly one image"
            )

        image = self.__class__._load_single_item(
            images[0],
            Modality.IMAGE,
            discard_alpha_channel=True,
        )
        pixel_values, grid_hw = load_u1_native_image(image)
        input_ids, image_offsets, _ = build_u1_vlm_input_ids_and_offsets(
            tokenizer=self._tokenizer,
            grid_hw=grid_hw,
            question=self._extract_question(input_text),
        )
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=pixel_values,
            model_specific_data={"image_grid_hws": grid_hw},
            offsets=image_offsets,
        )
        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=[item],
            im_start_id=self._token_id(U1_IMG_START_TOKEN),
            im_end_id=self._token_id(U1_IMG_END_TOKEN),
            im_token_id=self._token_id(U1_IMG_CONTEXT_TOKEN),
        )

    def _extract_question(self, input_text: str | list[int]) -> str:
        if isinstance(input_text, list):
            input_text = self._tokenizer.decode(input_text)
        text = str(input_text)
        if "<|im_start|>user" in text:
            text = text.split("<|im_start|>user")[-1]
        if "<|im_end|>" in text:
            text = text.split("<|im_end|>", 1)[0]
        if "<|im_start|>assistant" in text:
            text = text.split("<|im_start|>assistant", 1)[0]
        for token in (
            U1_IMAGE_PLACEHOLDER,
            U1_IMG_START_TOKEN,
            U1_IMG_END_TOKEN,
            U1_IMG_CONTEXT_TOKEN,
        ):
            text = text.replace(token, "")
        question = text.strip()
        if not question:
            raise ValueError("SenseNova U1 VLM image input requires a text question")
        return question

    def _token_id(self, token: str) -> int:
        token_id = self._tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            raise ValueError(f"SenseNova U1 tokenizer has no token {token!r}")
        return int(token_id)
