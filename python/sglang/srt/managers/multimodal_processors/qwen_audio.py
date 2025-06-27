import re
from typing import List, Union

import torch

from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.qwen2_audio import Qwen2AudioForConditionalGeneration


class Qwen2AudioMultimodalProcessor(BaseMultimodalProcessor):
    models = [Qwen2AudioForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.AUDIO_TOKEN = "<|audio_bos|><|AUDIO|><|audio_eos|>"
        self.AUDIO_TOKEN_REGEX = re.compile(
            r"<\|audio_bos\|>(?:<\|AUDIO\|>)+<\|audio_eos\|>"
        )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        max_req_input_len,
        **kwargs,
    ):
        audio_data = request_obj.audio_data
        if not isinstance(audio_data, list):
            audio_data = [audio_data]

        base_output = self.load_mm_data(
            prompt=input_text,
            max_req_input_len=max_req_input_len,
            audio_data=audio_data,
            multimodal_tokens=MultimodalSpecialTokens(
                audio_token=self.AUDIO_TOKEN,
                audio_token_regex=self.AUDIO_TOKEN_REGEX,
            ),
        )
        if base_output is None:
            return None

        res = self.process_mm_data(
            input_text=base_output.input_text,
            audio=base_output.audios,
        )

        # Collect special token ids
        tokenizer = self._processor.tokenizer
        audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_bos|>")
        audio_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
        audio_end_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")

        items = []
        input_ids = res["input_ids"].flatten()

        if (
            "input_features" in res
            and res["input_features"] is not None
            and len(res["input_features"]) != 0
        ):
            if audio_start_id is not None and audio_end_id is not None:
                audio_offsets = self.get_mm_items_offset_by_pair(
                    input_ids=input_ids,
                    mm_start_id=audio_start_id,
                    mm_end_id=audio_end_id,
                )
            else:
                audio_offsets = None

            input_lengths = res["feature_attention_mask"].sum(dim=-1)
            input_lengths = (input_lengths - 1) // 2 + 1
            output_lengths = (input_lengths - 2) // 2 + 1

            item = MultimodalDataItem(
                audio_features=res["input_features"],
                audio_feature_lens=output_lengths,
                audio_offsets=audio_offsets,
                modality=Modality.AUDIO,
            )
            items += [item]

        return {
            "mm_items": items,
            "input_ids": input_ids.tolist(),
            "audio_start_id": audio_start_id,
            "audio_token_id": audio_token_id,
            "audio_end_id": audio_end_id,
        }
