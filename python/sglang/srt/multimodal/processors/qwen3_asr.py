import re
from typing import Union

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput
from sglang.srt.models.qwen3_asr import Qwen3ASRForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

_DEFAULT_ASR_PROMPT = (
    "<|im_start|>user\n"
    "<|audio_start|><|audio_pad|><|audio_end|>"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)


class Qwen3ASRMultimodalProcessor(BaseMultimodalProcessor):
    models = [Qwen3ASRForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.AUDIO_TOKEN = "<|audio_start|><|audio_pad|><|audio_end|>"
        self.AUDIO_TOKEN_REGEX = re.compile(
            r"<\|audio_start\|>(?:<\|audio_pad\|>)+<\|audio_end\|>"
        )
        tokenizer = self._processor.tokenizer
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        self.audio_end_id = tokenizer.convert_tokens_to_ids("<|audio_end|>")

        self.mm_tokens = MultimodalSpecialTokens(
            audio_token=self.AUDIO_TOKEN,
            audio_token_regex=self.AUDIO_TOKEN_REGEX,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

        self.ATTR_NAME_TO_MODALITY.update({"feature_attention_mask": Modality.AUDIO})

    def _build_transcription_prompt(self, input_text: Union[str, list]) -> str:
        if isinstance(input_text, list):
            input_text = self._tokenizer.decode(input_text)
        if not input_text or not input_text.strip():
            return _DEFAULT_ASR_PROMPT
        return input_text

    def compute_mrope_positions(self, input_ids, mm_items):
        if isinstance(input_ids, list):
            seq_len = len(input_ids)
        else:
            seq_len = input_ids.shape[-1] if input_ids.dim() > 1 else input_ids.shape[0]
        positions = torch.arange(seq_len, dtype=torch.long)
        mrope_positions = positions.unsqueeze(0).expand(3, -1).clone()
        return mrope_positions, torch.tensor([0], dtype=torch.long)

    async def process_mm_data_async(
        self,
        audio_data=None,
        input_text=None,
        request_obj=None,
        **kwargs,
    ):
        if not audio_data:
            return None

        prompt = self._build_transcription_prompt(input_text)

        base_output = self.load_mm_data(
            prompt=prompt,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
        )
        if base_output is None:
            return None

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        mrope_positions, mrope_position_delta = self.compute_mrope_positions(
            input_ids, mm_items
        )

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            audio_start_id=self.audio_start_id,
            audio_token_id=self.audio_token_id,
            audio_end_id=self.audio_end_id,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
        )
