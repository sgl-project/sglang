import re

from sglang.srt.models.glmasr import GlmAsrForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class GlmAsrProcessor(BaseMultimodalProcessor):
    models = [GlmAsrForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.AUDIO_TOKEN = "<|begin_of_audio|><|pad|><|end_of_audio|>"
        self.AUDIO_TOKEN_REGEX = re.compile(
            r"<\|begin_of_audio\|><\|pad\|><\|end_of_audio\|>"
        )
        # Collect special token ids
        tokenizer = self._processor.tokenizer
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<|begin_of_audio|>")
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
        self.audio_end_id = tokenizer.convert_tokens_to_ids("<|end_of_audio|>")

        self.mm_tokens = MultimodalSpecialTokens(
            audio_token=self.AUDIO_TOKEN,
            audio_token_regex=self.AUDIO_TOKEN_REGEX,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

    async def process_mm_data_async(
        self,
        audio_data,
        input_text,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
        )
        if base_output is None:
            return None
        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )
        return {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "audio_start_id": self.audio_start_id,
            "audio_token_id": self.audio_token_id,
            "audio_end_id": self.audio_end_id,
        }
