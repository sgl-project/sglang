import re

from sglang.srt.models.glmasr import GlmasrForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class GlmasrProcessor(BaseMultimodalProcessor):
    models = [GlmasrForConditionalGeneration]

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
        print(mm_items)
        audio_lengths = ret.pop("attention_mask").sum(-1)

        def _get_audio_token_length(audio_length: int, merge_factor: int = 4) -> int:
            for padding, kernel_size, stride in [(1, 3, 1), (1, 3, 2)]:
                audio_length = (
                    audio_length + 2 * padding - (kernel_size - 1) - 1
                ) // stride + 1
            num_tokens = (audio_length - merge_factor) // merge_factor + 1
            return min(num_tokens, 1500 // merge_factor)

        input_lengths = _get_audio_token_length(audio_lengths)
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1

        mm_items[0].audio_feature_lens = output_lengths

        return {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "audio_start_id": self.audio_start_id,
            "audio_token_id": self.audio_token_id,
            "audio_end_id": self.audio_end_id,
        }
