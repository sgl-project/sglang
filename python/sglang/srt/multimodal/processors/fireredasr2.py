import re

from sglang.srt.configs.fireredasr2 import AUDIO_TOKEN, DEFAULT_ASR_PROMPT
from sglang.srt.managers.req_types import MultimodalProcessorOutput
from sglang.srt.models.fireredasr2 import FireRedASR2ForConditionalGeneration
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class FireRedASR2MultimodalProcessor(BaseMultimodalProcessor):
    models = [FireRedASR2ForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.audio_token_id = self._tokenizer.convert_tokens_to_ids(AUDIO_TOKEN)
        self.mm_tokens = MultimodalSpecialTokens(
            audio_token=AUDIO_TOKEN,
            audio_token_regex=re.compile(re.escape(AUDIO_TOKEN)),
            audio_token_id=self.audio_token_id,
        ).build(_processor)
        self.ATTR_NAME_TO_MODALITY = {
            **self.ATTR_NAME_TO_MODALITY,
            "speech_lengths": Modality.AUDIO,
            "fake_token_lengths": Modality.AUDIO,
        }

    async def process_mm_data_async(self, audio_data=None, input_text=None, **kwargs):
        if not audio_data:
            return None
        if len(audio_data) != 1:
            raise ValueError("FireRedASR2 supports one audio input per request")
        if isinstance(input_text, list):
            input_text = self._tokenizer.decode(input_text)
        prompt = input_text if input_text and input_text.strip() else DEFAULT_ASR_PROMPT
        base_output = await self.load_mm_data(
            prompt=prompt,
            audio_data=audio_data,
            audio_sample_rate=16000,
            multimodal_tokens=self.mm_tokens,
        )
        if base_output is None:
            return None
        mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
            base_output, self.mm_tokens
        )
        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            audio_token_id=self.audio_token_id,
        )
