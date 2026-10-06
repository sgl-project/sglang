import re

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
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

    # GLM-ASR's chat template keys on ``{"type": "audio"}`` (or any mapping
    # containing an ``audio`` key) to emit its audio placeholder span.
    _TRANSCRIPTION_CONVERSATION = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": ""},
                {"type": "text", "text": "Transcribe the audio."},
            ],
        }
    ]

    def _build_transcription_prompt(self, input_text) -> str:
        """Fall back to a default ASR prompt for audio-only requests.

        The ``/v1/audio/transcriptions`` endpoint sends empty text (and hence
        empty ``input_ids``), which carries no audio placeholder. Render the
        GLM-ASR chat prompt with one audio span so the encoder features have a
        slot to fill; otherwise the caller-supplied text is used as-is.
        """
        if isinstance(input_text, list):
            input_text = (
                self._processor.tokenizer.decode(input_text) if input_text else ""
            )
        if input_text and input_text.strip():
            return input_text
        return self._processor.apply_chat_template(
            self._TRANSCRIPTION_CONVERSATION,
            add_generation_prompt=True,
            tokenize=False,
        )

    async def process_mm_data_async(
        self,
        audio_data,
        input_text,
        **kwargs,
    ):
        prompt = self._build_transcription_prompt(input_text)
        base_output = await self.load_mm_data(
            prompt=prompt,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
        )
        if base_output is None:
            return None
        mm_items, input_ids, ret = await self.process_and_combine_mm_data_async(
            base_output, self.mm_tokens
        )
        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            audio_start_id=self.audio_start_id,
            audio_token_id=self.audio_token_id,
            audio_end_id=self.audio_end_id,
        )
