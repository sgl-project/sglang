from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput
from sglang.srt.models.ultravox import UltravoxModel
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class UltravoxMultimodalProcessor(BaseMultimodalProcessor):
    models = [UltravoxModel]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.AUDIO_TOKEN = "<|audio|>"
        self.audio_token_id = self._processor.tokenizer.convert_tokens_to_ids(
            self.AUDIO_TOKEN
        )
        # UltravoxProcessor substitutes EOT for the placeholder because HF locates
        # audio spans by an explicit start index. SGLang locates them by the
        # placeholder id, so keep <|audio|> in the emitted input_ids instead.
        self._processor.audio_token_replacement = self.AUDIO_TOKEN

        self.mm_tokens = MultimodalSpecialTokens(
            audio_token=self.AUDIO_TOKEN,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

        # UltravoxProcessor names the mel `audio_values` and its true per-chunk
        # frame counts `audio_lens`; neither is in the shared attribute table.
        self.ATTR_NAME_TO_MODALITY.update(
            {"audio_values": Modality.AUDIO, "audio_lens": Modality.AUDIO}
        )
        self.FEATURE_NAMES.append("audio_values")

    # Ultravox reuses the stock Llama-3 chat template, which has no audio span of
    # its own; the placeholder is written into the user turn instead.
    _TRANSCRIPTION_PROMPT = (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "<|audio|>\nTranscribe the audio. Output only the exact transcription, "
        "with no preamble, prefix, commentary, or quotation marks."
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    def _build_transcription_prompt(self, input_text) -> str:
        """Fall back to a default ASR prompt for audio-only requests.

        The ``/v1/audio/transcriptions`` endpoint sends empty text, which carries
        no audio placeholder; otherwise the caller-supplied text is used as-is.
        """
        if isinstance(input_text, list):
            input_text = (
                self._processor.tokenizer.decode(input_text) if input_text else ""
            )
        if input_text and input_text.strip():
            return input_text
        return self._TRANSCRIPTION_PROMPT

    def process_mm_data(
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ):
        """UltravoxProcessor splits the text on the placeholder itself, so it takes a
        single string and rejects a list, and it sets its own audio `padding`, so the
        batched padded call the base class makes raises on both counts.
        """
        processor, _ = self._resolve_processor(kwargs.pop("processor", None))
        return processor(
            text=input_text,
            audios=audios,
            sampling_rate=processor.audio_processor.sampling_rate,
            return_tensors="pt",
        )

    async def process_mm_data_async(self, audio_data, input_text, **kwargs):
        prompt = self._build_transcription_prompt(input_text)
        base_output = await self.load_mm_data(
            prompt=prompt,
            audio_data=audio_data,
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
