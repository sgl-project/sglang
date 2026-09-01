import logging
import re

import numpy as np

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.qwen2_audio import Qwen2AudioForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

logger = logging.getLogger(__name__)


class Qwen2AudioMultimodalProcessor(BaseMultimodalProcessor):
    models = [Qwen2AudioForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.AUDIO_TOKEN = "<|audio_bos|><|AUDIO|><|audio_eos|>"
        self.AUDIO_TOKEN_REGEX = re.compile(
            r"<\|audio_bos\|>(?:<\|AUDIO\|>)+<\|audio_eos\|>"
        )
        # Collect special token ids
        tokenizer = self._processor.tokenizer
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_bos|>")
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
        self.audio_end_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")

        self.mm_tokens = MultimodalSpecialTokens(
            audio_token=self.AUDIO_TOKEN,
            audio_token_regex=self.AUDIO_TOKEN_REGEX,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

        self.ATTR_NAME_TO_MODALITY.update({"feature_attention_mask": Modality.AUDIO})

        # Qwen2-Audio's audio tower requires exactly 3000 mel frames (a fixed
        # 30s window); its HF feature extractor truncates to that by default.
        # BaseMultimodalProcessor otherwise passes ``truncation=False`` to audio
        # processors (needed by chunking encoders), which would feed >3000-frame
        # mel for clips longer than 30s and make the tower raise. Force
        # truncation so long clips are capped to the model's window.
        self.audio_config = {**self.audio_config, "truncation": True}

    # Qwen2-Audio's chat template matches a bare ``audio`` key; the strict instruction
    # keeps the model from emitting a "The content of this audio is:" preamble
    # that would otherwise inflate WER.
    _TRANSCRIPTION_CONVERSATION = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": ""},
                {
                    "type": "text",
                    "text": (
                        "Transcribe the audio. Output only the exact transcription, "
                        "with no preamble, prefix, commentary, or quotation marks."
                    ),
                },
            ],
        }
    ]

    def _build_transcription_prompt(self, input_text) -> str:
        """Fall back to a default ASR prompt for audio-only requests.

        The ``/v1/audio/transcriptions`` endpoint sends empty text (and hence
        empty ``input_ids``), which carries no audio placeholder. Render the
        Qwen2-Audio chat prompt with one audio span so the encoder features have
        a slot to fill; otherwise the caller-supplied text is used as-is.
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

    def _warn_if_audio_exceeds_window(self, audios) -> None:
        # Qwen2-Audio's encoder is a single fixed 30s window, so
        # warn user if audio is truncated.
        feature_extractor = self._processor.feature_extractor
        max_samples = int(
            feature_extractor.sampling_rate * feature_extractor.chunk_length
        )
        for audio in audios:
            if isinstance(audio, np.ndarray) and audio.shape[-1] > max_samples:
                logger.warning(
                    "Qwen2-Audio input is %.1fs but the encoder window is %ds; "
                    "only the first %ds will be transcribed (audio truncated).",
                    audio.shape[-1] / feature_extractor.sampling_rate,
                    feature_extractor.chunk_length,
                    feature_extractor.chunk_length,
                )

    def get_mm_data(self, prompt, embeddings, **kwargs):
        audio_feature_lens = kwargs.get("audio_feature_lens", None)

        # Convert audio_feature_lens to token counts for build_input_ids
        output_lengths = None
        input_lengths = None
        if audio_feature_lens is not None:
            if audio_feature_lens.dim() > 1:
                audio_feature_lens = audio_feature_lens.flatten()
            input_lengths = (audio_feature_lens - 1) // 2 + 1
            output_lengths = (input_lengths - 2) // 2 + 1

        input_ids, offsets, modality_list = self.build_input_ids(
            prompt,
            audio_seq_lens=output_lengths,
        )

        mm_items = []
        consumed_per_modality = {}

        for modality, offset in zip(modality_list, offsets):
            num_tokens = offset[1] - offset[0] + 1
            embedding_start = consumed_per_modality.get(modality, 0)
            embedding_slice = embeddings[modality][
                embedding_start : embedding_start + num_tokens
            ]
            consumed_per_modality[modality] = embedding_start + num_tokens
            mm_items.append(
                MultimodalDataItem(
                    modality=modality,
                    offsets=[offset],
                    precomputed_embeddings=embedding_slice,
                )
            )

        if mm_items:
            mm_items[0].audio_feature_lens = output_lengths

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids,
            audio_start_id=self.audio_start_id,
            audio_token_id=self.audio_token_id,
            audio_end_id=self.audio_end_id,
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

        self._warn_if_audio_exceeds_window(base_output.audios)

        mm_items, input_ids, ret = await self.process_and_combine_mm_data_async(
            base_output, self.mm_tokens
        )

        assert (
            "feature_attention_mask" in ret
        ), "feature_attention_mask not found in processor output"
        input_lengths = ret["feature_attention_mask"].sum(dim=-1)
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1

        mm_items[0].audio_feature_lens = output_lengths

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            audio_start_id=self.audio_start_id,
            audio_token_id=self.audio_token_id,
            audio_end_id=self.audio_end_id,
        )
