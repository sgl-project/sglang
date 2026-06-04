import re
from typing import Union

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput
from sglang.srt.models.moss_transcribe_diarize import (
    MossTranscribeDiarizeForConditionalGeneration,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"


class MossTranscribeDiarizeMultimodalProcessor(BaseMultimodalProcessor):
    models = [MossTranscribeDiarizeForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.AUDIO_TOKEN = AUDIO_PLACEHOLDER
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
        self.ATTR_NAME_TO_MODALITY.update(
            {
                "audio_feature_lengths": Modality.AUDIO,
                "audio_chunk_mapping": Modality.AUDIO,
            }
        )

    def _build_prompt(self, input_text: Union[str, list, None]) -> str:
        if isinstance(input_text, list):
            input_text = self._tokenizer.decode(input_text)
        input_text = input_text or ""
        if "<|audio_pad|>" in input_text:
            return input_text

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": ""},
                    {"type": "text", "text": input_text},
                ],
            }
        ]
        return self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def process_mm_data(
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ):
        if images or videos:
            raise ValueError("MOSS-Transcribe-Diarize only supports audio inputs.")
        if not audios:
            return self._tokenizer(
                input_text,
                return_tensors="pt",
                add_special_tokens=True,
            )

        if self.audio_config:
            kwargs.setdefault("audio_kwargs", {}).update(self.audio_config)
        result = self._processor(
            text=input_text,
            audio=audios,
            return_tensors="pt",
            **kwargs,
        )
        if not self.server_args.keep_mm_feature_on_device:
            for feature_name in self.FEATURE_NAMES:
                if feature_name in result and isinstance(
                    result[feature_name], torch.Tensor
                ):
                    result[feature_name] = result[feature_name].to("cpu")
        return result

    async def process_mm_data_async(
        self,
        audio_data=None,
        input_text=None,
        request_obj=None,
        **kwargs,
    ):
        if not audio_data:
            return None

        prompt = self._build_prompt(input_text)
        sampling_rate = int(self._processor.feature_extractor.sampling_rate)
        base_output = await self.load_mm_data(
            prompt=prompt,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
            audio_sample_rate=sampling_rate,
        )
        if base_output is None:
            return None

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output,
            self.mm_tokens,
        )

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            audio_start_id=self.audio_start_id,
            audio_token_id=self.audio_token_id,
            audio_end_id=self.audio_end_id,
        )
