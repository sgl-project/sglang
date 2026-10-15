import logging
import re

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput
from sglang.srt.models.midashenglm import MiDashengLMModel
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

logger = logging.getLogger(__name__)


class MiDashengLMMultimodalProcessor(BaseMultimodalProcessor):
    """Multimodal processor for MiDashengLM audio-language model."""

    models = [MiDashengLMModel]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.AUDIO_TOKEN = "<|audio_bos|><|AUDIO|><|audio_eos|>"
        self.AUDIO_TOKEN_REGEX = re.compile(
            r"<\|audio_bos\|>(?:<\|AUDIO\|>)+<\|audio_eos\|>"
        )

        tokenizer = self._processor.tokenizer
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_bos|>")
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
        self.audio_end_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")

        self.mm_tokens = MultimodalSpecialTokens(
            audio_token=self.AUDIO_TOKEN,
            audio_token_regex=self.AUDIO_TOKEN_REGEX,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

        self.ATTR_NAME_TO_MODALITY.update(
            {
                "input_values": Modality.AUDIO,
                "audio_length": Modality.AUDIO,
            }
        )

        if "input_values" not in self.FEATURE_NAMES:
            self.FEATURE_NAMES.append("input_values")

    def process_mm_data(
        self,
        input_text,
        images=None,
        videos=None,
        audios=None,
        processor=None,
        **kwargs,
    ):
        """Override to use correct audio parameter name for MiDashengLM processor."""
        if images:
            kwargs["images"] = images
        if videos:
            kwargs["videos"] = videos
        if audios:
            kwargs["audio"] = audios
            kwargs.setdefault("audio_kwargs", {})
            kwargs["audio_kwargs"].setdefault("truncation", False)
            if self.audio_config:
                kwargs["audio_kwargs"].update(self.audio_config)

        # Take the worker pool's per-thread clone when it hands one over; falling
        # back to self._processor would put every worker on one shared object.
        processor, _ = self._resolve_processor(processor)
        result = processor.__call__(
            text=[input_text],
            padding=True,
            return_tensors="pt",
            **kwargs,
        )

        if not self.use_cuda_ipc:
            for feature_name in ["input_values"]:
                if feature_name in result:
                    result[feature_name] = result[feature_name].cpu()

        return result

    def _build_transcription_prompt(self, input_text) -> str:
        """Normalize ``input_text`` for the ``/v1/audio/transcriptions`` endpoint.

        That endpoint sends empty text, which the tokenizer manager turns into
        empty ``input_ids`` (a list) rather than an empty string when a
        multimodal processor is attached -- ``AUDIO_TOKEN_REGEX.search``
        requires a str, so this must be normalized before use. Same pattern as
        ``qwen_audio.py``'s ``_build_transcription_prompt``.
        """
        if isinstance(input_text, list):
            input_text = (
                self._processor.tokenizer.decode(input_text) if input_text else ""
            )
        return input_text

    async def process_mm_data_async(
        self,
        audio_data,
        input_text,
        **kwargs,
    ):
        """Process audio data for MiDashengLM model."""
        input_text = self._build_transcription_prompt(input_text)

        if audio_data and not self.AUDIO_TOKEN_REGEX.search(input_text):
            input_text = f"{self.AUDIO_TOKEN}{input_text}"

        base_output = await self.load_mm_data(
            prompt=input_text,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
        )
        if base_output is None:
            return None

        mm_items, input_ids, ret = await self.process_and_combine_mm_data_async(
            base_output, self.mm_tokens
        )

        if "audio_length" in ret and len(mm_items) > 0:
            audio_length = ret["audio_length"]
            if isinstance(audio_length, torch.Tensor):
                audio_length = (
                    audio_length.item()
                    if audio_length.numel() == 1
                    else audio_length[0].item()
                )
            mm_items[0].audio_length = audio_length
        elif "input_values" in ret and len(mm_items) > 0:
            input_values = ret["input_values"]
            audio_length = (
                input_values.shape[-1]
                if input_values.ndim >= 2
                else input_values.shape[0]
            )
            mm_items[0].audio_length = audio_length

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            audio_start_id=self.audio_start_id,
            audio_token_id=self.audio_token_id,
            audio_end_id=self.audio_end_id,
        )
