"""MiMo-V2-ASR multimodal processor.

Audio preprocessing is delegated to :class:`MiMoAudioPipeline`; this
processor only handles the special-token contract and content interleaving.
"""

import asyncio
import re
from dataclasses import dataclass
from typing import List, Literal, Union

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.mimo_v2_asr import MiMoV2ASRForCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.mimo_audio import (
    AudioInput,
    MiMoAudioPipeline,
)
from sglang.utils import logger

TextInput = str | list[int]


@dataclass
class _Content:
    type: Literal["text", "audio"]
    content: TextInput | AudioInput


class MiMoV2ASRProcessor(BaseMultimodalProcessor):
    """ASR-only MiMo processor.

    Wires three special tokens into the input id stream around each audio
    span: ``<|sosp|> <|empty|>* <|eosp|>``. The actual mel/codec preparation
    is owned by :class:`MiMoAudioPipeline`, which is shared with the
    multimodal MiMo-V2 processor.
    """

    models = [MiMoV2ASRForCausalLM]

    AUDIO_PAD_TOKEN = "<|empty|>"
    AUDIO_START_TOKEN = "<|sosp|>"
    AUDIO_END_TOKEN = "<|eosp|>"

    AUDIO_REGEX = re.compile(r"<\|sosp\|>(?:<\|empty\|>)+<\|eosp\|>")

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.tokenizer = _processor

        self.audio_pipeline = MiMoAudioPipeline(
            audio_token_id=self._resolve_special_token_id(self.AUDIO_PAD_TOKEN),
            audio_start_token_id=self._resolve_special_token_id(self.AUDIO_START_TOKEN),
            audio_end_token_id=self._resolve_special_token_id(self.AUDIO_END_TOKEN),
            audio_sampling_rate=24000,
        )

        self.mm_tokens = MultimodalSpecialTokens(
            audio_token=f"{self.AUDIO_START_TOKEN}{self.AUDIO_PAD_TOKEN}{self.AUDIO_END_TOKEN}",
            audio_token_id=self.audio_token_id,
            audio_token_regex=self.AUDIO_REGEX,
        ).build(_processor)

    def __getattr__(self, name):
        # Delegate audio_pipeline fields so callers can use self.audio_token_id
        # etc. directly. Only triggers when normal attribute lookup fails;
        # __dict__.get avoids recursion before audio_pipeline is assigned.
        pipeline = self.__dict__.get("audio_pipeline")
        if pipeline is not None and hasattr(pipeline, name):
            return getattr(pipeline, name)
        raise AttributeError(name)

    def _resolve_special_token_id(self, name: str) -> int:
        tid = self.tokenizer.convert_tokens_to_ids(name)
        if tid is None or tid == self.tokenizer.unk_token_id:
            raise ValueError(
                f"tokenizer missing required special token {name!r}; "
                "checkpoint vocab does not match MiMo-V2-ASR"
            )
        return int(tid)

    def _process_contents(self, contents: List[_Content]):
        """Run pipeline + tokenizer over an interleaved content list.

        Returns ``(input_ids: Tensor[L], audio_inputs: list[Tensor],
        position_ids: Tensor[3,L], rope_deltas: Tensor[1,1])``.
        """
        input_ids: List[int] = []
        audio_inputs: List[torch.Tensor] = []

        for content in contents:
            if content.type == "text":
                if isinstance(content.content, str):
                    input_ids.extend(self.tokenizer.encode(content.content))
                else:
                    input_ids.extend(content.content)
            elif content.type == "audio":
                result = self.audio_pipeline.process_audio_input(content.content)
                audio_inputs.append(result["audio_input"])
                input_ids.extend(result["input_ids"])

        ids = torch.as_tensor(input_ids)
        position_ids = torch.arange(ids.shape[0]).expand(3, -1)
        rope_deltas = torch.zeros((1, 1), dtype=torch.int32)
        return ids, audio_inputs, position_ids, rope_deltas

    def process_mm_data(
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ) -> dict:
        if audios and not self.AUDIO_REGEX.search(input_text or ""):
            input_text = f"{self.mm_tokens.audio_token}{input_text or ''}"

        processed_audios: List[Union[tuple, torch.Tensor]] = []
        if audios:
            for audio in audios:
                if isinstance(audio, np.ndarray):
                    audio_tensor = torch.from_numpy(audio).float()
                elif isinstance(audio, torch.Tensor):
                    audio_tensor = audio.float()
                else:
                    processed_audios.append(audio)
                    continue
                if audio_tensor.ndim == 1:
                    processed_audios.append(
                        (audio_tensor.cpu().contiguous(), self.audio_sampling_rate)
                    )
                else:
                    processed_audios.append(audio_tensor.cpu().contiguous())

        contents: List[_Content] = []
        if input_text and processed_audios:
            multimodal_tokens_pattern = self.mm_tokens.get_combined_regex()
            text_parts = re.split(multimodal_tokens_pattern, input_text)
            audio_iter = iter(processed_audios)

            for text_part in text_parts:
                if multimodal_tokens_pattern.match(text_part):
                    modality = self.mm_tokens.get_modality_of_token(text_part)
                    if modality == Modality.AUDIO:
                        try:
                            audio = next(audio_iter)
                            contents.append(
                                _Content(type="audio", content=AudioInput(audio=audio))
                            )
                        except StopIteration:
                            pass
                else:
                    if text_part:
                        contents.append(_Content(type="text", content=text_part))
        else:
            contents.extend(
                _Content(type="audio", content=AudioInput(audio=audio))
                for audio in processed_audios
            )

        if not contents:
            ids = self.tokenizer(
                input_text or "",
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids
            return {"input_ids": ids}

        input_ids, audio_inputs, position_ids, rope_deltas = self._process_contents(
            contents
        )

        ret: dict = {
            "input_ids": input_ids,
            "mrope_positions": position_ids,
            "mrope_position_delta": rope_deltas,
        }
        if audio_inputs:
            ret["audio_features"] = audio_inputs
        return ret

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        audio_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        if audio_data is None:
            audio_data = getattr(request_obj, "audio_data", [])
        if not audio_data:
            return None
        if not self.AUDIO_REGEX.search(input_text):
            input_text = f"{self.mm_tokens.audio_token}{input_text}"

        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=[],
            video_data=[],
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
            audio_sample_rate=self.audio_sampling_rate,
        )
        multimodal_tokens_pattern = self.mm_tokens.get_combined_regex()

        raw_audio_data = audio_data or []
        loaded_audio_iter = iter(base_output.audios)
        raw_audio_iter = iter(raw_audio_data)

        text_parts = re.split(multimodal_tokens_pattern, base_output.input_text)
        contents: List[_Content] = []

        for text_part in text_parts:
            if multimodal_tokens_pattern.match(text_part):
                modality = self.mm_tokens.get_modality_of_token(text_part)
                assert modality is not None

                if modality == Modality.AUDIO:
                    loaded_audio = next(loaded_audio_iter)
                    raw_audio_item = next(raw_audio_iter)

                    if isinstance(loaded_audio, np.ndarray):
                        audio_source = loaded_audio
                    elif isinstance(raw_audio_item, dict):
                        audio_source = raw_audio_item.get("url", loaded_audio)
                    elif isinstance(raw_audio_item, (str, bytes, torch.Tensor)):
                        audio_source = raw_audio_item
                    else:
                        raise ValueError(
                            f"unsupported audio item: loaded={type(loaded_audio).__name__}, "
                            f"raw={type(raw_audio_item).__name__}"
                        )

                    contents.append(
                        _Content(
                            type="audio",
                            content=AudioInput(audio=audio_source),
                        )
                    )
            else:
                if text_part:
                    contents.append(_Content(type="text", content=text_part))

        loop = asyncio.get_running_loop()
        try:
            input_ids, audio_inputs, position_ids, rope_deltas = (
                await loop.run_in_executor(
                    self.io_executor,
                    lambda: self._process_contents(contents),
                )
            )
        except RuntimeError as e:
            logger.error(f"MiMo ASR processor failed in process_mm_data_async: {e}")
            raise ValueError(f"Multimodal data is corrupted or cannot be decoded: {e}")

        input_ids_flat = input_ids.flatten()
        if audio_inputs:
            mm_items = [
                MultimodalDataItem(
                    modality=Modality.AUDIO,
                    feature=audio_inputs,
                    offsets=self.get_mm_items_offset(
                        input_ids=input_ids_flat,
                        mm_token_id=self.audio_token_id,
                    ),
                )
            ]
        else:
            mm_items = []

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids_flat.tolist(),
            audio_token_id=self.audio_token_id,
            audio_start_id=self.audio_start_token_id,
            audio_end_id=self.audio_end_token_id,
            mrope_positions=position_ids,
            mrope_position_delta=rope_deltas,
        )
