import asyncio
import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.dots3 import Dots3NoteOmniForCausalLM
from sglang.srt.models.dots_omni_towers import (
    DotsNoteOmniImagePreprocessor,
    OmniAudioConfig,
    get_audio_token_string,
    load_omni_component_config,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.dots_note_omni_video import (
    preprocess_dots_video,
)

logger = logging.getLogger(__name__)


class DotsNoteOmniProcessor(BaseMultimodalProcessor):
    """Native image/audio processor for dots.note.omni."""

    models = [Dots3NoteOmniForCausalLM]
    gpu_image_decode = False

    def __init__(self, hf_config, server_args, processor, transport_mode, **kwargs):
        self.image_start_token = getattr(hf_config, "im_start_token", "<|img|>")
        self.image_token = getattr(hf_config, "im_token", "<|imgpad|>")
        self.image_end_token = getattr(hf_config, "im_end_token", "<|endofimg|>")
        self.audio_start_token = getattr(
            hf_config, "audio_start_token", "<|audio_comp_start|>"
        )
        self.audio_token = getattr(hf_config, "audio_token", "<|audio_comp_pad|>")
        self.audio_end_token = getattr(
            hf_config, "audio_end_token", "<|audio_comp_end|>"
        )
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.image_token,
            image_token_id=self._token_id(processor, self.image_token),
            image_token_regex=re.compile(
                re.escape(
                    self.image_start_token + self.image_token + self.image_end_token
                )
            ),
            audio_token=self.audio_token,
            audio_token_id=self._token_id(processor, self.audio_token),
            audio_token_regex=re.compile(
                re.escape(
                    self.audio_start_token + self.audio_token + self.audio_end_token
                )
            ),
        ).build(processor)
        self.mm_token_ids = {
            "im_start_id": self._token_id(processor, self.image_start_token),
            "im_token_id": self._token_id(processor, self.image_token),
            "im_end_id": self._token_id(processor, self.image_end_token),
            "audio_start_id": self._token_id(processor, self.audio_start_token),
            "audio_token_id": self._token_id(processor, self.audio_token),
            "audio_end_id": self._token_id(processor, self.audio_end_token),
        }

        model_dir = Path(hf_config._name_or_path)
        self.image_preprocessor = DotsNoteOmniImagePreprocessor(str(model_dir))
        self.audio_processor_config = OmniAudioConfig(
            **load_omni_component_config(model_dir, "audio")
        )
        super().__init__(hf_config, server_args, processor, transport_mode, **kwargs)

    @staticmethod
    def _token_id(processor, token: str) -> int:
        tokenizer = getattr(processor, "tokenizer", processor)
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Dots omni special token {token!r} must encode to one id, got "
                f"{token_ids}"
            )
        return token_ids[0]

    @staticmethod
    def _normalize_audio(audio) -> torch.Tensor:
        if isinstance(audio, torch.Tensor):
            waveform = audio
        elif isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio)
        else:
            waveform = torch.as_tensor(audio)
        waveform = waveform.float().squeeze()
        if waveform.ndim != 1:
            raise ValueError(
                f"Dots omni audio must be mono, got shape={tuple(waveform.shape)}"
            )
        return waveform.contiguous()

    def _render_video_content(
        self, input_text: str, question: str, content: list[dict]
    ) -> tuple[str, list[str], list[str]]:
        """Convert the adapter's OpenAI content back to native model markers."""
        rendered = []
        images = []
        audios = []
        for item in content:
            item_type = item.get("type")
            if item_type == "text":
                rendered.append(item.get("text", ""))
            elif item_type == "image_url":
                images.append(item["image_url"]["url"])
                rendered.append(
                    self.image_start_token + self.image_token + self.image_end_token
                )
            elif item_type == "audio_url":
                audios.append(item["audio_url"]["url"])
                rendered.append(
                    self.audio_start_token + self.audio_token + self.audio_end_token
                )
            else:
                raise ValueError(f"Unsupported preprocessed video item: {item_type}")

        expanded = "".join(rendered)
        if question and question in input_text:
            input_text = input_text.replace(question, expanded, 1)
        else:
            # The normal dots template starts the user turn with <|user|>.  Keep
            # system text ahead of video media if a custom template is used.
            user_marker = "<|user|>"
            pos = input_text.rfind(user_marker)
            insert_at = pos + len(user_marker) if pos >= 0 else 0
            input_text = input_text[:insert_at] + expanded + input_text[insert_at:]
        return input_text, images, audios

    async def process_mm_data_async(
        self,
        input_text: Union[List[int], str],
        request_obj: GenerateReqInput,
        max_req_input_len: int,
        *args,
        image_data: Optional[list] = None,
        audio_data: Optional[list] = None,
        video_data=None,
        **kwargs,
    ):
        video_data = getattr(request_obj, "video_data", None) or video_data
        if not image_data and not audio_data and not video_data:
            return None
        if video_data:
            if image_data or audio_data:
                raise ValueError(
                    "Dots note omni does not yet support mixing a native video "
                    "with separate image/audio inputs"
                )
            if len(video_data) != 1:
                raise ValueError(
                    "Dots note omni currently supports one video per request"
                )
        if not isinstance(input_text, str):
            raise ValueError(
                "Dots note omni requires a text prompt for multimodal requests"
            )

        if video_data:
            question = getattr(request_obj, "video_question", None) or ""
            sampling_params = getattr(request_obj, "sampling_params", None) or {}
            if isinstance(sampling_params, dict):
                max_new_tokens = sampling_params.get("max_new_tokens") or 0
            else:
                max_new_tokens = getattr(sampling_params, "max_new_tokens", 0) or 0
            loop = asyncio.get_running_loop()
            preprocess_started = time.perf_counter()
            content = await loop.run_in_executor(
                self.io_executor,
                lambda: preprocess_dots_video(
                    video_data[0],
                    question,
                    tokenizer=self._tokenizer,
                    seq=getattr(request_obj, "seq", 131072),
                    output_reserve=getattr(request_obj, "output_reserve", None),
                    audio_cap=getattr(request_obj, "audio_cap", 1.0),
                    audio_sr=getattr(request_obj, "audio_sr", 16000),
                    k_mode=getattr(request_obj, "k_mode", "eval_ek"),
                    max_new_tokens=max_new_tokens,
                ),
            )
            preprocess_elapsed = time.perf_counter() - preprocess_started
            logger.info(
                "[dots_video_preprocess] rid=%s elapsed=%.3fs frames=%d "
                "audio_segments=%d content_items=%d",
                getattr(request_obj, "rid", None),
                preprocess_elapsed,
                sum(item.get("type") == "image_url" for item in content),
                sum(item.get("type") == "audio_url" for item in content),
                len(content),
            )
            input_text, image_data, audio_data = self._render_video_content(
                input_text, question, content
            )

        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            audio_data=audio_data,
            video_data=None,
            multimodal_tokens=self.mm_tokens,
            audio_sample_rate=self.audio_processor_config.sampling_rate,
        )

        pattern = self.mm_tokens.get_combined_regex()
        parts = re.split(pattern, base_output.input_text)
        modality_order = [
            modality
            for part in parts
            if (modality := self.mm_tokens.get_modality_of_token(part)) is not None
        ]
        if modality_order.count(Modality.IMAGE) != len(base_output.images):
            raise ValueError("Image placeholder count does not match image_data")
        if modality_order.count(Modality.AUDIO) != len(base_output.audios):
            raise ValueError("Audio placeholder count does not match audio_data")

        image_features, image_grids, image_token_strings = (
            self.image_preprocessor.process_images(base_output.images)
            if base_output.images
            else ([], [], [])
        )
        audio_features = [self._normalize_audio(audio) for audio in base_output.audios]
        audio_token_strings = [
            get_audio_token_string(waveform.numel(), self.audio_processor_config)
            for waveform in audio_features
        ]
        feature_iters = {
            Modality.IMAGE: iter(zip(image_features, image_grids, image_token_strings)),
            Modality.AUDIO: iter(zip(audio_features, audio_token_strings)),
        }

        input_ids = []
        mm_items = []
        add_special_tokens = True
        for part in parts:
            modality = self.mm_tokens.get_modality_of_token(part)
            if modality is None:
                input_ids.extend(
                    self._tokenizer.encode(part, add_special_tokens=add_special_tokens)
                )
                add_special_tokens = False
                continue

            if modality == Modality.IMAGE:
                feature, grid_thw, expanded_token_string = next(feature_iters[modality])
                pad_token_id = self.mm_token_ids["im_token_id"]
                model_specific_data = {"image_grid_thw": grid_thw.reshape(-1, 3)}
            else:
                feature, expanded_token_string = next(feature_iters[modality])
                pad_token_id = self.mm_token_ids["audio_token_id"]
                model_specific_data = {}

            item_token_ids = self._tokenizer.encode(
                expanded_token_string, add_special_tokens=False
            )
            item_start = len(input_ids)
            input_ids.extend(item_token_ids)
            local_offsets = self.get_mm_items_offset(
                torch.tensor(item_token_ids), pad_token_id
            )
            offsets = [
                (item_start + start, item_start + end) for start, end in local_offsets
            ]
            item = MultimodalDataItem(
                modality=modality,
                feature=feature,
                offsets=offsets,
                model_specific_data=model_specific_data,
            )
            item.set_pad_value()
            mm_items.append(item)

        if len(input_ids) > max_req_input_len:
            raise ValueError(
                "Dots note omni expanded prompt is too long: "
                f"{len(input_ids)} > {max_req_input_len}"
            )
        padded_input_ids = MultimodalProcessorOutput.build_padded_input_ids(
            input_ids, mm_items
        )
        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids,
            padded_input_ids=padded_input_ids,
            **self.mm_token_ids,
        )
