# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Prompt expansion for GigaChat3.5-Omni.

Decoding and preprocessing come from the checkpoint's own processor
(`processing_gigachat35_omni.py`, loaded with `trust_remote_code`), so serving
feeds the encoders what training did. This module lays out the placeholders:

- image: `[image_token]` x 256 per crop;
- video: the frames' `[video_image_token]` runs, sized for the kept tokens; with
  `use_audio_in_video`, every frame is `HH:MM:SS`, its frame tokens and its
  share of the soundtrack's `[video_audio_token]`s;
- audio: `[audio_token]` x 6.25 per second with an `HH:MM:SS` stamp before
  every 60th second, then the clip duration as `HH:MM:SS`.
"""

import asyncio
import functools
from collections import deque
from typing import Any, List, Optional, Tuple

import torch

from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalProcessorOutput
from sglang.srt.models.gigachat35_omni import GigaChat35OmniForConditionalGeneration
from sglang.srt.multimodal.gigachat35_omni import GigaChat35OmniVideoEmbedding
from sglang.srt.multimodal.media_io import (
    CLIENT_MEDIA_EXCEPTIONS,
    _normalize_video_input,
    load_image,
)
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor


class GigaChat35OmniMultimodalProcessor(BaseMultimodalProcessor):
    """Expands GigaChat3.5-Omni media placeholders into per-token runs."""

    models = [GigaChat35OmniForConditionalGeneration]

    # One audio token covers 160 ms; training stamps the audio every 60 s.
    AUDIO_TOKEN_MS = 160
    AUDIO_STAMP_PERIOD_S = 60
    # One 20 ms mel frame at 16 kHz; shorter soundtracks count as silent.
    MIN_SOUNDTRACK_SAMPLES = 320

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.image_token_id = hf_config.image_token_id
        self.video_token_id = hf_config.video_token_id
        self.audio_token_id = hf_config.audio_token_id
        self.video_audio_token_id = hf_config.video_audio_token_id
        vision = hf_config.vision_config
        self.frame_tokens = int(
            (vision.image_size // vision.patch_size) ** 2
            * hf_config.downsample_ratio**2
        )
        self.frame_side = int(self.frame_tokens**0.5)
        self.video_tokens_per_frame = hf_config.video_tokens_per_frame

    @staticmethod
    def seconds_to_hms(seconds: float) -> str:
        """Formats a time as training does.

        Args:
            seconds: Time in seconds, truncated to whole seconds.

        Returns:
            `HH:MM:SS`.
        """
        seconds = int(seconds)
        return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"

    def encode_hms(self, seconds: float) -> List[int]:
        """Tokenizes a timestamp on its own, as training does.

        Args:
            seconds: Time in seconds.

        Returns:
            Token ids of `HH:MM:SS`.
        """
        return self._tokenizer.encode(
            self.seconds_to_hms(seconds), add_special_tokens=False
        )

    @staticmethod
    def append_run(input_ids: List[int], token_id: int, count: int) -> Tuple[int, int]:
        """Appends `count` copies of a placeholder to the prompt.

        Args:
            input_ids: Expanded prompt so far, extended in place.
            token_id: The placeholder.
            count: Number of copies.

        Returns:
            The inclusive span of the copies.
        """
        input_ids.extend([token_id] * count)
        return len(input_ids) - count, len(input_ids) - 1

    def load_image(self, data: Any) -> torch.Tensor:
        """Decodes one image and splits it into crops.

        Args:
            data: Image as a URL, path, base64 string, bytes or `ImageData`.

        Returns:
            Crops `[n_crops, 3, 448, 448]` bfloat16.
        """
        image, _ = load_image(data, gpu_image_decode=False)
        return self._processor.preprocess_image(image)

    def load_video(self, data: Any, use_audio: bool) -> dict:
        """Samples frames and, if asked, the soundtrack of one video.

        Args:
            data: Video as a URL, path, base64 string, bytes or `VideoData`.
            use_audio: Whether to decode the soundtrack.

        Returns:
            Dict with `pixel_values`, `timestamps` and `mel` (`[T_mel, 64]` or
            None when there is no soundtrack).
        """
        video = self._processor.load_video(
            _normalize_video_input(getattr(data, "url", data)), use_audio=use_audio
        )
        audio = video.pop("audio")
        has_soundtrack = audio is not None and len(audio) >= self.MIN_SOUNDTRACK_SAMPLES
        video["mel"] = self._processor.extract_mel(audio) if has_soundtrack else None
        return video

    def load_audio(self, data: Any) -> tuple[torch.Tensor, float]:
        """Decodes one audio file into mel frames.

        Args:
            data: Audio as a URL, path, base64 string or bytes.

        Returns:
            Mel frames `[T_mel, 64]` float32 and the duration in seconds.
        """
        waveform, duration = self._processor.load_audio(_normalize_video_input(data))
        return self._processor.extract_mel(waveform), duration

    def expand_video(
        self,
        video: dict,
        video_index: int,
        input_ids: List[int],
        items: List[MultimodalDataItem],
    ) -> None:
        """Appends the placeholders of one video and creates its items.

        Args:
            video: Output of `load_video`.
            video_index: Position of the video among the request's videos.
            input_ids: Expanded prompt so far, extended in place.
            items: Items so far, extended in place.
        """
        pixel_values, mel = video["pixel_values"], video["mel"]
        num_frames = pixel_values.shape[0]
        counts = GigaChat35OmniVideoEmbedding.placeholder_counts(
            num_frames, self.frame_tokens, self.video_tokens_per_frame
        )
        frame_offsets = []

        if mel is None:
            frame_offsets.append(
                self.append_run(input_ids, self.video_token_id, sum(counts))
            )
        else:
            num_audio = self._processor.num_audio_tokens(mel.shape[0])
            audio_per_frame, remainder = divmod(num_audio, num_frames)
            audio_offsets = []

            for i, (timestamp, count) in enumerate(zip(video["timestamps"], counts)):
                input_ids.extend(self.encode_hms(timestamp))
                frame_offsets.append(
                    self.append_run(input_ids, self.video_token_id, count)
                )
                num_audio_frame = audio_per_frame + (i < remainder)

                if num_audio_frame:
                    audio_offsets.append(
                        self.append_run(
                            input_ids, self.video_audio_token_id, num_audio_frame
                        )
                    )

        items.append(
            MultimodalDataItem(
                modality=Modality.VIDEO,
                feature=pixel_values,
                offsets=frame_offsets,
                model_specific_data={
                    "thw_grids": [(num_frames, self.frame_side, self.frame_side)]
                },
            )
        )

        if mel is not None:
            items.append(
                MultimodalDataItem(
                    modality=Modality.AUDIO,
                    feature=mel,
                    offsets=audio_offsets,
                    model_specific_data={
                        GigaChat35OmniVideoEmbedding.SOUNDTRACK_OF_KEY: video_index
                    },
                )
            )

    def expand_audio(
        self,
        mel: torch.Tensor,
        duration: float,
        input_ids: List[int],
        items: List[MultimodalDataItem],
    ) -> None:
        """Appends the placeholders, minute stamps and duration of one audio file.

        Args:
            mel: Mel frames of the clip.
            duration: Clip duration in seconds.
            input_ids: Expanded prompt so far, extended in place.
            items: Items so far, extended in place.
        """
        num_tokens = self._processor.num_audio_tokens(mel.shape[0])
        seconds = range(
            0, num_tokens * self.AUDIO_TOKEN_MS // 1000, self.AUDIO_STAMP_PERIOD_S
        )
        starts = [second * 1000 // self.AUDIO_TOKEN_MS for second in seconds]
        offsets = []

        for second, start, end in zip(seconds, starts, starts[1:] + [num_tokens]):
            input_ids.extend(self.encode_hms(second))
            offsets.append(self.append_run(input_ids, self.audio_token_id, end - start))

        # A clip under one second has no stamp.
        if not offsets:
            offsets.append(self.append_run(input_ids, self.audio_token_id, num_tokens))

        input_ids.extend(self.encode_hms(duration))
        items.append(
            MultimodalDataItem(modality=Modality.AUDIO, feature=mel, offsets=offsets)
        )

    async def process_mm_data_async(
        self,
        image_data: Optional[List[Any]],
        audio_data: Optional[List[Any]],
        input_text: str | List[int],
        request_obj: Any,
        **kwargs: Any,
    ) -> Optional[MultimodalProcessorOutput]:
        """Loads the request's media and expands its placeholders.

        Args:
            image_data: Images in prompt order.
            audio_data: Audio files in prompt order.
            input_text: Rendered prompt or its token ids.
            request_obj: The request; `video_data` and `use_audio_in_video` are read.
            **kwargs: Unused.

        Returns:
            Expanded token ids and one item per image, video, soundtrack and audio file.

        Raises:
            ValueError: A media item cannot be loaded, or the prompt has a
                different number of placeholders than media.
        """
        video_data = request_obj.video_data

        if not image_data and not audio_data and not video_data:
            return None

        images, videos, audios = image_data or [], video_data or [], audio_data or []
        use_audio = getattr(request_obj, "use_audio_in_video", False)
        loop = asyncio.get_running_loop()
        run = functools.partial(loop.run_in_executor, self.io_executor)

        try:
            loaded = await asyncio.gather(
                *(run(self.load_image, x) for x in images),
                *(run(self.load_video, x, use_audio) for x in videos),
                *(run(self.load_audio, x) for x in audios),
            )
        except CLIENT_MEDIA_EXCEPTIONS as e:
            raise ValueError(f"Cannot load media: {e}") from e

        if isinstance(input_text, list):
            token_ids = input_text
        else:
            # The chat template already wrote the BOS.
            add_bos = not input_text.startswith(self._tokenizer.bos_token)
            token_ids = self._tokenizer.encode(input_text, add_special_tokens=add_bos)

        num_images = len(images)
        num_visual = num_images + len(videos)
        pending = {
            self.image_token_id: deque(loaded[:num_images]),
            self.video_token_id: deque(loaded[num_images:num_visual]),
            self.audio_token_id: deque(loaded[num_visual:]),
        }
        input_ids: List[int] = []
        items: List[MultimodalDataItem] = []
        video_index = 0

        for token_id in token_ids:
            if token_id == self.video_audio_token_id:
                raise ValueError(
                    "Soundtrack placeholders are added by the server; "
                    "send the video with use_audio_in_video instead"
                )

            queue = pending.get(token_id)

            if queue is None:
                input_ids.append(token_id)
                continue

            if not queue:
                raise ValueError(
                    "More media placeholders in the prompt than media items"
                )

            if token_id == self.image_token_id:
                crops = queue.popleft()
                span = self.append_run(
                    input_ids, self.image_token_id, self.frame_tokens * len(crops)
                )
                items.append(
                    MultimodalDataItem(
                        modality=Modality.IMAGE, feature=crops, offsets=[span]
                    )
                )
            elif token_id == self.video_token_id:
                self.expand_video(queue.popleft(), video_index, input_ids, items)
                video_index += 1
            else:
                self.expand_audio(*queue.popleft(), input_ids, items)

        if any(pending.values()):
            raise ValueError("More media items than media placeholders in the prompt")

        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=items,
            im_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            audio_token_id=self.audio_token_id,
        )
