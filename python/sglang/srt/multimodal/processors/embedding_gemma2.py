# Copyright 2025 SGLang Team
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

"""Multimodal processor for EmbeddingGemma2Model (EmbeddingGemma v2)."""

import logging
from typing import Any

import numpy as np
import torch
from sglang.srt.models.embedding_gemma2 import EmbeddingGemma2Model
from sglang.srt.multimodal.processors.base_processor import Modality
from sglang.srt.multimodal.processors.gemma4 import Gemma4SGLangProcessor
from sglang.srt.utils.common import load_audio
from sglang.srt.utils.video_decoder import VideoDecoderWrapper

logger = logging.getLogger(__name__)


def _load_audio_from_memory(
    audio_file: Any, sr: int | None = 16000, mono: bool = True
) -> np.ndarray:
    """Load or resample audio waveforms from in-memory numpy array, torch tensor, tuple, or path."""
    if sr is None:
        sr = 16000

    input_sr = None
    if isinstance(audio_file, tuple) and len(audio_file) == 2:
        audio_file, input_sr = audio_file

    if isinstance(audio_file, torch.Tensor):
        audio_file = audio_file.detach().cpu().numpy()

    if isinstance(audio_file, np.ndarray):
        waveform = audio_file.astype(np.float32, copy=False)
        if waveform.ndim > 1 and mono:
            if waveform.shape[0] <= 8 and waveform.shape[0] < waveform.shape[-1]:
                waveform = waveform.mean(axis=0)
            else:
                waveform = waveform.mean(axis=-1)
        if (
            input_sr is not None
            and sr is not None
            and input_sr != sr
            and len(waveform) > 0
        ):
            try:
                import torchaudio  # type: ignore[import-untyped]

                audio_tensor = torch.from_numpy(waveform).float()
                resampler = torchaudio.transforms.Resample(
                    orig_freq=input_sr, new_freq=sr
                )
                waveform = resampler(audio_tensor).numpy().astype(np.float32)
            except Exception:  # noqa: BLE001
                from scipy import signal  # type: ignore[import-untyped]

                num_samples = int(len(waveform) * float(sr) / input_sr)
                waveform = signal.resample(waveform, num_samples).astype(np.float32)
        return waveform

    return load_audio(audio_file, sr=sr, mono=mono)


class EmbeddingGemma2SGLangProcessor(Gemma4SGLangProcessor):
    """Multimodal input processor for EmbeddingGemma2Model.

    Preprocesses image, video, and audio inputs for EmbeddingGemma v2:
      - Prompt retention: retains prompt task/title prefixes verbatim, disabling the
        anchored prompt suppression used in generative Gemma4.
      - Video processing: forwards video frames and metadata with do_sample_frames=True,
        handling 1fps/cap32 uniform frame sampling.
      - Audio processing: provides unpadded raw waveforms (16kHz mono float32).
    """

    models: list[Any] = [EmbeddingGemma2Model]  # type: ignore[assignment]  # noqa: RUF012

    # Plain nvJPEG skips fancy chroma upsampling and drifts from the PIL decode the
    # checkpoint was validated with (embedding cos ~0.99 vs HF); nvjpeg_fancy matches it.
    gpu_image_decode = "nvjpeg_fancy"

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.disable_fast_image_processor = True

        # Resolve special tokens
        self.IM_START_TOKEN_ID = getattr(hf_config, "boi_token_id", 255999)
        self.IM_END_TOKEN_ID = getattr(hf_config, "eoi_token_id", 258882)
        self.IMAGE_TOKEN_ID = getattr(hf_config, "image_token_id", 258880)
        self.AUDIO_TOKEN_ID = getattr(hf_config, "audio_token_id", 258881)
        self.VIDEO_TOKEN_ID = getattr(hf_config, "video_token_id", 258884)
        self.AUDIO_START_TOKEN_ID = getattr(hf_config, "boa_token_id", 256000)
        self.AUDIO_END_TOKEN_ID = getattr(
            hf_config, "eoa_token_id", getattr(hf_config, "eoa_token_index", 258883)
        )

    @classmethod
    def _load_single_item(
        cls,
        data: Any,
        modality: Modality,
        frame_count_limit: int | None = None,
        audio_sample_rate: int | None = None,
        discard_alpha_channel: bool = True,
    ) -> Any:
        if modality == Modality.AUDIO:
            return _load_audio_from_memory(data, sr=audio_sample_rate, mono=True)
        return super()._load_single_item(
            data,
            modality=modality,
            frame_count_limit=frame_count_limit,
            audio_sample_rate=audio_sample_rate,
            discard_alpha_channel=discard_alpha_channel,
        )

    def _video_decoder_to_tensor(self, vdw: VideoDecoderWrapper) -> torch.Tensor:
        total = len(vdw)
        indices = list(range(total))
        frames_np = vdw.get_frames_at(indices)  # (N, H, W, C)
        return torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()

    def process_mm_data(  # type: ignore[override]
        self,
        input_text: Any,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        audios: list[Any] | None = None,
        **kwargs: Any,
    ):
        if "num_frames" in kwargs or "num_frames" in kwargs.get("videos_kwargs", {}):
            raise ValueError(
                "Sampling with `num_frames` is not supported for EmbeddingGemma2. "
                "Use `fps` and `max_frames` instead."
            )

        # 1. Prompt retention: do NOT suppress task/title prefixes.
        # input_text passes through verbatim.

        # 2. Audio: unpadded raw waveforms (16kHz mono float32)
        if audios:
            loaded_audios = []
            for a in audios:
                if isinstance(a, (np.ndarray, torch.Tensor, tuple)):
                    loaded_audios.append(_load_audio_from_memory(a, sr=16000, mono=True))
                else:
                    loaded_audios.append(np.asarray(a, dtype=np.float32))
            kwargs["audio"] = loaded_audios
            kwargs.setdefault("audio_kwargs", {})["truncation"] = False
            audios = None

        # 3. Video: forward (frames, metadata) with do_sample_frames=True
        if videos:
            unpacked_videos = []
            video_metadata_list = []
            has_metadata = False
            for v in videos:
                if isinstance(v, VideoDecoderWrapper):
                    tensor_v = self._video_decoder_to_tensor(v)
                    unpacked_videos.append(tensor_v)
                    total_f = len(tensor_v)
                    fps = getattr(v, "avg_fps", None)
                    dur = getattr(v, "duration", None)
                    if dur is None and fps and fps > 0:
                        dur = total_f / fps
                    video_metadata_list.append(
                        {
                            "total_num_frames": total_f,
                            "fps": fps,
                            "duration": dur,
                            "frames_indices": list(range(total_f)),
                        }
                    )
                    has_metadata = True
                elif isinstance(v, tuple) and len(v) == 2:
                    frames, meta = v
                    if isinstance(meta, dict):
                        meta = {
                            k: val for k, val in meta.items() if k != "do_sample_frames"
                        }
                    unpacked_videos.append(frames)
                    video_metadata_list.append(meta)
                    has_metadata = True
                else:
                    unpacked_videos.append(v)
                    total_f = len(v) if hasattr(v, "__len__") else 1
                    video_metadata_list.append(
                        {
                            "total_num_frames": total_f,
                            "fps": None,
                            "duration": None,
                            "frames_indices": list(range(total_f)),
                        }
                    )

            videos = unpacked_videos
            if has_metadata:
                videos_kwargs = kwargs.setdefault("videos_kwargs", {})
                videos_kwargs.setdefault("video_metadata", video_metadata_list)
            # EmbeddingGemma2 HF processor requires do_sample_frames=True
            kwargs.setdefault("do_sample_frames", True)

        return super(Gemma4SGLangProcessor, self).process_mm_data(
            input_text, images=images, videos=videos, audios=audios, **kwargs
        )
