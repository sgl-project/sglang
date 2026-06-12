"""Stateful audio preprocessing pipeline shared by MiMo multimodal and ASR processors."""

import io
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pybase64
import requests
import torch

from sglang.utils import logger

try:
    from torchcodec.decoders import AudioDecoder
except ImportError:
    logger.warning(
        "torchcodec is not installed; audio inputs will fail at request time"
    )
    AudioDecoder = None

try:
    import torchaudio
    from torchaudio.transforms import MelSpectrogram
except ImportError:
    logger.warning(
        "torchaudio is not installed; audio inputs will fail at request time"
    )
    torchaudio = None
    MelSpectrogram = None


@dataclass
class AudioInput:
    """
    if audio is str or bytes, only load it as mel spectrogram.
    if audio is tuple, it is (waveform, original_sr)
    if audio is torch.Tensor, it is tokenized input ids with shape (T, n_vq+).
    if audio is np.ndarray, it is a pre-loaded waveform (1D, already resampled).
    """

    audio: str | bytes | tuple | torch.Tensor | np.ndarray

    def __post_init__(self):
        if not isinstance(self.audio, (str, bytes, tuple, torch.Tensor, np.ndarray)):
            raise ValueError(
                f"audio must be a str, bytes, tuple, torch.Tensor, or np.ndarray, but got {type(self.audio)}"
            )
        if isinstance(self.audio, tuple):
            if (
                len(self.audio) != 2
                or not isinstance(self.audio[0], torch.Tensor)
                or not isinstance(self.audio[1], (int, float))
            ):
                raise ValueError(
                    f"audio must be a tuple of (waveform-T, original_sr-int/float), but got {len(self.audio)} elements and {type(self.audio[0])} and {type(self.audio[1])}"
                )
            if self.audio[0].ndim != 1:
                raise ValueError(
                    f"waveform must be a 1D tensor, but got {self.audio[0].ndim}D tensor"
                )
            if self.audio[1] <= 0:
                raise ValueError(
                    f"original_sr must be a positive number, but got {self.audio[1]}"
                )
        if isinstance(self.audio, torch.Tensor) and self.audio.ndim != 2:
            raise ValueError(
                f"audio must be a 2D tensor, but got {self.audio.ndim}D tensor"
            )


class MiMoAudioPipeline:
    """Stateful audio preprocessing pipeline.

    Composable: held by both MiMoProcessor (multimodal) and MiMoV2ASRProcessor.
    Owns the mel spectrogram, resampler cache, http session, and the special
    token ids for ``<|sosp|> <|empty|>* <|eosp|>`` placeholders.
    """

    def __init__(
        self,
        *,
        audio_token_id: int,
        audio_start_token_id: int,
        audio_end_token_id: int,
        audio_kernel_size: int = 3,
        audio_stride_size: int = 2,
        audio_avg_pooler: int = 2,
        audio_group_size: int = 4,
        audio_channels: int = 8,
        audio_sampling_rate: int = 24000,
        audio_nfft: int = 960,
        audio_hop_length: int = 240,
        audio_window_size: int = 960,
        audio_fmin: int = 0,
        audio_fmax: Optional[int] = None,
        audio_n_mels: int = 128,
        audio_input_id_per_second: int = 25,
        max_resamplers: int = 16,
    ) -> None:
        self.audio_token_id = audio_token_id
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id

        self.audio_kernel_size = audio_kernel_size
        self.audio_stride_size = audio_stride_size
        self.audio_avg_pooler = audio_avg_pooler
        self.audio_group_size = audio_group_size
        self.audio_channels = audio_channels

        self.audio_sampling_rate = audio_sampling_rate
        self.audio_nfft = audio_nfft
        self.audio_hop_length = audio_hop_length
        self.audio_window_size = audio_window_size
        self.audio_fmin = audio_fmin
        self.audio_fmax = audio_fmax
        self.audio_n_mels = audio_n_mels
        self.audio_input_id_per_second = audio_input_id_per_second

        self.mel_spectrogram_kwargs = dict(
            sample_rate=audio_sampling_rate,
            n_fft=audio_nfft,
            hop_length=audio_hop_length,
            win_length=audio_window_size,
            f_min=audio_fmin,
            f_max=audio_fmax,
            n_mels=audio_n_mels,
            power=1.0,
            center=True,
        )
        self._mel_spectrogram = None
        self._resamplers: OrderedDict[int, torchaudio.transforms.Resample] = (
            OrderedDict()
        )
        self._resamplers_max = max_resamplers

        self.http_session = requests.Session()

    @property
    def audio_token_per_second(self) -> float:
        return self.audio_input_id_per_second / self.audio_group_size

    @staticmethod
    def _ensure_audio_dependencies() -> None:
        if torchaudio is None or MelSpectrogram is None:
            raise RuntimeError(
                "torchaudio is required for audio inputs; install torchaudio"
            )

    @property
    def mel_spectrogram(self):
        self._ensure_audio_dependencies()
        if self._mel_spectrogram is None:
            self._mel_spectrogram = MelSpectrogram(**self.mel_spectrogram_kwargs)
        return self._mel_spectrogram

    def compute_audio_token_len(self, mel_len: int) -> int:
        n = mel_len + 3 - self.audio_kernel_size
        n = (n + 2 - self.audio_kernel_size) // self.audio_stride_size + 1
        n = n // self.audio_avg_pooler + int(n % self.audio_avg_pooler != 0)
        return math.ceil(n / self.audio_group_size)

    def preprocess_audio(self, audio):
        """Load audio source → log-mel spectrogram + token length.

        Input: filename string, bytes, or tuple of (waveform, original_sr).
        Output: (mel-spectrogram tensor [T, n_mels], audio_token_len int).
        """
        self._ensure_audio_dependencies()
        assert isinstance(
            audio, (str, bytes, tuple)
        ), f"audio must be a str, bytes or tuple, but got {type(audio)}"
        if isinstance(audio, tuple):
            waveform, original_sr = audio
        else:
            if isinstance(audio, bytes):
                file = io.BytesIO(audio)
            elif isinstance(audio, str):
                if audio.startswith("data:"):
                    file = io.BytesIO(
                        pybase64.b64decode(audio.split(",")[1], validate=True)
                    )
                elif audio.startswith("http://") or audio.startswith("https://"):
                    dl_start = time.perf_counter()
                    timeout = int(os.getenv("REQUEST_TIMEOUT", "5"))
                    try:
                        response = self.http_session.get(
                            audio, stream=True, timeout=timeout
                        )
                        dl_elapsed_ms = (time.perf_counter() - dl_start) * 1000
                        if dl_elapsed_ms > 1000.0:
                            content_len = len(response.content)
                            logger.warning(
                                f"Slow audio download: {dl_elapsed_ms:.2f}ms, "
                                f"size={content_len / 1024:.1f}KB, url={audio}"
                            )
                        file = io.BytesIO(response.content)
                        response.close()
                    except Exception as e:
                        dl_elapsed_ms = (time.perf_counter() - dl_start) * 1000
                        logger.error(
                            f"Failed to download audio: {dl_elapsed_ms:.2f}ms, "
                            f"error={type(e).__name__}: {e}, url={audio}"
                        )
                        raise
                else:
                    file = audio
            if AudioDecoder is None:
                raise RuntimeError(
                    "torchcodec is required for audio decoding; install with `pip install torchcodec`."
                )
            try:
                samples = AudioDecoder(file).get_all_samples()
            except RuntimeError as e:
                audio_source = (
                    audio
                    if isinstance(audio, str)
                    and (audio.startswith("http://") or audio.startswith("https://"))
                    else "<bytes or base64>"
                )
                logger.error(f"Failed to decode audio: {e}, source={audio_source}")
                raise ValueError(
                    f"Invalid audio format: source={audio_source}, detail={e}"
                ) from e
            waveform = samples.data
            original_sr = samples.sample_rate

        if original_sr != self.audio_sampling_rate:
            if original_sr in self._resamplers:
                self._resamplers.move_to_end(original_sr)
            else:
                if len(self._resamplers) >= self._resamplers_max:
                    self._resamplers.popitem(last=False)
                self._resamplers[original_sr] = torchaudio.transforms.Resample(
                    orig_freq=original_sr, new_freq=self.audio_sampling_rate
                )
            waveform = self._resamplers[original_sr](waveform)
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        spec = self.mel_spectrogram(waveform[None, :])
        spec = torch.log(torch.clip(spec, min=1e-7)).squeeze()
        spec = spec.transpose(0, 1)

        audio_token_len = self.compute_audio_token_len(spec.shape[0])
        return spec, audio_token_len

    def process_audio(self, audio_input: AudioInput):
        """Dispatch on the underlying audio payload.

        - str/bytes/tuple/np.ndarray waveform → returns (mel-spec, token_len) tuple
        - 2D tensor of pre-tokenized audio codes → returns padded codes tensor
          shaped [T//group, group, channels]
        """
        audio = audio_input.audio
        if isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio).float()
            audio = (waveform, self.audio_sampling_rate)
        if isinstance(audio, (str, bytes, tuple)):
            return self.preprocess_audio(audio)

        assert (
            audio.shape[1] >= self.audio_channels
        ), f"audio must have at least {self.audio_channels} channels, but got {audio.shape[1]}"
        T = audio.shape[0]
        audio = audio[:, : self.audio_channels].to(torch.long)
        padded_T = (
            (T + self.audio_group_size - 1)
            // self.audio_group_size
            * self.audio_group_size
        )
        padded_audio = torch.cat(
            [
                audio,
                torch.zeros(padded_T - T, self.audio_channels, dtype=torch.long)
                + audio[-1, :],
            ],
            dim=0,
        )
        padded_audio = padded_audio.reshape(
            padded_T // self.audio_group_size,
            self.audio_group_size,
            self.audio_channels,
        )
        return padded_audio

    def build_audio_placeholder_ids(self, audio_token_len: int) -> list[int]:
        return (
            [self.audio_start_token_id]
            + [self.audio_token_id] * audio_token_len
            + [self.audio_end_token_id]
        )

    def process_audio_input(self, audio_input: AudioInput) -> dict:
        """Run process_audio and produce the placeholder input_ids.

        Replaces the duplicated _process_audio_content bodies in both processors.
        Returns dict with input_ids, audio_input (mel or codes), and is_tokenized.
        """
        processed = self.process_audio(audio_input)
        if isinstance(processed, tuple):
            is_tokenized = False
            audio_spec, audio_token_len = processed
            payload = audio_spec
        else:
            is_tokenized = True
            audio_token_len = processed.shape[0]
            payload = processed

        return {
            "input_ids": self.build_audio_placeholder_ids(audio_token_len),
            "audio_input": payload,
            "audio_token_len": audio_token_len,
            "is_tokenized": is_tokenized,
        }
