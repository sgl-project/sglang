"""MiMo-V2-ASR multimodal processor"""

import asyncio
import io
import math
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

import numpy as np
import pybase64
import requests
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
from sglang.utils import logger
from torchcodec.decoders import AudioDecoder

try:
    import torchaudio
    from torchaudio.transforms import MelSpectrogram
except ImportError:
    print(
        "[Warning] torchaudio is not installed, audio inference will not be supported"
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


TextInput = str | list[int]


@dataclass
class Content:
    type: Literal["text", "audio"]
    content: TextInput | AudioInput
    is_target: Optional[bool] = None

    def __post_init__(self):
        if self.type not in ["text", "audio"]:
            raise ValueError(f"type must be one of text, audio, but got {self.type}")
        if self.type == "text":
            if not isinstance(self.content, (str, list)) or (
                isinstance(self.content, list)
                and not all(isinstance(item, int) for item in self.content)
            ):
                raise ValueError(
                    f"content must be a str or a list of ints, but got {type(self.content)}"
                )
        elif self.type == "audio":
            if not isinstance(self.content, AudioInput):
                raise ValueError(
                    f"content must be a AudioInput, but got {type(self.content)}"
                )


@dataclass
class MiMoASRInputSample:
    input_ids: torch.Tensor
    labels: Optional[torch.Tensor]
    audio_inputs: list[torch.Tensor]
    position_ids: Optional[torch.Tensor] = None
    rope_deltas: Optional[torch.Tensor] = None
    extra: dict = field(default_factory=dict)


class MiMoASRBaseProcessor:
    def __init__(
        self,
        tokenizer,
        audio_kernel_size=3,
        audio_stride_size=2,
        audio_avg_pooler=2,
        audio_sampling_rate=24000,
        audio_nfft=960,
        audio_hop_length=240,
        audio_window_size=960,
        audio_fmin=0,
        audio_fmax=None,
        audio_n_mels=128,
        audio_segment_size=6000,
        audio_channels=8,
        audio_group_size=4,
        audio_input_id_per_second=25,
        audio_zeroemb_idx=4096,
        audio_token_id=None,
        audio_start_token_id=None,
        audio_end_token_id=None,
        pad_token_id=None,
        rope_type="rope",
        device=None,
        **kwargs,
    ):
        self.tokenizer = tokenizer

        if device is None:
            self.device = None
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        self.rope_type = rope_type
        if self.rope_type == "1d":
            self.rope_type = "rope"
        assert self.rope_type in ["rope", "mrope"]

        self.audio_token_id = audio_token_id
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id
        self.pad_token_id = pad_token_id

        self.audio_sampling_rate = audio_sampling_rate
        self.audio_nfft = audio_nfft
        self.audio_hop_length = audio_hop_length
        self.audio_window_size = audio_window_size
        self.audio_fmin = audio_fmin
        self.audio_fmax = audio_fmax
        self.audio_n_mels = audio_n_mels

        self.audio_segment_size = audio_segment_size

        self.audio_kernel_size = audio_kernel_size
        self.audio_stride_size = audio_stride_size
        self.audio_avg_pooler = audio_avg_pooler

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
        self._resamplers = OrderedDict()
        self._resamplers_max = 16

        self.audio_channels = audio_channels
        self.audio_group_size = audio_group_size
        self.audio_input_id_per_second = audio_input_id_per_second
        if isinstance(audio_zeroemb_idx, int):
            self.audio_zeroemb_idxs = torch.tensor(
                [audio_zeroemb_idx] * self.audio_channels, dtype=torch.int32
            )
        elif isinstance(audio_zeroemb_idx, list):
            if len(audio_zeroemb_idx) == 1:
                self.audio_zeroemb_idxs = torch.tensor(
                    audio_zeroemb_idx * self.audio_channels, dtype=torch.int32
                )
            elif len(audio_zeroemb_idx) == self.audio_channels:
                self.audio_zeroemb_idxs = torch.tensor(
                    audio_zeroemb_idx, dtype=torch.int32
                )
            else:
                raise ValueError(
                    f"audio_zeroemb_idx must be a list of 1 or {self.audio_channels} integers, but got {len(audio_zeroemb_idx)}"
                )
        else:
            raise ValueError(
                f"audio_zeroemb_idx must be an integer or a list of {self.audio_channels} integers, but got {type(audio_zeroemb_idx)}"
            )

        self.http_session = requests.Session()
        for k in kwargs:
            logger.info(f"[Warning] Ignored unknown parameter {k} for MiMoASRProcessor")

    @property
    def mel_spectrogram(self):
        if self._mel_spectrogram is None:
            self._mel_spectrogram = MelSpectrogram(**self.mel_spectrogram_kwargs)
        return self._mel_spectrogram

    def preprocess_audio(self, audio: str | bytes):
        """
        - Input: audio filename string, bytes, or tuple of (waveform, original_sr)
        - Output:
            - mel spectrogram: torch.Tensor (T, n_mels)
            - number of tokens: int
        """
        assert isinstance(audio, (str, bytes, tuple)), (
            f"audio must be a str, bytes or tuple, but got {type(audio)}"
        )
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

        audio_token_len = spec.shape[0] + 3 - self.audio_kernel_size
        audio_token_len = (
            audio_token_len + 2 - self.audio_kernel_size
        ) // self.audio_stride_size + 1
        audio_token_len = audio_token_len // self.audio_avg_pooler + int(
            audio_token_len % self.audio_avg_pooler != 0
        )
        audio_token_len = math.ceil(audio_token_len / self.audio_group_size)

        return spec, audio_token_len

    def process_audio(self, audio: AudioInput):
        audio = audio.audio
        if isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio).float()
            audio = (waveform, self.audio_sampling_rate)
        if isinstance(audio, (str, bytes, tuple)):
            audio_spec, audio_token_len = self.preprocess_audio(audio)
            return audio_spec, audio_token_len

        assert audio.shape[1] >= self.audio_channels, (
            f"audio must have at least {self.audio_channels} channels, but got {audio.shape[1]}"
        )
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

    def _process_text_content(self, content, verbose):
        if isinstance(content.content, str):
            _input_ids = self.tokenizer.encode(content.content)
        else:
            _input_ids = content.content
        _labels = _input_ids if content.is_target else None

        verbose_str = ""
        if verbose:
            if isinstance(content.content, str):
                verbose_str = f"Text: [{repr(content.content)}]\n"
            else:
                verbose_str = (
                    f"Text: [{repr(self.tokenizer.decode(content.content))}]\n"
                )

        return {"input_ids": _input_ids, "labels": _labels, "verbose": verbose_str}

    def _process_audio_content(self, content, verbose):
        processed_audio = self.process_audio(content.content)
        if isinstance(processed_audio, tuple):
            is_tokenized = False
            audio_spec, audio_token_len = processed_audio
            audio_input = audio_spec
        else:
            is_tokenized = True
            audio_token_len = processed_audio.shape[0]
            audio_input = processed_audio
        _input_ids = (
            [self.audio_start_token_id]
            + [self.audio_token_id] * audio_token_len
            + [self.audio_end_token_id]
        )

        verbose_str = ""
        if verbose:
            verbose_str = f"Audio (is_tokenized={is_tokenized}): [<|sosp|>{audio_token_len}*<|empty|><|eosp|>]\n"

        return {
            "input_ids": _input_ids,
            "audio_input": audio_input,
            "is_tokenized": is_tokenized,
            "verbose": verbose_str,
        }

    def process(self, contents: list[Content], verbose: bool = False):
        input_ids, labels = [], []
        audio_inputs = []
        is_audio_tokenized = []
        extra = {}
        verbose_str = ""

        for content in contents:
            _labels = None

            if content.type == "text":
                result = self._process_text_content(content, verbose)
                _labels = result["labels"]

            elif content.type == "audio":
                result = self._process_audio_content(content, verbose)
                audio_inputs.append(result["audio_input"])
                is_audio_tokenized.append(result["is_tokenized"])

            input_ids.extend(result["input_ids"])
            labels.extend(_labels or [self.pad_token_id] * len(result["input_ids"]))
            verbose_str += result.get("verbose", "")

        input_ids = torch.tensor(input_ids)
        labels = np.roll(labels, shift=-1)
        labels[-1] = self.pad_token_id
        labels = torch.tensor(labels)

        if len(is_audio_tokenized) > 0:
            assert all(is_audio_tokenized) or not any(is_audio_tokenized), (
                "All audio inputs must be tokenized or not tokenized"
            )
            extra["is_audio_tokenized"] = is_audio_tokenized[0]

        if self.rope_type == "rope":
            position_ids = torch.arange(input_ids.shape[0]).expand(3, -1)
            rope_deltas = torch.zeros((1, 1), dtype=torch.int32)
        elif self.rope_type == "mrope":
            position_ids = torch.arange(input_ids.shape[0]).expand(3, -1)
            rope_deltas = torch.zeros((1, 1), dtype=torch.int32)

        if verbose:
            print(verbose_str.strip())

        return MiMoASRInputSample(
            input_ids=input_ids,
            labels=labels,
            audio_inputs=audio_inputs,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            extra=extra,
        )


class MiMoV2ASRProcessor(BaseMultimodalProcessor):
    models = [MiMoV2ASRForCausalLM]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.tokenizer = _processor

        self.audio_sample_rate = 24000

        self.audio_token_id = int(self.tokenizer.convert_tokens_to_ids("<|empty|>"))
        self.audio_start_token_id = int(
            self.tokenizer.convert_tokens_to_ids("<|sosp|>")
        )
        self.audio_end_token_id = int(self.tokenizer.convert_tokens_to_ids("<|eosp|>"))

        self.mimo_processor = MiMoASRBaseProcessor(
            tokenizer=self.tokenizer,
            audio_token_id=self.audio_token_id,
            audio_start_token_id=self.audio_start_token_id,
            audio_end_token_id=self.audio_end_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            rope_type="rope",
            device=None,
        )
        self._processor = self.mimo_processor

        self.regex = re.compile(r"<\|sosp\|>(?:<\|empty\|>)+<\|eosp\|>")

        self.mm_tokens = MultimodalSpecialTokens(
            audio_token="<|sosp|><|empty|><|eosp|>",
            audio_token_id=self.audio_token_id,
            audio_token_regex=self.regex,
        ).build(_processor)

    def process_mm_data(
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ) -> dict:
        if audios and not self.regex.search(input_text or ""):
            input_text = f"{self.mm_tokens.audio_token}{input_text or ''}"

        processed_audios = []

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
                        (audio_tensor.cpu().contiguous(), self.audio_sample_rate)
                    )
                else:
                    processed_audios.append(audio_tensor.cpu().contiguous())

        contents = []

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
                                Content(type="audio", content=AudioInput(audio=audio))
                            )
                        except StopIteration:
                            pass
                else:
                    if text_part:
                        contents.append(Content(type="text", content=text_part))
        else:
            contents.extend(
                Content(type="audio", content=AudioInput(audio=audio))
                for audio in processed_audios
            )

        if not contents:
            input_ids = self.mimo_processor.tokenizer(
                input_text or "",
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids
            return {"input_ids": input_ids}

        input_sample = self.mimo_processor.process(contents, verbose=False)

        ret = {
            "input_ids": input_sample.input_ids,
            "mrope_positions": getattr(input_sample, "position_ids", None),
            "mrope_position_delta": getattr(input_sample, "rope_deltas", None),
        }
        audio_inputs = getattr(input_sample, "audio_inputs", None)
        if audio_inputs is not None and len(audio_inputs) > 0:
            ret["audio_features"] = audio_inputs
            audio_attention_mask = getattr(
                input_sample, "audio_attention_mask", None
            ) or getattr(input_sample, "feature_attention_mask", None)
            if audio_attention_mask is not None:
                ret["audio_attention_mask"] = audio_attention_mask
            audio_feature_lens = getattr(input_sample, "audio_feature_lens", None)
            if audio_feature_lens is None:
                audio_feature_lens = audio_attention_mask
                if audio_feature_lens is not None:
                    audio_feature_lens = audio_feature_lens.sum(dim=-1)
            if audio_feature_lens is not None:
                ret["audio_feature_lens"] = audio_feature_lens

        device = kwargs.get("device")
        if device:
            for key in (
                "audio_features",
                "audio_feature_lens",
            ):
                if key in ret and isinstance(ret[key], torch.Tensor):
                    ret[key] = ret[key].to(device)

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
        if audio_data and not self.regex.search(input_text):
            input_text = f"{self.mm_tokens.audio_token}{input_text}"

        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=[],
            video_data=[],
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
            audio_sample_rate=self.audio_sample_rate,
        )
        multimodal_tokens_pattern = self.mm_tokens.get_combined_regex()

        raw_audio_data = audio_data or []

        loaded_audio_iter = iter(base_output.audios)
        raw_audio_iter = iter(raw_audio_data)

        text_parts = re.split(multimodal_tokens_pattern, base_output.input_text)
        contents = []

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

                    contents.append(
                        Content(
                            type="audio",
                            content=AudioInput(
                                audio=audio_source,
                            ),
                        )
                    )
            else:
                if text_part:
                    contents.append(Content(type="text", content=text_part))

        loop = asyncio.get_running_loop()
        try:
            input_sample = await loop.run_in_executor(
                self.io_executor,
                lambda: self.mimo_processor.process(contents, verbose=False),
            )
        except RuntimeError as e:
            logger.error(f"MiMo ASR processor failed in process_mm_data_async: {e}")
            raise ValueError(f"Multimodal data is corrupted or cannot be decoded: {e}")

        input_ids = input_sample.input_ids.flatten()
        mm_items: list[MultimodalDataItem] = []
        audio_inputs = getattr(input_sample, "audio_inputs", None)
        if audio_inputs is not None and len(audio_inputs) > 0:
            audio_item = MultimodalDataItem(
                modality=Modality.AUDIO,
                feature=audio_inputs,
                offsets=self.get_mm_items_offset(
                    input_ids=input_ids, mm_token_id=self.mimo_processor.audio_token_id
                ),
            )
            audio_feature_lens = getattr(input_sample, "audio_feature_lens", None)
            if audio_feature_lens is None:
                audio_attention_mask = getattr(
                    input_sample, "audio_attention_mask", None
                ) or getattr(input_sample, "feature_attention_mask", None)
                if audio_attention_mask is not None:
                    audio_feature_lens = audio_attention_mask.sum(dim=-1)
            if audio_feature_lens is not None:
                audio_item.audio_feature_lens = audio_feature_lens
            mm_items.append(audio_item)

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            audio_token_id=self.mimo_processor.audio_token_id,
            audio_start_id=self.mimo_processor.audio_start_token_id,
            audio_end_id=self.mimo_processor.audio_end_token_id,
            mrope_positions=input_sample.position_ids,
            mrope_position_delta=input_sample.rope_deltas,
        )
