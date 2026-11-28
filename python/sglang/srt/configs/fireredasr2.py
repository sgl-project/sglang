# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/processors/fireredasr2.py

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BatchFeature
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor

AUDIO_TOKEN = "<|AUDIO|>"
DEFAULT_ASR_PROMPT = (
    "<|im_start|>user\n<|AUDIO|>请转写音频为文字<|im_end|>\n<|im_start|>assistant\n"
)


def get_audio_token_count(num_frames: int, downsample_rate: int = 2) -> int:
    # Six context frames are appended before the two stride-2 convolutions.
    return max(1, ((num_frames + 3) // 4) // downsample_rate)


class FireRedASR2FeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["input_features", "speech_lengths", "fake_token_lengths"]

    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        padding_value=0.0,
        chunk_length=30,
        max_length=3000,
        dim=80,
        means=None,
        inverse_std_variences=None,
        downsample_rate=2,
        left_context=3,
        right_context=3,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        if left_context != 3 or right_context != 3:
            raise ValueError("FireRedASR2 requires three left and right context frames")
        if dim != feature_size or means is None or inverse_std_variences is None:
            raise ValueError("FireRedASR2 requires checkpoint CMVN statistics")
        self.chunk_length = chunk_length
        self.max_length = max_length
        self.downsample_rate = downsample_rate
        self.means = np.asarray(means)
        self.inverse_std_variences = np.asarray(inverse_std_variences)
        if self.means.shape != (dim,) or self.inverse_std_variences.shape != (dim,):
            raise ValueError("CMVN statistics must match the feature dimension")

    def __call__(self, raw_speech, sampling_rate=None, return_tensors="pt", **kwargs):
        from torchaudio.compliance.kaldi import fbank

        if sampling_rate != self.sampling_rate:
            raise ValueError(f"FireRedASR2 requires {self.sampling_rate} Hz audio")
        features, lengths, counts = [], [], []
        for speech in raw_speech:
            speech = np.asarray(speech, dtype=np.float32)
            if (
                speech.ndim != 1
                or not 400 <= speech.size <= self.chunk_length * sampling_rate
            ):
                raise ValueError(
                    "FireRedASR2 requires mono audio between 25 ms and 30 s"
                )
            # The checkpoint was trained with Kaldi PCM amplitudes.
            waveform = torch.from_numpy(speech * 32768).unsqueeze(0)
            mel = fbank(
                waveform,
                num_mel_bins=self.feature_size,
                sample_frequency=sampling_rate,
                frame_length=25.0,
                frame_shift=10.0,
                dither=0.0,
                snip_edges=True,
                window_type="povey",
                low_freq=20.0,
                high_freq=0.0,
                preemphasis_coefficient=0.97,
                remove_dc_offset=True,
                use_energy=False,
                energy_floor=0.0,
            )
            mel = torch.from_numpy(
                (mel.numpy() - self.means) * self.inverse_std_variences
            ).float()
            length = mel.shape[0]
            features.append(F.pad(mel, (0, 0, 0, self.max_length - length)))
            lengths.append(length)
            counts.append(get_audio_token_count(length, self.downsample_rate))
        return BatchFeature(
            {
                "input_features": torch.stack(features),
                "speech_lengths": torch.tensor(lengths),
                "fake_token_lengths": torch.tensor(counts),
            },
            tensor_type=return_tensors,
        )


class FireRedASR2Processor:
    def __init__(self, feature_extractor, tokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.audio_token = AUDIO_TOKEN
        self.audio_token_id = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN)

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        feature_extractor = FireRedASR2FeatureExtractor.from_pretrained(
            model_path, **kwargs
        )
        return cls(feature_extractor, tokenizer)

    def preprocess_fingerprint_payload(self) -> dict:
        extractor = self.feature_extractor
        return {
            "sampling_rate": extractor.sampling_rate,
            "feature_size": extractor.feature_size,
            "chunk_length": extractor.chunk_length,
            "max_length": extractor.max_length,
            "downsample_rate": extractor.downsample_rate,
            "means": extractor.means.tolist(),
            "inverse_std_variences": extractor.inverse_std_variences.tolist(),
            "audio_token_id": self.audio_token_id,
        }

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def __call__(self, text, audios=None, return_tensors="pt", **kwargs):
        if isinstance(text, str):
            text = [text]
        if audios is None:
            return self.tokenizer(text, return_tensors=return_tensors, **kwargs)
        if sum(sample.count(AUDIO_TOKEN) for sample in text) != len(audios):
            raise ValueError(
                "The number of audio placeholders must match the audio inputs"
            )
        inputs = self.feature_extractor(
            audios,
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors=return_tensors,
        )
        counts = iter(inputs["fake_token_lengths"].tolist())
        expanded = []
        for sample in text:
            parts = sample.split(AUDIO_TOKEN)
            expanded.append(
                parts[0]
                + "".join(AUDIO_TOKEN * next(counts) + part for part in parts[1:])
            )
        inputs.update(self.tokenizer(expanded, return_tensors=return_tensors, **kwargs))
        return inputs
