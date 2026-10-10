"""Synthetic audio workloads for the OpenAI chat serving benchmark."""

import base64
import io
import math
import wave
from argparse import Namespace
from dataclasses import dataclass
from typing import List

import numpy as np

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow

SAMPLE_RATE = 16000


@dataclass
class AudioDataset(BaseDataset):
    num_requests: int
    audio_duration: float
    audio_count: int
    random_audio_count: bool
    output_len: int
    seed: int
    backend: str

    def __post_init__(self):
        if self.backend != "sglang-oai-chat":
            raise ValueError("The audio dataset requires --backend sglang-oai-chat")
        if self.num_requests <= 0 or self.output_len <= 0:
            raise ValueError("Audio request count and output length must be positive")
        if self.audio_count < 0:
            raise ValueError("--audio-count must be nonnegative")
        if not math.isfinite(self.audio_duration) or self.audio_duration <= 0:
            raise ValueError("--audio-duration must be finite and positive")
        if int(self.audio_duration * SAMPLE_RATE) < 1:
            raise ValueError("--audio-duration must contain at least one audio sample")

    @classmethod
    def from_args(cls, args: Namespace) -> "AudioDataset":
        if getattr(args, "tokenize_prompt", False):
            raise ValueError("--tokenize-prompt is incompatible with the audio dataset")
        return cls(
            num_requests=args.num_prompts,
            audio_duration=args.audio_duration,
            audio_count=args.audio_count,
            random_audio_count=args.random_audio_count,
            output_len=args.random_output_len,
            seed=args.seed,
            backend=args.backend,
        )

    def load(self, tokenizer=None, model_id=None) -> List[DatasetRow]:
        # Local RNGs keep workload generation independent of model imports and
        # the global RNG used later for request arrival times.
        count_seed, sample_seed = np.random.SeedSequence(self.seed).spawn(2)
        count_rng = np.random.default_rng(count_seed)
        sample_rng = np.random.default_rng(sample_seed)
        num_samples = int(self.audio_duration * SAMPLE_RATE)
        requests = []
        for _ in range(self.num_requests):
            count = self.audio_count
            if self.random_audio_count and count > 0:
                count = int(count_rng.integers(1, count + 1))
            content = []
            for _ in range(count):
                # Signed little-endian PCM16 noise, encoded as a complete WAV.
                samples = sample_rng.integers(
                    -32768, 32768, size=num_samples, dtype=np.int16
                ).astype("<i2", copy=False)
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(SAMPLE_RATE)
                    wav.writeframes(samples.tobytes())
                content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
                            "format": "wav",
                        },
                    }
                )
            content.append(
                {
                    "type": "text",
                    "text": "Describe the audio."
                    if count
                    else "Write a short description.",
                }
            )
            requests.append(
                DatasetRow(
                    prompt=[{"role": "user", "content": content}],
                    # Audio token expansion belongs to the server's processor.
                    # Zero is only a placeholder until response usage arrives.
                    prompt_len=0,
                    output_len=self.output_len,
                    prompt_len_from_usage=True,
                    audio_duration=count * num_samples / SAMPLE_RATE,
                )
            )
        return requests
