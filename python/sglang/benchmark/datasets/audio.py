import io
import random
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pybase64
import soundfile as sf
from transformers import AutoProcessor

from sglang.benchmark.datasets.common import (
    BaseDataset,
    DatasetRow,
    compute_random_lens,
    gen_mm_prompt,
)
from sglang.benchmark.utils import get_processor

DEFAULT_AUDIO_SAMPLE_RATE = 48000


@dataclass
class AudioDataset(BaseDataset):
    num_requests: int
    audio_count: int
    input_len: int
    output_len: int
    range_ratio: float
    audio_duration: float
    audio_channels: int
    audio_format: str
    backend: str
    random_audio_count: bool
    seed: int

    @classmethod
    def from_args(cls, args: Namespace) -> "AudioDataset":
        return cls(
            num_requests=args.num_prompts,
            audio_count=args.audio_count,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            range_ratio=args.random_range_ratio,
            audio_duration=args.audio_duration,
            audio_channels=args.audio_channels,
            audio_format=args.audio_format,
            backend=args.backend,
            random_audio_count=args.random_audio_count,
            seed=args.seed,
        )

    def load(self, tokenizer=None, model_id=None) -> List[DatasetRow]:
        processor = get_processor(model_id)
        # Processor initialization may consume global RNG state. Reset it here so
        # --seed fixes the generated prompts and audio content.
        random.seed(self.seed)
        np.random.seed(self.seed)
        return sample_audio_requests(
            num_requests=self.num_requests,
            audio_count=self.audio_count,
            input_len=self.input_len,
            output_len=self.output_len,
            range_ratio=self.range_ratio,
            processor=processor,
            audio_duration=self.audio_duration,
            audio_channels=self.audio_channels,
            audio_format=self.audio_format,
            backend=self.backend,
            random_audio_count=self.random_audio_count,
        )


def generate_synthetic_audio_data_uri(
    duration: float,
    num_channels: int,
    audio_format: str = "wav",
    sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
) -> Tuple[np.ndarray, str, int]:
    """Generate synthetic audio with random samples.

    Returns the raw sample array (for local processor-based token counting),
    the encoded data URI (for the actual request), and its byte length.
    """
    num_samples = int(sample_rate * duration)
    audio_data = np.random.uniform(-0.5, 0.5, (num_samples, num_channels))
    audio_data = np.clip(audio_data, -1.0, 1.0)

    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format=audio_format.upper())
    buffer.seek(0)
    audio_bytes = buffer.read()

    encoded = pybase64.b64encode(audio_bytes).decode("utf-8")
    mime_format = "mpeg" if audio_format.lower() == "mp3" else audio_format.lower()
    audio_url = f"data:audio/{mime_format};base64,{encoded}"
    return audio_data, audio_url, len(audio_url.encode("utf-8"))


def create_mm_data_row(
    text_prompt, audios: list, audios_base64, sample_rate, output_len, processor, backend
):
    content_items = [
        {"type": "audio", "audio": audio_base64} for audio_base64 in audios_base64
    ]
    content_items.append({"type": "text", "text": text_prompt})
    try:
        prompt_str = processor.apply_chat_template(
            [{"role": "user", "content": content_items}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception as e:
        print(f"Error applying chat template: {e}, fallback to <audio> tag")
        prompt_str = f"<audio>{text_prompt}"

    prompt_len = processor(
        text=[prompt_str],
        audios=audios,
        sampling_rate=sample_rate,
        padding=False,
        return_tensors="pt",
    )["input_ids"].numel()

    try:
        text_only_prompt = processor.apply_chat_template(
            [{"role": "user", "content": text_prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        text_prompt_len = processor(
            text=[text_only_prompt],
            padding=False,
            return_tensors="pt",
        )["input_ids"].numel()
    except Exception:
        tokenizer_to_use = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        text_prompt_len = len(tokenizer_to_use.encode(text_prompt))

    vision_prompt_len = prompt_len - text_prompt_len

    supported_backends = ("sglang", "sglang-native", "sglang-oai-chat", "vllm-chat")
    if backend not in supported_backends:
        raise ValueError(
            f"Audio dataset only supports backends: {supported_backends}, "
            f"got '{backend}'."
        )
    use_raw_prompt = backend in ("sglang-oai-chat", "vllm-chat")

    return DatasetRow(
        prompt=text_prompt if use_raw_prompt else prompt_str,
        prompt_len=prompt_len,
        output_len=output_len,
        text_prompt_len=text_prompt_len,
        vision_prompt_len=vision_prompt_len,
        audio_data=audios_base64,
    )


def sample_audio_requests(
    num_requests: int,
    audio_count: int,
    input_len: int,
    output_len: int,
    range_ratio: float,
    processor: AutoProcessor,
    audio_duration: float,
    audio_channels: int,
    audio_format: str,
    backend: str,
    random_audio_count: bool = False,
) -> List[DatasetRow]:
    """Generate requests with synthetic audio clips.

    - If ``random_audio_count`` is True, each request includes a random number of clips between 1 and ``audio_count``.
    - If ``random_audio_count`` is False, each request includes exactly ``audio_count`` clips.
    - Each clip is ``audio_duration`` seconds of random samples at ``audio_channels`` channels.
    """

    if random_audio_count:
        audio_counts = np.random.randint(1, audio_count + 1, size=num_requests)
        total_audios = np.sum(audio_counts)
    else:
        audio_counts = np.full(num_requests, audio_count)
        total_audios = audio_count * num_requests

    input_lens = compute_random_lens(
        full_len=input_len, range_ratio=range_ratio, num=num_requests
    )
    output_lens = compute_random_lens(
        full_len=output_len, range_ratio=range_ratio, num=num_requests
    )

    dataset: List[DatasetRow] = []
    total_audio_bytes = 0
    for i in range(num_requests):
        request_audio_count = int(audio_counts[i])

        text_prompt = gen_mm_prompt(
            processor.tokenizer if hasattr(processor, "tokenizer") else processor,
            processor.audio_token_id if hasattr(processor, "audio_token_id") else None,
            int(input_lens[i]),
        )

        audios, audios_base64, audios_bytes = zip(
            *[
                generate_synthetic_audio_data_uri(
                    audio_duration, audio_channels, audio_format
                )
                for _ in range(request_audio_count)
            ]
        )
        total_audio_bytes += sum(audios_bytes)

        data_row = create_mm_data_row(
            text_prompt,
            list(audios),
            list(audios_base64),
            DEFAULT_AUDIO_SAMPLE_RATE,
            int(output_lens[i]),
            processor,
            backend,
        )
        dataset.append(data_row)

    print(f"#Input tokens: {np.sum([x.prompt_len for x in dataset])}")
    print(f"#Output tokens: {np.sum([x.output_len for x in dataset])}")
    print(f"#Total audio clips: {total_audios}")

    if random_audio_count:
        print(
            f"#Audio clips per request: min={np.min(audio_counts)}, max={np.max(audio_counts)}, mean={np.mean(audio_counts):.2f}"
        )
    else:
        print(f"#Audio clips per request: {audio_count} (fixed)")

    text_prompt_lens = np.array([r.text_prompt_len for r in dataset])
    vision_prompt_lens = np.array([r.vision_prompt_len for r in dataset])
    text_prompt_overheads = text_prompt_lens - input_lens
    stat_fields = [
        ("Raw text prompt tokens (without overhead)", input_lens),
        ("Text prompt tokens (with chat template)", text_prompt_lens),
        ("Text prompt overhead", text_prompt_overheads),
        ("Audio tokens", vision_prompt_lens),
    ]
    print("\n=== Token Breakdown (per request avg / total) ===")
    for label, vals in stat_fields:
        print(f"  {label}: avg={np.mean(vals):.1f}, total={np.sum(vals)}")

    print(
        f"\nCreated {len(dataset)} {audio_format} audio clips with average {total_audio_bytes // num_requests} bytes per request"
    )
    return dataset
