import io
import wave
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pybase64
from transformers import AutoProcessor

from sglang.benchmark.datasets.common import (
    BaseDataset,
    DatasetRow,
    compute_random_lens,
    gen_mm_prompt,
)
from sglang.benchmark.utils import get_processor

# Backends that understand multimodal audio payloads.
# - sglang / sglang-native: /generate accepts a native ``audio_data`` field.
# - sglang-oai-chat: the OpenAI chat handler accepts ``input_audio`` content.
SUPPORTED_AUDIO_BACKENDS = ["sglang", "sglang-native", "sglang-oai-chat"]


@dataclass
class AudioDataset(BaseDataset):
    num_requests: int
    audio_count: int
    input_len: int
    output_len: int
    range_ratio: float
    audio_content: str
    audio_length: float
    sample_rate: int
    backend: str
    random_audio_count: bool

    @classmethod
    def from_args(cls, args: Namespace) -> "AudioDataset":
        return cls(
            num_requests=args.num_prompts,
            audio_count=args.audio_count,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            range_ratio=args.random_range_ratio,
            audio_content=args.audio_content,
            audio_length=args.audio_length,
            sample_rate=args.audio_sample_rate,
            backend=args.backend,
            random_audio_count=args.random_audio_count,
        )

    def load(self, tokenizer=None, model_id=None) -> List[DatasetRow]:
        processor = get_processor(model_id)
        return sample_audio_requests(
            num_requests=self.num_requests,
            audio_count=self.audio_count,
            input_len=self.input_len,
            output_len=self.output_len,
            range_ratio=self.range_ratio,
            processor=processor,
            audio_content=self.audio_content,
            audio_length=self.audio_length,
            sample_rate=self.sample_rate,
            backend=self.backend,
            random_audio_count=self.random_audio_count,
        )


def gen_audio_data_uri(
    audio_content: str, audio_length: float, sample_rate: int
) -> Tuple[np.ndarray, str, int]:
    """Generate a single synthetic mono WAV clip and return it as a data URI.

    Returns ``(samples, data_uri, num_bytes)`` where ``samples`` is the raw
    float32 waveform in ``[-1, 1]`` (handy for processor-side token counting)
    and ``data_uri`` is a ``data:audio/wav;base64,...`` string accepted by the
    SGLang multimodal audio loader.
    """
    num_samples = max(int(audio_length * sample_rate), 1)
    if audio_content == "silence":
        samples = np.zeros(num_samples, dtype=np.float32)
    else:
        # Random sine tone mixed with light noise so each clip differs.
        t = np.arange(num_samples, dtype=np.float32) / sample_rate
        freq = np.random.uniform(200.0, 2000.0)
        tone = np.sin(2.0 * np.pi * freq * t)
        noise = np.random.uniform(-0.1, 0.1, size=num_samples).astype(np.float32)
        samples = (0.3 * tone + noise).astype(np.float32)

    pcm16 = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())

    encoded = pybase64.b64encode(buf.getvalue()).decode("utf-8")
    data_uri = f"data:audio/wav;base64,{encoded}"
    return samples, data_uri, len(data_uri.encode("utf-8"))


def count_audio_prompt_tokens(
    text_prompt: str,
    audios: List[np.ndarray],
    processor: AutoProcessor,
    sample_rate: int,
) -> Tuple[str, int, int]:
    """Best-effort token accounting for an audio + text prompt.

    Returns ``(prompt_str, prompt_len, text_prompt_len)``. Audio processors
    expose inconsistent signatures, so every step degrades gracefully and falls
    back to a text-only count rather than raising.
    """
    try:
        content_items = [{"type": "audio", "audio": audio} for audio in audios]
        content_items.append({"type": "text", "text": text_prompt})
        prompt_str = processor.apply_chat_template(
            [{"role": "user", "content": content_items}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception as e:
        print(f"Error applying chat template: {e}, fallback to <audio> tag")
        prompt_str = f"<audio>{text_prompt}"

    prompt_len = None
    for audio_kw in ("audio", "audios"):
        try:
            prompt_len = processor(
                text=[prompt_str],
                **{audio_kw: audios},
                sampling_rate=sample_rate,
                padding=False,
                return_tensors="pt",
            )["input_ids"].numel()
            break
        except Exception:
            continue

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if prompt_len is None:
        prompt_len = len(tokenizer.encode(prompt_str))

    try:
        text_only_prompt = processor.apply_chat_template(
            [{"role": "user", "content": text_prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        text_prompt_len = len(tokenizer.encode(text_only_prompt))
    except Exception:
        text_prompt_len = len(tokenizer.encode(text_prompt))

    return prompt_str, prompt_len, text_prompt_len


def create_audio_data_row(
    text_prompt: str,
    audios: List[np.ndarray],
    audios_base64: List[str],
    output_len: int,
    processor: AutoProcessor,
    sample_rate: int,
    backend: str,
) -> DatasetRow:
    if backend not in SUPPORTED_AUDIO_BACKENDS:
        raise ValueError(
            f"Audio dataset only supports backends: {SUPPORTED_AUDIO_BACKENDS}, "
            f"got '{backend}'."
        )

    prompt_str, prompt_len, text_prompt_len = count_audio_prompt_tokens(
        text_prompt, audios, processor, sample_rate
    )
    # Audio tokens = total tokens - text tokens (non-negative).
    audio_prompt_len = max(prompt_len - text_prompt_len, 0)

    # sglang-oai-chat: the server applies the chat template, so send raw text.
    # sglang/sglang-native: /generate does not apply a chat template, so send
    #         prompt_str which carries the audio placeholder tokens.
    use_raw_prompt = backend == "sglang-oai-chat"

    return DatasetRow(
        prompt=text_prompt if use_raw_prompt else prompt_str,
        prompt_len=prompt_len,
        output_len=output_len,
        text_prompt_len=text_prompt_len,
        vision_prompt_len=audio_prompt_len,
        audio_data=audios_base64,
    )


def sample_audio_requests(
    num_requests: int,
    audio_count: int,
    input_len: int,
    output_len: int,
    range_ratio: float,
    processor: AutoProcessor,
    audio_content: str,
    audio_length: float,
    sample_rate: int,
    backend: str,
    random_audio_count: bool = False,
) -> List[DatasetRow]:
    """Generate benchmark requests with synthetic audio input.

    - If ``random_audio_count`` is True, each request includes a random number
      of clips between 1 and ``audio_count``; otherwise exactly ``audio_count``.
    - Clips are mono 16-bit PCM WAV of ``audio_length`` seconds at
      ``sample_rate`` Hz, encoded as ``data:audio/wav;base64,...`` URIs.
    - Text lengths follow the 'random' dataset sampling rule. ``prompt_len``
      counts text + audio tokens; ``text_prompt_len`` counts text only.
    """
    if audio_count <= 0:
        # Degenerate config: no audio per request (graceful text-only fallback).
        audio_counts = np.zeros(num_requests, dtype=int)
    elif random_audio_count:
        audio_counts = np.random.randint(1, audio_count + 1, size=num_requests)
    else:
        audio_counts = np.full(num_requests, audio_count)
    total_audios = int(np.sum(audio_counts))

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
            None,
            int(input_lens[i]),
        )

        if request_audio_count > 0:
            audios, audios_base64, audios_bytes = zip(
                *[
                    gen_audio_data_uri(audio_content, audio_length, sample_rate)
                    for _ in range(request_audio_count)
                ]
            )
            audios, audios_base64 = list(audios), list(audios_base64)
            total_audio_bytes += sum(audios_bytes)
        else:
            audios, audios_base64 = [], []

        dataset.append(
            create_audio_data_row(
                text_prompt,
                audios,
                audios_base64,
                int(output_lens[i]),
                processor,
                sample_rate,
                backend,
            )
        )

    print(f"#Input tokens: {np.sum([x.prompt_len for x in dataset])}")
    print(f"#Output tokens: {np.sum([x.output_len for x in dataset])}")
    print(f"#Total audio clips: {total_audios}")
    if random_audio_count:
        print(
            f"#Clips per request: min={np.min(audio_counts)}, "
            f"max={np.max(audio_counts)}, mean={np.mean(audio_counts):.2f}"
        )
    else:
        print(f"#Clips per request: {audio_count} (fixed)")
    print(
        f"Created {len(dataset)} {audio_content} {audio_length}s WAV clips "
        f"with average {total_audio_bytes // num_requests} bytes per request"
    )
    return dataset
