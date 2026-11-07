import random
import warnings
from argparse import Namespace
from dataclasses import dataclass
from tempfile import TemporaryDirectory
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
from sglang.benchmark.datasets.image import (
    parse_image_resolution,
    parse_random_image_resolution,
)
from sglang.benchmark.utils import get_processor

DEFAULT_VIDEO_ENCODE_FPS = 30.0


@dataclass
class VideoDataset(BaseDataset):
    num_requests: int
    video_count: int
    input_len: int
    output_len: int
    range_ratio: float
    video_frames: int
    video_resolution: str
    video_format: str
    backend: str
    random_video_count: bool
    seed: int

    @classmethod
    def from_args(cls, args: Namespace) -> "VideoDataset":
        return cls(
            num_requests=args.num_prompts,
            video_count=args.video_count,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            range_ratio=args.random_range_ratio,
            video_frames=args.video_frames,
            video_resolution=args.video_resolution,
            video_format=args.video_format,
            backend=args.backend,
            random_video_count=args.random_video_count,
            seed=args.seed,
        )

    def load(self, tokenizer=None, model_id=None) -> List[DatasetRow]:
        processor = get_processor(model_id)
        # Processor initialization may consume global RNG state. Reset it here so
        # --seed fixes the generated prompts, video sizes, and frame counts.
        random.seed(self.seed)
        np.random.seed(self.seed)
        return sample_video_requests(
            num_requests=self.num_requests,
            video_count=self.video_count,
            input_len=self.input_len,
            output_len=self.output_len,
            range_ratio=self.range_ratio,
            processor=processor,
            video_frames=self.video_frames,
            video_resolution=self.video_resolution,
            video_format=self.video_format,
            backend=self.backend,
            random_video_count=self.random_video_count,
        )


def generate_synthetic_video_data_uri(
    width: int,
    height: int,
    num_frames: int,
    video_format: str = "mp4",
    encode_fps: float = DEFAULT_VIDEO_ENCODE_FPS,
) -> Tuple[np.ndarray, str, int]:
    """Generate a synthetic video with random frames.

    Returns the raw frame array (for local processor-based token counting),
    the encoded data URI (for the actual request), and its byte length.

    `encode_fps` is container metadata only; a target model may resample at
    its own native fps, so the frame count it actually processes can differ.
    """
    import cv2

    random_pixels = np.random.randint(
        0, 256, (num_frames, height, width, 3), dtype=np.uint8
    )

    with TemporaryDirectory() as temp_dir:
        temp_path = f"{temp_dir}/video.{video_format}"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            temp_path, fourcc=fourcc, fps=encode_fps, frameSize=(width, height)
        )
        try:
            if not video_writer.isOpened():
                raise RuntimeError("Failed to create video writer")
            for frame in random_pixels:
                video_writer.write(frame)
        finally:
            video_writer.release()

        with open(temp_path, "rb") as f:
            video_bytes = f.read()

    encoded = pybase64.b64encode(video_bytes).decode("utf-8")
    video_data = f"data:video/{video_format};base64,{encoded}"
    return random_pixels, video_data, len(video_data.encode("utf-8"))


def create_mm_data_row(
    text_prompt, videos: list, videos_base64, output_len, processor, backend
):
    content_items = [
        {"type": "video", "video": {"url": video_base64}}
        for video_base64 in videos_base64
    ]
    content_items.append({"type": "text", "text": text_prompt})
    try:
        prompt_str = processor.apply_chat_template(
            [{"role": "user", "content": content_items}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception as e:
        print(f"Error applying chat template: {e}, fallback to <video> tag")
        prompt_str = f"<video>{text_prompt}"

    prompt_len = processor(
        text=[prompt_str],
        videos=videos,
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
            f"Video dataset only supports backends: {supported_backends}, "
            f"got '{backend}'."
        )
    use_raw_prompt = backend in ("sglang-oai-chat", "vllm-chat")

    return DatasetRow(
        prompt=text_prompt if use_raw_prompt else prompt_str,
        prompt_len=prompt_len,
        output_len=output_len,
        text_prompt_len=text_prompt_len,
        vision_prompt_len=vision_prompt_len,
        video_data=videos_base64,
    )


def sample_video_requests(
    num_requests: int,
    video_count: int,
    input_len: int,
    output_len: int,
    range_ratio: float,
    processor: AutoProcessor,
    video_frames: int,
    video_resolution: str,
    video_format: str,
    backend: str,
    random_video_count: bool = False,
) -> List[DatasetRow]:
    """Generate requests with synthetic videos.

    - If ``random_video_count`` is True, each request includes a random number of videos between 1 and ``video_count``.
    - If ``random_video_count`` is False, each request includes exactly ``video_count`` videos.
    - Resolution supports the same presets/custom/random bounds as the image dataset.
    - ``video_frames`` is a raw frame count encoded into the synthetic container;
      a target model's own video preprocessing determines how many of those
      frames it actually samples.
    """

    random_resolution_bounds = parse_random_image_resolution(video_resolution)
    if random_resolution_bounds is None:
        width, height = parse_image_resolution(video_resolution)
        min_width = max_width = width
        min_height = max_height = height
    else:
        (min_width, min_height), (max_width, max_height) = random_resolution_bounds

    if random_video_count:
        video_counts = np.random.randint(1, video_count + 1, size=num_requests)
        total_videos = np.sum(video_counts)
    else:
        video_counts = np.full(num_requests, video_count)
        total_videos = video_count * num_requests

    if max_width * max_height >= 1920 * 1080 and total_videos >= 20:
        warnings.warn(
            f"High resolution (up to {max_width}x{max_height}) with {total_videos} total videos "
            f"may take a long time. Consider reducing resolution, frame count, or video count.",
            UserWarning,
            stacklevel=2,
        )

    input_lens = compute_random_lens(
        full_len=input_len, range_ratio=range_ratio, num=num_requests
    )
    output_lens = compute_random_lens(
        full_len=output_len, range_ratio=range_ratio, num=num_requests
    )

    def _gen_random_video() -> Tuple[np.ndarray, str, int]:
        if random_resolution_bounds is None:
            width, height = min_width, min_height
        else:
            width = np.random.randint(min_width, max_width + 1)
            height = np.random.randint(min_height, max_height + 1)
        return generate_synthetic_video_data_uri(
            width, height, video_frames, video_format
        )

    dataset: List[DatasetRow] = []
    total_video_bytes = 0
    for i in range(num_requests):
        request_video_count = int(video_counts[i])

        text_prompt = gen_mm_prompt(
            processor.tokenizer if hasattr(processor, "tokenizer") else processor,
            processor.video_token_id if hasattr(processor, "video_token_id") else None,
            int(input_lens[i]),
        )

        videos, videos_base64, videos_bytes = zip(
            *[_gen_random_video() for _ in range(request_video_count)]
        )
        total_video_bytes += sum(videos_bytes)

        data_row = create_mm_data_row(
            text_prompt,
            list(videos),
            list(videos_base64),
            int(output_lens[i]),
            processor,
            backend,
        )
        dataset.append(data_row)

    print(f"#Input tokens: {np.sum([x.prompt_len for x in dataset])}")
    print(f"#Output tokens: {np.sum([x.output_len for x in dataset])}")
    print(f"#Total videos: {total_videos}")

    if random_video_count:
        print(
            f"#Videos per request: min={np.min(video_counts)}, max={np.max(video_counts)}, mean={np.mean(video_counts):.2f}"
        )
    else:
        print(f"#Videos per request: {video_count} (fixed)")

    text_prompt_lens = np.array([r.text_prompt_len for r in dataset])
    vision_prompt_lens = np.array([r.vision_prompt_len for r in dataset])
    text_prompt_overheads = text_prompt_lens - input_lens
    stat_fields = [
        ("Raw text prompt tokens (without overhead)", input_lens),
        ("Text prompt tokens (with chat template)", text_prompt_lens),
        ("Text prompt overhead", text_prompt_overheads),
        ("Vision tokens", vision_prompt_lens),
    ]
    print("\n=== Token Breakdown (per request avg / total) ===")
    for label, vals in stat_fields:
        print(f"  {label}: avg={np.mean(vals):.1f}, total={np.sum(vals)}")

    print(
        f"\nCreated {len(dataset)} {video_format} videos with average {total_video_bytes // num_requests} bytes per request"
    )
    return dataset
