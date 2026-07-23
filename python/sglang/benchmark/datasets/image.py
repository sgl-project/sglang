import io
import warnings
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pybase64
from PIL import Image
from transformers import AutoProcessor

from sglang.benchmark.datasets.common import (
    BaseDataset,
    DatasetRow,
    compute_random_lens,
    gen_mm_prompt,
)
from sglang.benchmark.utils import get_processor


@dataclass
class ImageDataset(BaseDataset):
    num_requests: int
    image_count: int
    input_len: int
    output_len: int
    range_ratio: float
    image_content: str
    image_format: str
    image_resolution: str
    backend: str
    random_image_count: bool

    @classmethod
    def from_args(cls, args: Namespace) -> "ImageDataset":
        return cls(
            num_requests=args.num_prompts,
            image_count=args.image_count,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            range_ratio=args.random_range_ratio,
            image_content=args.image_content,
            image_format=args.image_format,
            image_resolution=args.image_resolution,
            backend=args.backend,
            random_image_count=args.random_image_count,
        )

    def load(self, tokenizer=None, model_id=None) -> List[DatasetRow]:
        processor = get_processor(model_id)
        return sample_image_requests(
            num_requests=self.num_requests,
            image_count=self.image_count,
            input_len=self.input_len,
            output_len=self.output_len,
            range_ratio=self.range_ratio,
            processor=processor,
            image_content=self.image_content,
            image_format=self.image_format,
            image_resolution=self.image_resolution,
            backend=self.backend,
            random_image_count=self.random_image_count,
        )


def parse_image_resolution(image_resolution: str) -> Tuple[int, int]:
    """Parse image resolution into (width, height).

    Supports presets '1080p', '720p', '360p' and custom 'heightxwidth' format
    (e.g., '1080x1920' means height=1080, width=1920).
    """
    resolution_to_size = {
        "4k": (3840, 2160),
        "1080p": (1920, 1080),
        "720p": (1280, 720),
        "360p": (640, 360),
    }
    if image_resolution in resolution_to_size:
        return resolution_to_size[image_resolution]

    res = image_resolution.strip().lower()
    if "x" in res:
        parts = res.split("x")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            height = int(parts[0])
            width = int(parts[1])
            if height > 0 and width > 0:
                return (width, height)

    raise ValueError(
        f"Unsupported image resolution: {image_resolution}. "
        "Choose from 4k, 1080p, 720p, 360p, or provide custom 'heightxwidth' (e.g., 1080x1920)."
    )


def parse_random_image_resolution(
    image_resolution: str,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Parse ``random:<min_h>x<min_w>-<max_h>x<max_w>`` image bounds.

    Returns ``None`` for fixed resolutions. The returned dimensions are
    ``(width, height)`` pairs, matching :func:`parse_image_resolution`.
    """

    prefix = "random:"
    if not image_resolution.strip().lower().startswith(prefix):
        return None

    bounds = image_resolution.strip()[len(prefix) :].split("-", maxsplit=1)
    if len(bounds) != 2:
        raise ValueError(
            "Random image resolution must be 'random:<min_h>x<min_w>-"
            "<max_h>x<max_w>', for example 'random:256x256-1024x1024'."
        )

    min_width, min_height = parse_image_resolution(bounds[0])
    max_width, max_height = parse_image_resolution(bounds[1])
    if min_width > max_width or min_height > max_height:
        raise ValueError("Random image resolution minimum cannot exceed maximum.")
    return (min_width, min_height), (max_width, max_height)


def create_mm_data_row(
    text_prompt, images: list, images_base64, output_len, processor, backend
):
    try:
        if type(processor).__name__ == "Phi4MMProcessor":
            # <|endoftext10|> is the image token used in the phi-4-multimodal model.
            content_items = text_prompt.replace("image 1", "|endoftext10|")
        else:
            content_items = [
                {"type": "image", "image": {"url": image_base64}}
                for image_base64 in images_base64
            ]
            content_items.append({"type": "text", "text": text_prompt})
        prompt_str = processor.apply_chat_template(
            [{"role": "user", "content": content_items}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception as e:
        # Note (Xinyuan): This is a workaround for an issue where some tokenizers do not support content as a list. (e.g. InternVL)
        print(f"Error applying chat template: {e}, fallback to <image> tag")
        # Some tokenizers do not support list content; fall back to a placeholder in the text
        if type(processor).__name__ == "MiniCPMOProcessor":
            prompt_str = f"(<image>./</image>){text_prompt}"
        else:
            prompt_str = f"<image>{text_prompt}"

    # Calculate total tokens (text + vision)
    if type(processor).__name__ == "KimiK25Processor":
        medias = [{"type": "image", "image": img} for img in images]
        prompt_len = processor(
            text=prompt_str,
            medias=medias,
            return_tensors="pt",
        )["input_ids"].numel()
    elif type(processor).__name__ == "VLChatProcessor":
        prompt_len = processor(
            prompt=prompt_str,
            images=images,
            force_batchify=False,
        )["input_ids"].numel()
    elif type(processor).__name__ == "DeepseekVLV2Processor":
        result = processor(
            conversations=prompt_str,
            images=images,
            inference_mode=True,
        )
        prompt_len = result.input_ids.numel()
    else:
        prompt_len = processor(
            text=[prompt_str],
            images=images,
            padding=False,
            return_tensors="pt",
        )["input_ids"].numel()

    # Calculate text-only tokens
    try:
        # Create text-only version of the prompt
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
        # Fallback: just tokenize the text prompt directly
        tokenizer_to_use = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        text_prompt_len = len(tokenizer_to_use.encode(text_prompt))

    # Vision tokens = total tokens - text tokens
    vision_prompt_len = prompt_len - text_prompt_len

    supported_backends = [
        "sglang",
        "sglang-native",
        "sglang-oai-chat",
        "vllm-chat",
    ]
    if backend not in supported_backends:
        raise ValueError(
            f"Image dataset only supports backends: {supported_backends}, "
            f"got '{backend}'."
        )

    # OpenAI chat handlers apply the chat template and receive images separately, so
    # send the raw text. /generate does not apply a chat template, so it needs
    # prompt_str, which contains the multimodal processor's image placeholders.
    use_raw_prompt = backend in ("sglang-oai-chat", "vllm-chat")

    return DatasetRow(
        prompt=text_prompt if use_raw_prompt else prompt_str,
        prompt_len=prompt_len,
        output_len=output_len,
        text_prompt_len=text_prompt_len,
        vision_prompt_len=vision_prompt_len,
        image_data=images_base64,
    )


def sample_image_requests(
    num_requests: int,
    image_count: int,
    input_len: int,
    output_len: int,
    range_ratio: float,
    processor: AutoProcessor,
    image_content: str,
    image_format: str,
    image_resolution: str,
    backend: str,
    random_image_count: bool = False,
) -> List[DatasetRow]:
    """Generate requests with images.

    - If ``random_image_count`` is True, each request includes a random number of images between 1 and ``image_count``.
    - If ``random_image_count`` is False, each request includes exactly ``image_count`` images.
    - Supported resolutions: 4k (3840x2160), 1080p (1920x1080), 720p
      (1280x720), 360p (640x360), custom ``heightxwidth`` (e.g.,
      1080x1920), or ``random:<min_h>x<min_w>-<max_h>x<max_w>``.
    - Text lengths follow the 'random' dataset sampling rule. ``prompt_len``
      only counts text tokens and excludes image data.
    """

    random_resolution_bounds = parse_random_image_resolution(image_resolution)
    if random_resolution_bounds is None:
        width, height = parse_image_resolution(image_resolution)
        min_width = max_width = width
        min_height = max_height = height
    else:
        (min_width, min_height), (max_width, max_height) = random_resolution_bounds

    # Determine image counts for each request
    if random_image_count:
        # Random number of images per request
        image_counts = np.random.randint(1, image_count + 1, size=num_requests)
        total_images = np.sum(image_counts)
    else:
        # Fixed number of images per request
        image_counts = np.full(num_requests, image_count)
        total_images = image_count * num_requests

    # Check for potentially problematic combinations and warn user
    if max_width * max_height >= 1920 * 1080 and total_images >= 100:
        warnings.warn(
            f"High resolution (up to {max_width}x{max_height}) with {total_images} total images "
            f"may take a long time. Consider reducing resolution or image count.",
            UserWarning,
            stacklevel=2,
        )

    # Sample text lengths
    input_lens = compute_random_lens(
        full_len=input_len,
        range_ratio=range_ratio,
        num=num_requests,
    )
    output_lens = compute_random_lens(
        full_len=output_len,
        range_ratio=range_ratio,
        num=num_requests,
    )

    def _gen_random_image_data_uri() -> Tuple[Image.Image, str, int, Tuple[int, int]]:
        if random_resolution_bounds is None:
            width, height = min_width, min_height
        else:
            width = np.random.randint(min_width, max_width + 1)
            height = np.random.randint(min_height, max_height + 1)
        if image_content == "blank":
            # Generate blank white image
            arr = np.full((height, width, 3), 255, dtype=np.uint8)
        else:
            # Generate random colored image
            arr = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format=image_format, quality=85)
        encoded = pybase64.b64encode(buf.getvalue()).decode("utf-8")
        image_data = f"data:image/{image_format};base64,{encoded}"
        image_bytes = len(image_data.encode("utf-8"))
        return img, image_data, image_bytes, (width, height)

    dataset: List[DatasetRow] = []
    total_image_bytes = 0
    all_image_sizes: list[Tuple[int, int]] = []
    for i in range(num_requests):
        # Get the number of images for this request
        request_image_count = int(image_counts[i])

        # Generate text prompt
        text_prompt = gen_mm_prompt(
            processor.tokenizer if hasattr(processor, "tokenizer") else processor,
            processor.image_token_id if hasattr(processor, "image_token_id") else None,
            int(input_lens[i]),
        )

        # Generate image list
        images, images_base64, images_bytes, image_sizes = zip(
            *[_gen_random_image_data_uri() for _ in range(request_image_count)]
        )
        total_image_bytes += sum(images_bytes)
        all_image_sizes.extend(image_sizes)

        data_row = create_mm_data_row(
            text_prompt,
            list(images),
            list(images_base64),
            int(output_lens[i]),
            processor,
            backend,
        )
        dataset.append(data_row)

    # Print statistics
    print(f"#Input tokens: {np.sum([x.prompt_len for x in dataset])}")
    print(f"#Output tokens: {np.sum([x.output_len for x in dataset])}")
    print(f"#Total images: {total_images}")

    if random_image_count:
        print(
            f"#Images per request: min={np.min(image_counts)}, max={np.max(image_counts)}, mean={np.mean(image_counts):.2f}"
        )
    else:
        print(f"#Images per request: {image_count} (fixed)")

    if random_resolution_bounds is not None:
        widths, heights = zip(*all_image_sizes)
        print(
            "#Image resolution: "
            f"min={min(widths)}x{min(heights)}, "
            f"max={max(widths)}x{max(heights)}, "
            f"mean={np.mean(widths):.1f}x{np.mean(heights):.1f}"
        )

    # Detailed token breakdown (derived from dataset + input_lens)
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
        f"\nCreated {len(dataset)} {image_content} {image_format} images with average {total_image_bytes // num_requests} bytes per request"
    )
    return dataset
