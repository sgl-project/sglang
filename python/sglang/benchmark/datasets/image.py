import io
import warnings
from typing import List, Tuple

import numpy as np

from sglang.benchmark.datasets.common import (
    BaseDatasetLoader,
    DatasetRow,
    create_mm_data_row,
)
from sglang.benchmark.utils import gen_prompt, get_processor


class ImageLoader(BaseDatasetLoader):
    def _parse_resolution(self, image_resolution: str) -> Tuple[int, int]:
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

    def load(self) -> List[DatasetRow]:
        try:
            import pybase64
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "Please install Pillow to generate random images: pip install pillow"
            ) from e

        num_requests = self.args.num_prompts
        image_count = (self.args.image_count,)
        input_len = self.args.random_input_len
        output_len = self.args.random_output_len
        range_ratio = self.args.random_range_ratio
        processor = get_processor(self.args.model)
        image_content = (self.args.image_content,)
        image_format = (self.args.image_format,)
        image_resolution = (self.args.image_resolution,)

        # Parse resolution (supports presets and 'heightxwidth')
        width, height = self._parse_resolution(image_resolution)

        # Check for potentially problematic combinations and warn user
        if width * height >= 1920 * 1080 and image_count * num_requests >= 100:
            warnings.warn(
                f"High resolution ({width}x{height}) with {image_count * num_requests} total images "
                f"may take a long time. Consider reducing resolution or image count.",
                UserWarning,
                stacklevel=2,
            )

        # Sample text lengths
        input_lens = np.random.randint(
            max(int(input_len * range_ratio), 1), input_len + 1, size=num_requests
        )
        output_lens = np.random.randint(
            int(output_len * range_ratio), output_len + 1, size=num_requests
        )

        def _gen_random_image_data_uri(
            width: int = width, height: int = height
        ) -> Tuple[Image, str, int]:
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
            return img, image_data, image_bytes

        dataset: List[DatasetRow] = []
        total_image_bytes = 0
        for i in range(num_requests):
            # Generate text prompt
            text_prompt = gen_prompt(processor.tokenizer, int(input_lens[i]))

            # Generate image list
            images, images_base64, images_bytes = zip(
                *[_gen_random_image_data_uri() for _ in range(image_count)]
            )
            total_image_bytes += sum(list(images_bytes))

            data_row = create_mm_data_row(
                text_prompt,
                list(images),
                list(images_base64),
                int(output_lens[i]),
                processor,
            )

            dataset.append(data_row)

        print(f"#Input tokens: {np.sum([x.prompt_len for x in dataset])}")
        print(f"#Output tokens: {np.sum([x.output_len for x in dataset])}")
        print(
            f"\nCreated {len(dataset)} {image_content} {image_format} images with average {total_image_bytes//num_requests} bytes per request"
        )
        return dataset
