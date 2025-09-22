import io
import warnings
from typing import List, Tuple

import numpy as np

from sglang.benchmark.datasets.common import BaseDatasetLoader, DatasetRow
from sglang.benchmark.utils import gen_prompt


class RandomImageLoader(BaseDatasetLoader):
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
            f"Unsupported random-image resolution: {image_resolution}. "
            "Choose from 4k, 1080p, 720p, 360p, or provide custom 'heightxwidth' (e.g., 1080x1920)."
        )

    def load(self) -> List[DatasetRow]:
        assert (
            not self.args.tokenize_prompt
        ), "random-image does not support --tokenize-prompt"
        try:
            import pybase64
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "Please install Pillow to generate random images: pip install pillow"
            ) from e

        num_requests = self.args.num_prompts
        num_images = self.args.random_image_num_images
        input_len = self.args.random_input_len
        output_len = self.args.random_output_len
        range_ratio = self.args.random_range_ratio
        apply_chat_template = self.args.apply_chat_template
        image_resolution = self.args.random_image_resolution

        # Parse resolution (supports presets and 'heightxwidth')
        width, height = self._parse_resolution(image_resolution)

        # Check for potentially problematic combinations and warn user
        if width * height >= 1920 * 1080 and num_images * num_requests >= 100:
            warnings.warn(
                f"High resolution ({width}x{height}) with {num_images * num_requests} total images "
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

        def _gen_random_image_data_uri(width: int = width, height: int = height) -> str:
            arr = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            encoded = pybase64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"

        dataset: List[DatasetRow] = []
        for i in range(num_requests):
            # Generate text prompt
            text_prompt = gen_prompt(self.tokenizer, int(input_lens[i]))

            # Generate image list
            images = [_gen_random_image_data_uri() for _ in range(num_images)]

            prompt_str = text_prompt
            if apply_chat_template:
                try:
                    content_items = [
                        {"type": "image_url", "image_url": {"url": img_url}}
                        for img_url in images
                    ]
                    content_items.append({"type": "text", "text": text_prompt})
                    prompt_str = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": content_items}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                except Exception:
                    # Some tokenizers do not support list content; fall back to a placeholder in the text
                    prompt_str = f"<image>{text_prompt}"

            prompt_token_ids = self.tokenizer.encode(prompt_str)
            prompt_token_len = len(prompt_token_ids)

            dataset.append(
                DatasetRow(
                    prompt=prompt_str,
                    prompt_len=prompt_token_len,
                    output_len=int(output_lens[i]),
                    image_data=images,
                )
            )

        print(f"#Input tokens: {np.sum([x.prompt_len for x in dataset])}")
        print(f"#Output tokens: {np.sum([x.output_len for x in dataset])}")
        return dataset
