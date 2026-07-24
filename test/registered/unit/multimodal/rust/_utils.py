import io
import json

import msgspec
import numpy as np
from PIL import Image

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(
    est_time=0, suite="base-a-test-cpu", disabled="Rust multimodal test helpers"
)

IMAGE_TOKEN_ID = 900
VISION_START_ID = 901
VISION_END_ID = 902
VIDEO_TOKEN_ID = 903

PROCESSOR_CONFIGS = {
    "qwen2_vl": dict(
        patch_size=14,
        merge_size=2,
        temporal_patch_size=2,
        min_pixels=56 * 56,
        max_pixels=28 * 28 * 1280,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    ),
    "qwen2_5_vl": dict(
        patch_size=14,
        merge_size=2,
        temporal_patch_size=2,
        min_pixels=56 * 56,
        max_pixels=28 * 28 * 1280,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    ),
    "qwen3_5": dict(
        patch_size=16,
        merge_size=2,
        temporal_patch_size=2,
        min_pixels=65536,
        max_pixels=16777216,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    ),
}


def make_image(width, height, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = np.stack(
        (x * 255 / max(width - 1, 1), y * 255 / max(height - 1, 1), (x + y) % 256),
        axis=-1,
    )
    return Image.fromarray(
        np.clip(base + rng.integers(0, 24, base.shape), 0, 255).astype(np.uint8)
    )


def image_bytes(width, height, seed=0):
    buffer = io.BytesIO()
    make_image(width, height, seed).save(buffer, format="PNG")
    return buffer.getvalue()


def spec_json(config, image_token_id=IMAGE_TOKEN_ID):
    return json.dumps({"family": "qwen_vl", "image_token_id": image_token_id, **config})


def request_payload(input_ids, images):
    return msgspec.msgpack.encode([None, input_ids, images, None, None])
