"""Vision OpenAI server cases that stay on H100.

Qwen3-VL-30B-A3B and Kimi-VL-A3B do not fit a 32 GB card. Split from
test_vision_openai_server_a.py so the rest of that file runs on 1-gpu-small.
"""

import base64
import io
import re
import unittest

import openai
from PIL import Image, ImageDraw

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.vlm_utils import (
    ImageOpenAITestMixin,
    TestOpenAIMLLMServerBase,
    VideoOpenAITestMixin,
)

register_cuda_ci(est_time=200, stage="base-b", runner_config="1-gpu-large")

# --- Qwen3-VL grounding regression (deepstack fusion) --------------------------
# Guards Qwen3MoeLLMModel.forward: deepstack (multi-scale ViT features) injection
# must keep its original inference order. PR #14636 rerouted it through
# post_residual_addition (for RL on-policy / FSDP), which is FP-order-sensitive
# and regresses FP8 visual grounding (the predicted point drifts by ~150+ px).
_GROUNDING_IMG_SIZE = 1000
# Target box in pixels; on a 1000x1000 canvas this equals the 0-1000 normalized
# coordinate, so the check is robust to normalized-vs-pixel conventions.
_GROUNDING_BOX = (620, 180, 880, 360)  # (x0, y0, x1, y1), center (750, 270)
_GROUNDING_MARGIN = 60
_GROUNDING_SYSTEM = (
    "You are a UI grounding model. Treat the image as a 1000x1000 normalized "
    "coordinate system with the top-left at (0,0) and the bottom-right at "
    "(1000,1000). Return the geometric center of the requested element. "
    "Output ONLY one coordinate in the form (x, y) and nothing else."
)
_GROUNDING_COORD_RE = re.compile(r"\(?\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\)?")


def _make_grounding_image() -> str:
    """White canvas with a single red box at _GROUNDING_BOX; base64 data URI."""
    img = Image.new("RGB", (_GROUNDING_IMG_SIZE, _GROUNDING_IMG_SIZE), (255, 255, 255))
    ImageDraw.Draw(img).rectangle(_GROUNDING_BOX, fill=(220, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


class TestQwen3VLServer(ImageOpenAITestMixin, VideoOpenAITestMixin):
    model = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    extra_args = ["--cuda-graph-max-bs-decode=4"]

    @classmethod
    def setUpClass(cls):
        with envs.SGLANG_MM_FEATURE_CACHE_MB.override(512):
            super().setUpClass()

    def test_deepstack_grounding_hits_target_box(self):
        # Regression guard for the Qwen3-VL MoE deepstack fusion order: the
        # predicted point must land inside the target box; a deepstack corruption
        # drifts it out (see PR #14636).
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model="default",
            messages=[
                {"role": "system", "content": _GROUNDING_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _make_grounding_image()},
                        },
                        {
                            "type": "text",
                            "text": "Point at the center of the red rectangle.",
                        },
                    ],
                },
            ],
            temperature=0,
            **(self.get_vision_request_kwargs()),
        )
        out = response.choices[0].message.content
        match = _GROUNDING_COORD_RE.search(out or "")
        self.assertIsNotNone(match, f"could not parse a coordinate from: {out!r}")
        x, y = int(match.group(1)), int(match.group(2))
        x0, y0, x1, y1 = _GROUNDING_BOX
        inside = (x0 - _GROUNDING_MARGIN <= x <= x1 + _GROUNDING_MARGIN) and (
            y0 - _GROUNDING_MARGIN <= y <= y1 + _GROUNDING_MARGIN
        )
        self.assertTrue(
            inside,
            f"grounding output {out!r} -> ({x}, {y}) fell outside target box "
            f"{_GROUNDING_BOX} (margin {_GROUNDING_MARGIN}); deepstack fusion "
            f"likely regressed grounding.",
        )


class TestKimiVLServer(ImageOpenAITestMixin):
    model = "moonshotai/Kimi-VL-A3B-Instruct"
    extra_args = [
        "--context-length=8192",
        "--dtype=bfloat16",
        # Weights alone need ~0.39; 0.40 left <0.001 headroom and flaked at load.
        "--mem-fraction-static=0.42",
    ]

    @unittest.skip("model context length exceeded")
    def test_video_images_chat_completion(self):
        pass


# Delete the mixin classes so that they are not collected by pytest
del (
    TestOpenAIMLLMServerBase,
    ImageOpenAITestMixin,
    VideoOpenAITestMixin,
)


if __name__ == "__main__":
    unittest.main()
