"""
Usage:
python3 -m unittest test_vision_openai_server.TestOpenAIVisionServer.test_mixed_batch
python3 -m unittest test_vision_openai_server.TestOpenAIVisionServer.test_multi_images_chat_completion
"""

import base64
import io
import re
import unittest

import openai
from PIL import Image, ImageDraw

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)
from sglang.test.vlm_utils import (
    IMAGE_MAN_IRONING_URL,
    AudioOpenAITestMixin,
    CustomTestCase,
    ImageOpenAITestMixin,
    TestOpenAIMLLMServerBase,
    VideoOpenAITestMixin,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=560, stage="base-b", runner_config="1-gpu-large")


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


class TestLlavaServer(ImageOpenAITestMixin):
    model = "lmms-lab/llava-onevision-qwen2-0.5b-ov"


class TestLfm2VlServer(ImageOpenAITestMixin):
    model = "LiquidAI/LFM2.5-VL-1.6B"


class TestQwen25VLServer(ImageOpenAITestMixin, VideoOpenAITestMixin):
    model = "Qwen/Qwen2.5-VL-7B-Instruct"
    extra_args = [
        "--cuda-graph-max-bs-decode=4",
    ]


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


class TestQwen2VLContextLengthServer(CustomTestCase):
    # --context-length 300 is calibrated to this model's mm-token expansion:
    # it must sit above the warmup image's expanded length but below the test
    # image's. A cheaper VLM needs the bound recalibrated, not just swapped.
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen2-VL-7B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--context-length",
                "300",
                "--cuda-graph-max-bs-decode",
                "4",
            ],
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        terminate_and_kill_process_tree(cls.process, wait_timeout=60)

    def test_single_image_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                model="default",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": IMAGE_MAN_IRONING_URL},
                            },
                            {
                                "type": "text",
                                "text": "Give a lengthy description of this picture",
                            },
                        ],
                    },
                ],
                temperature=0,
            )

        # context length is checked first, then max_req_input_len, which is calculated from the former
        assert (
            "Multimodal prompt is too long after expanding multimodal tokens."
            in str(cm.exception)
            or "is longer than the model's context length" in str(cm.exception)
        )


# flaky
# class TestMllamaServer(ImageOpenAITestMixin):
#     model = "meta-llama/Llama-3.2-11B-Vision-Instruct"


class TestInternVL25Server(ImageOpenAITestMixin):
    model = "OpenGVLab/InternVL2_5-2B"
    extra_args = [
        "--cuda-graph-max-bs-decode=4",
    ]


@unittest.skip("temporarily disabled: NaN in next_token_logits")
class TestMiniCPMV4Server(ImageOpenAITestMixin):
    model = "openbmb/MiniCPM-V-4"
    extra_args = [
        "--cuda-graph-max-bs-decode=4",
    ]


@unittest.skip("temporarily disabled: NaN in next_token_logits")
class TestMiniCPMo26Server(ImageOpenAITestMixin, AudioOpenAITestMixin):
    model = "openbmb/MiniCPM-o-2_6"
    extra_args = [
        "--cuda-graph-max-bs-decode=4",
    ]


class TestGemma3itServer(ImageOpenAITestMixin):
    model = "google/gemma-3-4b-it"
    extra_args = [
        "--cuda-graph-max-bs-decode=4",
    ]


class TestKimiVLServer(ImageOpenAITestMixin):
    model = "moonshotai/Kimi-VL-A3B-Instruct"
    extra_args = [
        "--context-length=8192",
        "--dtype=bfloat16",
        # Weights alone need ~0.39; 0.40 left <0.001 headroom and flaked at load.
        "--mem-fraction-static=0.42",
    ]

    def test_video_images_chat_completion(self):
        # model context length exceeded
        pass


@unittest.skip(
    "Disabling this test to speed up CI. Prefer to test it within nightly test."
)
class TestGLM41VServer(ImageOpenAITestMixin, VideoOpenAITestMixin):
    model = "zai-org/GLM-4.1V-9B-Thinking"
    extra_args = [
        "--reasoning-parser=glm45",
    ]


class TestQwen2AudioServer(AudioOpenAITestMixin):
    model = "Qwen/Qwen2-Audio-7B-Instruct"


class TestDeepseekOCRServer(TestOpenAIMLLMServerBase):
    model = "deepseek-ai/DeepSeek-OCR"
    trust_remote_code = False
    extra_args = [
        "--mem-fraction-static=0.70",
        "--cuda-graph-max-bs-decode=4",
    ]

    def verify_single_image_response_for_ocr(self, response):
        """Verify DeepSeek-OCR grounding output with coordinates"""
        assert response.choices[0].message.role == "assistant"
        text = response.choices[0].message.content
        assert isinstance(text, str)

        # DeepSeek-OCR uses grounding format, outputs coordinates
        assert "text" in text.lower(), f"OCR text: {text}, should contain 'text'"

        # Verify coordinate format [[x1, y1, x2, y2]]
        import re

        coord_pattern = r"\[\[[\d\s,]+\]\]"
        assert re.search(
            coord_pattern, text
        ), f"OCR text: {text}, should contain coordinate format [[x1, y1, x2, y2]]"

        # Verify basic response fields
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def test_single_image_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        image_url = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/ocr-text.png"

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                        {
                            "type": "text",
                            "text": "<|grounding|>Convert the document to markdown.",
                        },
                    ],
                },
            ],
            temperature=0,
            **(self.get_vision_request_kwargs()),
        )

        self.verify_single_image_response_for_ocr(response)


# Delete the mixin classes so that they are not collected by pytest
del (
    TestOpenAIMLLMServerBase,
    ImageOpenAITestMixin,
    VideoOpenAITestMixin,
    AudioOpenAITestMixin,
)


if __name__ == "__main__":
    unittest.main()
