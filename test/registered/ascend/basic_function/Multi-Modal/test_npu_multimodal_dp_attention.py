"""
NPU DP attention + multimodal tests.

  - multimodal + DP-attention + high concurrency correctness
  - multimodal + DP-attention + DP LM Head
"""

import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_5_9B_WEIGHTS_PATH,
    QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ascend.test_npu_multimodal_utils import (
    Color,
    Shape,
    assert_color_and_shape,
    chat,
    create_test_image,
    image_content,
    launch_server,
    text_content,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    CustomTestCase,
)

register_npu_ci(est_time=200, suite="full-2-npu-a3", nightly=True)


def _send_concurrent(base_url, image_b64, prompt, num_requests=50, max_tokens=32):
    client = openai.Client(api_key="sk-123456", base_url=f"{base_url}/v1")

    def _send():
        try:
            resp = client.chat.completions.create(
                model="default",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            image_content(image_b64),
                            text_content(prompt),
                        ],
                    }
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            return ("ok", resp.choices[0].message.content, resp.usage.completion_tokens)
        except Exception as e:
            return ("error", str(e), 0)

    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(_send) for _ in range(num_requests)]
        results = [f.result() for f in as_completed(futures)]
    return results


class TestMultimodalDPAttention(CustomTestCase):
    """Verify DP-attention + image handles high concurrency correctly.

    [Test Category] multimodal
    [Test Target] multimodal + DP-attention (2-NPU)
    """

    _model = QWEN3_5_9B_WEIGHTS_PATH
    _num_concurrent = 50

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.GREEN, shape=Shape.RECTANGLE
        )
        cls._prompt = (
            "What is the main color and shape in the image? "
            "Answer directly with just the color and shape, "
            "no reasoning or explanation."
        )
        cls._process, cls._url = launch_server(
            cls._model,
            extra_args=[
                "--enable-dp-attention",
                "--dp-size",
                "2",
                "--tp-size",
                "2",
                "--mem-fraction-static",
                "0.6",
                "--mamba-radix-cache-strategy",
                "extra_buffer",
                "--dtype",
                "bfloat16",
                "--mamba-ssm-dtype",
                "bfloat16",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls._process.pid)

    def test_dp_attention_concurrency(self):
        """Send 50 concurrent requests with DP-attention, verify all succeed."""
        results = _send_concurrent(
            self._url,
            self._image_b64,
            self._prompt,
            num_requests=self._num_concurrent,
            max_tokens=32,
        )

        ok_count = sum(1 for r in results if r[0] == "ok")
        self.assertEqual(
            ok_count,
            self._num_concurrent,
            f"Only {ok_count}/{self._num_concurrent} requests succeeded",
        )

        for idx, (status, content, _tokens) in enumerate(results):
            if status == "ok":
                self.assertTrue(
                    content and len(content) > 0,
                    f"Request {idx} returned empty content",
                )
                assert_color_and_shape(
                    self, content, "green", "rectangle", prefix=f"Request {idx}: "
                )


class TestMultimodalDpLmHead(CustomTestCase):
    """Verify DP LM head sharding does not affect image token projection.

    [Test Category] multimodal
    [Test Target] multimodal + DP-attention + DP LM Head (2-NPU)
    """

    _model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.BLUE, shape=Shape.RECTANGLE
        )
        cls._messages = [
            {
                "role": "user",
                "content": [
                    image_content(cls._image_b64),
                    text_content("Describe the image"),
                ],
            },
        ]
        cls._process, cls._url = launch_server(
            cls._model,
            extra_args=[
                "--mem-fraction-static",
                "0.4",
                "--tp-size",
                "2",
                "--dp-size",
                "2",
                "--enable-dp-attention",
                "--enable-dp-lm-head",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls._process.pid)

    def test_dp_lm_head_image(self):
        """Send image request with DP LM Head, verify output and feature gating."""
        output = chat(self._url, self._messages, max_tokens=128, seed=42)

        self.assertIsNotNone(output, "DP LM Head returned None")
        self.assertGreater(len(output), 0, "DP LM Head output is empty")

        assert_color_and_shape(
            self,
            output,
            "blue",
            "rectangle",
            prefix="test_dp_lm_head_image: ",
        )


if __name__ == "__main__":
    unittest.main()
