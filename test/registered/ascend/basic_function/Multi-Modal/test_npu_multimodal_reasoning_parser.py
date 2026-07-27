"""
NPU multimodal + reasoning parser tests.

Verify the reasoning parser with image input, testing both
``separate_reasoning=True`` and ``separate_reasoning=False`` modes.
"""

import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_5_35B_A3B_WEIGHTS_PATH,
)
from sglang.test.ascend.test_npu_multimodal_utils import (
    Color,
    Shape,
    assert_color_and_shape,
    create_test_image,
    image_content,
    launch_server,
    text_content,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    CustomTestCase,
)

register_npu_ci(est_time=90, suite="full-2-npu-a3", nightly=True)


_REASONING_PARSER_ARGS = ["--reasoning-parser", "qwen3"]

_EXTRA_SERVER_ARGS = [
    "--tp-size",
    "2",
    "--mem-fraction-static",
    "0.7",
]


class TestMultimodalReasoningParser(CustomTestCase):
    """Verify reasoning parser with image input.

    [Test Category] multimodal
    [Test Target] multimodal + reasoning parser

    +--------------------+------------------------------------------------+
    | separate_reasoning | Expected behaviour                             |
    +--------------------+------------------------------------------------+
    | True               | reasoning stripped from content, placed into   |
    |                    | reasoning_content                              |
    +--------------------+------------------------------------------------+
    | False              | reasoning text merged into message.content,    |
    |                    | reasoning_content is None                      |
    +--------------------+------------------------------------------------+

    """

    _model = QWEN3_5_35B_A3B_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        cls._process, cls._url = launch_server(
            cls._model,
            extra_args=_REASONING_PARSER_ARGS + _EXTRA_SERVER_ARGS,
        )
        _, cls._image_b64 = create_test_image(
            width=256,
            height=256,
            color=Color.GREEN,
            shape=Shape.ELLIPSE,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls._process.pid)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reasoning_request(client, image_b64, separate_reasoning):
        return client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        image_content(image_b64),
                        text_content(
                            "What color is the shape and what shape is drawn "
                            "in this image? Answer with the color and shape only."
                        ),
                    ],
                }
            ],
            temperature=0,
            max_tokens=512,
            extra_body={
                "enable_thinking": True,
                "separate_reasoning": separate_reasoning,
            },
        )

    def setUp(self):
        self._client = openai.Client(
            api_key="sk-123456",
            base_url=f"{self._url}/v1",
            timeout=120,
        )

    def test_separate_reasoning_true(self):
        """separate_reasoning=True: reasoning stripped into reasoning_content."""
        response = self._reasoning_request(
            self._client,
            self._image_b64,
            separate_reasoning=True,
        )
        msg = response.choices[0].message
        content = msg.content or ""
        reasoning = msg.reasoning_content

        self.assertIsNotNone(reasoning)
        self.assertGreater(len(reasoning), 0)

        if content:
            r_head = reasoning[:30]
            self.assertNotIn(r_head, content)

        assert_color_and_shape(
            self,
            content,
            "green",
            "ellipse",
            prefix="sep_true: ",
        )

    def test_separate_reasoning_false(self):
        """separate_reasoning=False: reasoning merged into content."""
        response = self._reasoning_request(
            self._client,
            self._image_b64,
            separate_reasoning=False,
        )
        msg = response.choices[0].message
        text = msg.content

        self.assertIsNone(msg.reasoning_content)
        self.assertGreater(len(text), 80)

        assert_color_and_shape(
            self,
            text,
            "green",
            "ellipse",
            prefix="sep_false: ",
        )


if __name__ == "__main__":
    unittest.main()
