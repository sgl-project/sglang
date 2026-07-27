"""
NPU multimodal tool call test.

"""

import json
import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ascend.test_npu_multimodal_utils import (
    Color,
    Shape,
    assert_color_and_shape,
    create_test_image,
    image_content,
    text_content,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=120, suite="full-1-npu-a3", nightly=True)


class TestMultimodalToolCall(CustomTestCase):
    """Verify tool calling works with image input.

    [Test Category] multimodal
    [Test Target] multimodal + tool calling
    """

    _model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
    _base_args = [
        "--device",
        "npu",
        "--attention-backend",
        "ascend",
        "--trust-remote-code",
        "--enable-multimodal",
        "--mm-attention-backend",
        "ascend_attn",
        "--mem-fraction-static",
        "0.4",
        "--tool-call-parser",
        "qwen",
    ]

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.BLUE, shape=Shape.RECTANGLE
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls._model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls._base_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_tool_call_with_image(self):
        """Send image with a registered tool, verify correct tool call format."""
        client = openai.Client(api_key="sk-123456", base_url=f"{self.base_url}/v1")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_image",
                    "description": "Analyze the content of an image and record a description",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "Detailed description of the image content",
                            },
                        },
                        "required": ["description"],
                    },
                },
            }
        ]

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        image_content(self._image_b64),
                        text_content(
                            "Analyze this image and call the analyze_image "
                            "function with a description of what you see."
                        ),
                    ],
                }
            ],
            temperature=0,
            max_tokens=256,
            tools=tools,
            tool_choice="auto",
        )

        message = response.choices[0].message
        self.assertIsNotNone(
            message.tool_calls, f"No tool calls. Content: '{message.content}'"
        )
        self.assertGreater(len(message.tool_calls), 0, "Empty tool_calls")

        tool_call = message.tool_calls[0]
        self.assertEqual(
            tool_call.function.name,
            "analyze_image",
            f"Expected 'analyze_image', got '{tool_call.function.name}'",
        )

        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            self.fail(f"Invalid JSON: {e}\n{tool_call.function.arguments}")

        self.assertIn("description", args, f"Missing 'description' in {args}")
        self.assertGreater(len(args["description"]), 0, "'description' empty")

        assert_color_and_shape(
            self,
            args["description"],
            "blue",
            "rectangle",
            prefix="test_tool_call_with_image: ",
        )


if __name__ == "__main__":
    unittest.main()
