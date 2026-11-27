"""Inkling-2-Mini NVFP4 on one Blackwell GPU with DSPARK speculative decoding.

The unit tests under ``test/registered/unit/models/test_inkling2*.py`` and
``test_inkling2_loop.py`` cover the looped core in isolation; this one boots the
real checkpoint, so it catches what only shows up end to end: a loop or DSPARK
regression that keeps the server healthy while the answers go wrong, and
breakage in the image and tool-call paths of the chat API.
"""

import base64
import io
import json
import os
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

register_cuda_ci(est_time=600, stage="extra-b", runner_config="2-gpu-gb300")

MODEL = os.environ.get(
    "INKLING2_MINI_TEST_MODEL_PATH", "thinkingmachines/Inkling-2-Mini"
)

SERVER_ARGS = [
    "--tp",
    "1",
    "--attention-backend",
    "fa4",
    "--dtype",
    "bfloat16",
    "--page-size",
    "128",
    "--speculative-algorithm",
    "DSPARK",
    "--grammar-backend",
    "xgrammar",
    "--enable-multimodal",
    "--reasoning-parser",
    "inkling",
    "--tool-call-parser",
    "inkling2",
]

# Measured 0.93 on B200 (temperature 0, no thinking); set several sigma of
# 200-question sampling noise below, so only a real accuracy drop trips it.
GSM8K_THRESHOLD = 0.85

# Measured 3.64 on B200; a draft that stops matching the target falls toward 1.0.
ACCEPT_LENGTH_THRESHOLD = 3.0


def _shapes_image_url() -> str:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([60, 60, 220, 220], fill="red")
    draw.ellipse([240, 240, 400, 400], fill="blue")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class TestInkling2MiniDSpark(GSM8KMixin, CustomTestCase):
    gsm8k_backend = "sgl_eval"
    gsm8k_score_threshold = GSM8K_THRESHOLD
    gsm8k_num_examples = 200
    gsm8k_num_threads = 64
    gsm8k_max_tokens = 8192
    gsm8k_temperature = 0.0
    gsm8k_top_p = 1.0

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=SERVER_ARGS,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def _chat(self, messages, **kwargs) -> dict:
        response = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 1024,
                **kwargs,
            },
            timeout=600,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["choices"][0]["message"]

    def test_gsm8k(self):
        """Answer quality through the looped core and the DSPARK verify path;
        the accept-length floor catches a draft that stopped matching the target."""
        super().test_gsm8k()
        states = requests.get(self.base_url + "/server_info", timeout=30).json()[
            "internal_states"
        ]
        accept_length = max(state["avg_spec_accept_length"] for state in states)
        self.assertGreater(accept_length, ACCEPT_LENGTH_THRESHOLD)

    def test_image_understanding(self):
        """The vision tokens must reach the language model."""
        message = self._chat(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What shapes and colors are in this image? "
                            "Answer briefly.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": _shapes_image_url()},
                        },
                    ],
                }
            ]
        )
        answer = message["content"].lower()
        for word in ("red", "blue", "square", "circle"):
            self.assertIn(word, answer)

    def test_tool_call(self):
        """Tools render through the chat template and the model's call parses
        back into an OpenAI tool_calls entry."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        message = self._chat(
            [
                {
                    "role": "user",
                    "content": "What's the weather in Paris right now? Use the tool.",
                }
            ],
            tools=tools,
        )
        self.assertTrue(message["tool_calls"], message)
        call = message["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "get_weather")
        self.assertEqual(json.loads(call["arguments"]), {"city": "Paris"})


if __name__ == "__main__":
    unittest.main()
