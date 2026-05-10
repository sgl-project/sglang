# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
from sglang.srt.entrypoints.openai.serving_responses import (
    _responses_input_to_omni_segments,
    _responses_omni_task,
    _responses_sampling_params,
)


class TestOmniResponsesAdapter(unittest.TestCase):
    def test_image_generation_input_maps_to_omni_segments(self):
        request = ResponsesRequest(
            model="sensenova-u1",
            tools=[{"type": "image_generation"}],
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Edit this image."},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,AAAA",
                            "detail": "auto",
                        },
                    ],
                }
            ],
        )

        self.assertEqual(
            _responses_input_to_omni_segments(request),
            [
                {"type": "text", "text": "Edit this image."},
                {
                    "type": "image",
                    "image": {"b64_json": "data:image/png;base64,AAAA"},
                },
            ],
        )

    def test_image_generation_tool_size_maps_to_sampling_params(self):
        params = _responses_sampling_params(
            {"sampling_params": {"num_steps": 30}, "seed": 7},
            SimpleNamespace(size="1024x1024"),
        )

        self.assertEqual(
            params,
            {"num_steps": 30, "width": 1024, "height": 1024, "seed": 7},
        )

    def test_image_generation_action_maps_to_omni_task(self):
        self.assertEqual(
            "t2i",
            _responses_omni_task({}, SimpleNamespace(action="generate"), None),
        )
        self.assertEqual(
            "edit",
            _responses_omni_task({}, SimpleNamespace(action="edit"), None),
        )
        self.assertEqual(
            "interleave",
            _responses_omni_task({}, SimpleNamespace(action="auto"), "session-0"),
        )


if __name__ == "__main__":
    unittest.main()
