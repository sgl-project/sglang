# SPDX-License-Identifier: Apache-2.0

import json
import unittest

from fastapi.testclient import TestClient

from sglang.omni.core.coordinator import OmniCoordinator
from sglang.omni.core.protocol import (
    GeneratedSegment,
    OmniBoundary,
    OmniContextBundle,
    OmniContextRef,
)
from sglang.omni.entrypoints.http_server import create_app


class _ScriptedARBackend:
    def __init__(self, boundaries):
        self._boundaries = list(boundaries)

    def begin_request_context(self, request, *, stream_sink=None):
        return OmniContextBundle(full=OmniContextRef(context_id="scripted"))

    def append_input_segments(self, context, request, *, stream_sink=None):
        return context

    def decode_until_boundary(self, context, *, request, stream_sink=None):
        if not self._boundaries:
            return OmniBoundary(type="done")
        return self._boundaries.pop(0)

    def append_generated_segment(self, context, segment, *, request):
        return context

    def get_context_ops(self, context):
        return None

    def release(self, context):
        return None


class _ImageBackend:
    def generate_segment(self, request, context_ops):
        return GeneratedSegment(type="image", image={"b64_json": "abc"})


class TestOmniHttp(unittest.TestCase):
    def test_generate_endpoint_returns_mixed_segments(self):
        app = create_app(
            coordinator=OmniCoordinator(
                _ScriptedARBackend(
                    [
                        OmniBoundary(type="text", text="before"),
                        OmniBoundary(type="image"),
                        OmniBoundary(type="done"),
                    ]
                ),
                _ImageBackend(),
            )
        )
        client = TestClient(app)

        response = client.post(
            "/v1/omni/generate",
            json={"messages": [{"type": "text", "text": "draw"}]},
        )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual(["text", "image"], [s["type"] for s in payload["segments"]])

    def test_generate_endpoint_streams_segment_events(self):
        app = create_app(
            coordinator=OmniCoordinator(
                _ScriptedARBackend(
                    [
                        OmniBoundary(type="text", text="before"),
                        OmniBoundary(type="image"),
                        OmniBoundary(type="done"),
                    ]
                ),
                _ImageBackend(),
            )
        )
        client = TestClient(app)

        response = client.post(
            "/v1/omni/generate",
            json={
                "stream": True,
                "messages": [{"type": "text", "text": "draw"}],
            },
        )

        self.assertEqual(200, response.status_code)
        events = [
            json.loads(line[len("data: ") :])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        self.assertEqual(
            ["text_delta", "text_end", "image_start", "image", "done"],
            [event["type"] for event in events],
        )
        self.assertEqual("before", events[0]["delta"])


if __name__ == "__main__":
    unittest.main()
