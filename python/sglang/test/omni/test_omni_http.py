# SPDX-License-Identifier: Apache-2.0

import unittest

from fastapi.testclient import TestClient

from sglang.omni.coordinator import OmniCoordinator
from sglang.omni.entrypoints.http_server import create_app
from sglang.omni.protocol import (
    GeneratedSegment,
    OmniBoundary,
    OmniContextBundle,
    OmniContextRef,
)


class _ScriptedARBackend:
    def __init__(self, boundaries):
        self._boundaries = list(boundaries)

    def prepare_context(self, request):
        del request
        return OmniContextBundle(full=OmniContextRef(context_id="scripted"))

    def decode_until_boundary(self, context, *, request):
        del context, request
        if not self._boundaries:
            return OmniBoundary(type="done")
        return self._boundaries.pop(0)

    def append_generated_segment(self, context, segment, *, request):
        del segment, request
        return context

    def get_context_ops(self, context):
        return None

    def release(self, context):
        del context


class _ImageBackend:
    def generate_segment(self, request, context_ops):
        del request, context_ops
        return GeneratedSegment(type="image", image={"b64_json": "abc"})


class TestOmniHttp(unittest.TestCase):
    def test_generate_endpoint_returns_mixed_segments(self):
        app = create_app(
            orchestrator=OmniCoordinator(
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


if __name__ == "__main__":
    unittest.main()
