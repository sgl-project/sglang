# SPDX-License-Identifier: Apache-2.0

import unittest
from base64 import b64encode
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from sglang.omni.backends.ar.srt import SRTARBackend
from sglang.omni.protocol import GeneratedSegment, OmniInputSegment, OmniRequest
from sglang.srt.omni_session.runtime_protocol import (
    OmniContextBundle as SRTOmniContextBundle,
    OmniContextHandle,
    OmniSessionHandle,
)


class TestOmniSRTBackend(unittest.TestCase):
    def test_interleave_exposes_text_boundary_and_commits_generated_image(self):
        bridge = _FakeBridge(
            pre_image_segments=[
                {
                    "type": "text",
                    "text": "I",
                    "metadata": {"token_ids": [1]},
                },
                {
                    "type": "text",
                    "text": " generated",
                    "metadata": {"token_ids": [2]},
                },
            ]
        )
        backend = SRTARBackend(bridge)
        request = OmniRequest(
            messages=(OmniInputSegment(type="text", text="draw"),),
            mode="interleave",
        )

        context = backend.prepare_context(request)
        first = backend.decode_until_boundary(context, request=request)
        second = backend.decode_until_boundary(context, request=request)
        backend.append_generated_segment(
            context,
            GeneratedSegment(type="image", image="image"),
            request=request,
        )
        third = backend.decode_until_boundary(context, request=request)

        self.assertEqual(
            ("text", "I generated", (1, 2)),
            (first.type, first.text, first.token_ids),
        )
        self.assertEqual("image", second.type)
        self.assertEqual("done", third.type)
        self.assertEqual(1, bridge.commit_count)

    def test_image_payload_accepts_b64_json(self):
        bridge = _ImageCaptureBridge()
        backend = SRTARBackend(bridge)
        request = OmniRequest(
            messages=(
                OmniInputSegment(type="text", text="describe"),
                OmniInputSegment(
                    type="image",
                    image={"b64_json": _tiny_png_b64(), "mime_type": "image/png"},
                ),
            ),
            mode="t2i",
        )

        backend.prepare_context(request)

        self.assertEqual((2, 2), bridge.images[0].size)
        self.assertEqual("RGB", bridge.images[0].mode)

    def test_vlm_mode_returns_text_and_releases_session(self):
        bridge = _FakeVLMBridge()
        backend = SRTARBackend(bridge)
        request = OmniRequest(
            messages=(OmniInputSegment(type="text", text="what is this"),),
            mode="vlm",
            metadata={"max_new_tokens": 4},
        )

        context = backend.prepare_context(request)
        first = backend.decode_until_boundary(context, request=request)
        second = backend.decode_until_boundary(context, request=request)
        backend.release(context)

        self.assertEqual("text", first.type)
        self.assertEqual("answer", first.text)
        self.assertEqual((3, 4), first.token_ids)
        self.assertEqual("done", second.type)
        self.assertEqual(4, bridge.max_new_tokens)
        self.assertEqual(["s1"], bridge.runtime.closed_sessions)


class _FakeBridge:
    def __init__(self, pre_image_segments=None):
        self.pre_image_segments = (
            [
                {
                    "type": "text",
                    "text": "thinking",
                    "metadata": {"token_ids": [1, 2]},
                }
            ]
            if pre_image_segments is None
            else pre_image_segments
        )
        self.commit_count = 0
        self.session_ids = []

    def prepare_ar_context_from_messages(self, **kwargs):
        self.session_ids.append(kwargs.get("session_id") or "s0")
        session = OmniSessionHandle(
            session_id=kwargs.get("session_id") or "s0",
            anchor_request_id="r0",
            context_length=4,
            context_version=len(self.session_ids),
        )
        full = OmniContextHandle(
            request_id="r0",
            token_count=4,
            session=session,
            metadata={"pre_image_segments": self.pre_image_segments},
        )
        text_cfg = OmniContextHandle(
            request_id="r0:text_cfg",
            token_count=4,
            session=session,
        )
        image_cfg = OmniContextHandle(
            request_id="r0:image_cfg",
            token_count=4,
            session=session,
        )
        return SRTOmniContextBundle(full=full, text_cfg=text_cfg, image_cfg=image_cfg)

    def commit_generated_segment(self, **kwargs):
        self.commit_count += 1

    def continue_ar_decode(self, **kwargs):
        return SimpleNamespace(type="done", text=None, token_ids=())

    def release(self, contexts):
        return None


class _FakeRuntime:
    def __init__(self):
        self.closed_sessions = []

    def close_session(self, session):
        self.closed_sessions.append(session.session_id)


class _FakeVLMBridge:
    def __init__(self):
        self.runtime = _FakeRuntime()
        self.max_new_tokens = None

    def generate_vlm_text(self, *, messages, max_new_tokens):
        self.max_new_tokens = max_new_tokens
        session = OmniSessionHandle(
            anchor_request_id="r1",
            session_id="s1",
            context_length=8,
            context_version=1,
        )
        return SimpleNamespace(
            session=session,
            text="answer",
            token_ids=(1, 2),
            next_token_ids=(3, 4),
            position_ids=(5, 6),
        )


class _ImageCaptureBridge(_FakeBridge):
    def __init__(self):
        super().__init__(pre_image_segments=[])
        self.images = []

    def prepare_ar_context_from_messages(self, **kwargs):
        for message in kwargs["messages"]:
            if message["type"] == "image":
                self.images.append(message["image"])
        return super().prepare_ar_context_from_messages(**kwargs)


def _tiny_png_b64():
    image = Image.new("RGB", (2, 2), (255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return b64encode(buffer.getvalue()).decode("ascii")


if __name__ == "__main__":
    unittest.main()
