# SPDX-License-Identifier: Apache-2.0

import unittest
from base64 import b64encode
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from sglang.omni.backends.srt import SRTARBackend
from sglang.omni.protocol import GeneratedSegment, OmniInputSegment, OmniRequest


class TestOmniSRTBackend(unittest.TestCase):
    def test_prepare_exposes_pre_image_text_before_first_image_boundary(self):
        backend = SRTARBackend(_FakeBridge())
        request = OmniRequest(messages=(OmniInputSegment(type="text", text="draw"),))

        context = backend.prepare_context(request)
        first = backend.decode_until_boundary(context, request=request)
        second = backend.decode_until_boundary(context, request=request)

        self.assertEqual("text", first.type)
        self.assertEqual("thinking", first.text)
        self.assertEqual((1, 2), first.token_ids)
        self.assertEqual("image", second.type)

    def test_t2i_stops_after_image_without_commit(self):
        bridge = _FakeBridge(pre_image_segments=[])
        backend = SRTARBackend(bridge)
        request = OmniRequest(
            messages=(OmniInputSegment(type="text", text="draw"),),
            mode="t2i",
        )

        context = backend.prepare_context(request)
        first = backend.decode_until_boundary(context, request=request)
        context = backend.append_generated_segment(
            context,
            GeneratedSegment(type="image", image="image"),
            request=request,
        )
        second = backend.decode_until_boundary(context, request=request)

        self.assertEqual("image", first.type)
        self.assertEqual("done", second.type)
        self.assertEqual(0, bridge.commit_count)

    def test_interleave_commits_before_continuing_decode(self):
        bridge = _FakeBridge(pre_image_segments=[])
        backend = SRTARBackend(bridge)
        request = OmniRequest(
            messages=(OmniInputSegment(type="text", text="draw"),),
            mode="interleave",
        )

        context = backend.prepare_context(request)
        first = backend.decode_until_boundary(context, request=request)
        backend.append_generated_segment(
            context,
            GeneratedSegment(type="image", image="image"),
            request=request,
        )
        second = backend.decode_until_boundary(context, request=request)

        self.assertEqual("image", first.type)
        self.assertEqual("done", second.type)
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

    def prepare_u_context_from_messages(self, **kwargs):
        del kwargs
        session = SimpleNamespace(session_id="s0", context_version=0)
        full = SimpleNamespace(
            request_id="r0",
            token_count=4,
            session=session,
            metadata={"pre_image_segments": self.pre_image_segments},
        )
        return SimpleNamespace(full=full, text_cfg=None, image_cfg=None)

    def commit_generated_segment(self, **kwargs):
        del kwargs
        self.commit_count += 1

    def continue_u_decode(self, **kwargs):
        del kwargs
        return SimpleNamespace(type="done")

    def release(self, contexts):
        del contexts


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
        del messages
        self.max_new_tokens = max_new_tokens
        session = SimpleNamespace(
            anchor_request_id="r1",
            session_id="s1",
            context_length=8,
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

    def prepare_u_context_from_messages(self, **kwargs):
        for message in kwargs["messages"]:
            if message["type"] == "image":
                self.images.append(message["image"])
        return super().prepare_u_context_from_messages(**kwargs)


def _tiny_png_b64():
    image = Image.new("RGB", (2, 2), (255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return b64encode(buffer.getvalue()).decode("ascii")


if __name__ == "__main__":
    unittest.main()
