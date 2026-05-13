# SPDX-License-Identifier: Apache-2.0

import unittest
from base64 import b64encode
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from sglang.omni.backends.ar.srt import SRTARBackend
from sglang.omni.core.interleaved import (
    INTERLEAVED_BOUNDARY_MODALITY_KEY,
    INTERLEAVED_BOUNDARY_POSITION_ID_KEY,
    INTERLEAVED_BOUNDARY_TOKEN_ID_KEY,
    INTERLEAVED_GENERATION_BOUNDARY_METADATA_KEY,
    INTERLEAVED_GENERATION_TOKEN_COUNT_KEY,
    INTERLEAVED_GENERATION_TOKEN_COUNT_SOURCE_KEY,
)
from sglang.omni.core.protocol import GeneratedSegment, OmniInputSegment, OmniRequest
from sglang.srt.omni_session.runtime_types import (
    OmniContextBundle as SRTOmniContextBundle,
    OmniContextHandle,
    OmniSessionHandle,
)


class TestOmniSRTBackend(unittest.TestCase):
    def test_interleave_exposes_text_boundary_and_commits_generated_image(self):
        session_adapter = _FakeSessionAdapter(
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
        backend = SRTARBackend(session_adapter)
        request = OmniRequest(
            messages=(OmniInputSegment(type="text", text="draw"),),
            mode="interleave",
        )

        context = backend.begin_request_context(request)
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
        self.assertEqual(1, session_adapter.commit_count)

    def test_image_boundary_metadata_reaches_generation_context_ops(self):
        boundary_metadata = {
            INTERLEAVED_BOUNDARY_TOKEN_ID_KEY: 123,
            INTERLEAVED_BOUNDARY_POSITION_ID_KEY: 226,
            INTERLEAVED_GENERATION_TOKEN_COUNT_KEY: 256,
            INTERLEAVED_GENERATION_TOKEN_COUNT_SOURCE_KEY: "ar",
        }
        session_adapter = _FakeSessionAdapter(
            pre_image_segments=[],
            pre_image_boundary_metadata=boundary_metadata,
        )
        backend = SRTARBackend(session_adapter)
        request = OmniRequest(
            messages=(OmniInputSegment(type="text", text="draw"),),
            mode="interleave",
        )

        context = backend.begin_request_context(request)
        boundary = backend.decode_until_boundary(context, request=request)
        context_ops = backend.get_context_ops(context)

        self.assertEqual("image", boundary.type)
        self.assertEqual(
            {
                **boundary_metadata,
                INTERLEAVED_BOUNDARY_MODALITY_KEY: "image",
            },
            context_ops.metadata[INTERLEAVED_GENERATION_BOUNDARY_METADATA_KEY],
        )

    def test_interleave_can_finish_before_image_marker(self):
        session_adapter = _FakeSessionAdapter(
            pre_image_segments=[
                {
                    "type": "text",
                    "text": "hello",
                    "metadata": {"token_ids": [1]},
                },
            ],
            reached_image_marker=False,
        )
        backend = SRTARBackend(session_adapter)
        request = OmniRequest(
            messages=(OmniInputSegment(type="text", text="hi"),),
            mode="interleave",
        )

        context = backend.begin_request_context(request)
        first = backend.decode_until_boundary(context, request=request)
        second = backend.decode_until_boundary(context, request=request)

        self.assertEqual(("text", "hello"), (first.type, first.text))
        self.assertEqual("done", second.type)
        self.assertEqual(0, session_adapter.commit_count)

    def test_image_payload_accepts_b64_json(self):
        session_adapter = _ImageCaptureSessionAdapter()
        backend = SRTARBackend(session_adapter)
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

        backend.begin_request_context(request)

        self.assertEqual((2, 2), session_adapter.images[0].size)
        self.assertEqual("RGB", session_adapter.images[0].mode)

    def test_vlm_mode_returns_text_and_releases_session(self):
        session_adapter = _FakeVLMSessionAdapter()
        backend = SRTARBackend(session_adapter)
        request = OmniRequest(
            messages=(OmniInputSegment(type="text", text="what is this"),),
            mode="vlm",
            metadata={"max_new_tokens": 4},
        )

        context = backend.begin_request_context(request)
        first = backend.decode_until_boundary(context, request=request)
        second = backend.decode_until_boundary(context, request=request)
        backend.release(context)

        self.assertEqual("text", first.type)
        self.assertEqual("answer", first.text)
        self.assertEqual((3, 4), first.token_ids)
        self.assertEqual("done", second.type)
        self.assertEqual(4, session_adapter.max_new_tokens)
        self.assertEqual(["s1"], session_adapter.runtime.closed_sessions)

    def test_context_ops_resolves_condition_path_role_map(self):
        session_adapter = _FakeSessionAdapter(pre_image_segments=[])
        session_adapter.condition_path_roles["edit_img_condition"] = (
            "u1_edit_img_condition"
        )
        backend = SRTARBackend(session_adapter)
        context = backend.begin_request_context(
            OmniRequest(
                messages=(OmniInputSegment(type="text", text="edit"),),
                mode="edit",
            )
        )

        context_ops = backend.get_context_ops(context)

        self.assertEqual(
            "u1_edit_img_condition",
            context_ops.get_role("edit_img_condition_role", "fallback"),
        )


class _FakeSessionAdapter:
    generation_kind = "pixel_flow"

    def __init__(
        self,
        pre_image_segments=None,
        reached_image_marker=True,
        pre_image_boundary_metadata=None,
    ):
        self.condition_path_roles = {}
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
        self.reached_image_marker = reached_image_marker
        self.pre_image_boundary_metadata = dict(pre_image_boundary_metadata or {})
        self.commit_count = 0
        self.session_ids = []

    def prefill_and_decode_to_image_boundary(self, **kwargs):
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
            metadata={
                "pre_image_segments": self.pre_image_segments,
                "pre_image_reached_image_marker": self.reached_image_marker,
                "pre_image_boundary_metadata": self.pre_image_boundary_metadata,
            },
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
        return SimpleNamespace(type="done", text=None, token_ids=(), metadata={})

    def release(self, contexts):
        return None

    def get_condition_path_role(self, name, default):
        key = name[:-5] if name.endswith("_role") else name
        return self.condition_path_roles.get(key, default)


class _FakeRuntime:
    def __init__(self):
        self.closed_sessions = []

    def close_session(self, session):
        self.closed_sessions.append(session.session_id)


class _FakeVLMSessionAdapter:
    def __init__(self):
        self.runtime = _FakeRuntime()
        self.max_new_tokens = None

    def generate_vlm_answer(self, *, messages, max_new_tokens):
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


class _ImageCaptureSessionAdapter(_FakeSessionAdapter):
    def __init__(self):
        super().__init__(pre_image_segments=[])
        self.images = []

    def prefill_and_decode_to_image_boundary(self, **kwargs):
        for message in kwargs["messages"]:
            if message["type"] == "image":
                self.images.append(message["image"])
        return super().prefill_and_decode_to_image_boundary(**kwargs)


def _tiny_png_b64():
    image = Image.new("RGB", (2, 2), (255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return b64encode(buffer.getvalue()).decode("ascii")


if __name__ == "__main__":
    unittest.main()
