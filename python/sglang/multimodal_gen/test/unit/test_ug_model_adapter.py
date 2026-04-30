# SPDX-License-Identifier: Apache-2.0

import unittest
from dataclasses import fields

import torch

from sglang.srt.session.session_controller import SessionController
from sglang.srt.ug.adapter import (
    UGModelAppendImageResult,
    UGModelPrefillResult,
    UGModelRunnerAdapter,
    UGModelSessionView,
)
from sglang.srt.ug.context import UGSRTRequestView
from sglang.srt.ug.runtime import (
    UGDecodeResult,
    UGInterleavedMessage,
    UGSessionRuntime,
    UGVelocityRequest,
)


class FakeTreeCache:
    def __init__(self):
        self.released_sessions = []

    def release_session(self, session_id):
        self.released_sessions.append(session_id)


class RecordingUGModelAdapter:
    def __init__(self):
        self.events = []
        self.session_views = []
        self.decode_count = 0

    def prefill_interleaved(self, *, session, messages):
        self._record("prefill", session)
        token_count = 0
        for message in messages:
            if message.type == "image":
                token_count += 2
            elif message.type == "text":
                token_count += len(str(message.content).split())
        return UGModelPrefillResult(added_tokens=token_count)

    def decode_next_segment(self, *, session):
        self._record("decode", session)
        self.decode_count += 1
        if self.decode_count == 1:
            return UGDecodeResult(type="image_marker")
        if self.decode_count == 2:
            return UGDecodeResult(type="text", text="adapter_text_after_image")
        return UGDecodeResult(type="done")

    def predict_velocity_from_session(self, *, session, request):
        self._record("velocity", session)
        scale = 2.0 + session.srt_request_count * 0.1
        return request.latent_tokens + scale * request.timestep.reshape(-1, 1, 1).to(
            request.latent_tokens
        )

    def append_generated_image(self, *, session, image):
        del image
        self._record("append_image", session)
        return UGModelAppendImageResult(added_tokens=2)

    def close_session(self, *, session_id):
        self.events.append(("close", session_id))

    def _record(self, event, session):
        self.events.append((event, session.handle.session_id))
        self.session_views.append(session)


class RecordingSRTForwardAdapter(RecordingUGModelAdapter):
    def __init__(self):
        super().__init__()
        self.srt_forward_requests = []

    def observe_srt_u_forward(self, *, session, request, messages):
        del messages
        self._record("srt_u_forward", session)
        self.srt_forward_requests.append(request)


class TestUGModelRunnerAdapter(unittest.TestCase):
    def test_session_view_does_not_expose_kv_details(self):
        names = {field.name for field in fields(UGModelSessionView)}

        self.assertFalse(any("kv" in name.lower() for name in names))
        self.assertFalse(any("slot" in name.lower() for name in names))
        self.assertFalse(any("page" in name.lower() for name in names))
        self.assertFalse(any("allocator" in name.lower() for name in names))

    def test_srt_request_view_does_not_expose_kv_details(self):
        names = {field.name for field in fields(UGSRTRequestView)}

        self.assertFalse(any("kv" in name.lower() for name in names))
        self.assertFalse(any("slot" in name.lower() for name in names))
        self.assertFalse(any("page" in name.lower() for name in names))
        self.assertFalse(any("allocator" in name.lower() for name in names))

    def test_runner_forwards_safe_srt_u_forward_view(self):
        model_adapter = RecordingSRTForwardAdapter()
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(model_adapter),
            session_controller=SessionController(FakeTreeCache()),
            srt_u_decode_max_new_tokens=1,
        )

        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="draw a boat")],
            session_id="adapter-srt-forward",
        )
        runtime.decode_next_segment(handle)

        self.assertEqual(
            [request.request_id for request in model_adapter.srt_forward_requests],
            ["adapter-srt-forward:u1", "adapter-srt-forward:d1"],
        )
        self.assertEqual(
            [request.state for request in model_adapter.srt_forward_requests],
            ["u_prefill", "u_decode"],
        )
        self.assertEqual(
            [request.max_new_tokens for request in model_adapter.srt_forward_requests],
            [0, 1],
        )
        self.assertEqual(
            model_adapter.srt_forward_requests[0].input_text, "draw a boat"
        )

    def test_adapter_entrypoints_reuse_srt_session_context(self):
        tree_cache = FakeTreeCache()
        controller = SessionController(tree_cache)
        model_adapter = RecordingUGModelAdapter()
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(model_adapter),
            session_controller=controller,
        )

        handle = runtime.prefill_interleaved(
            [
                UGInterleavedMessage(type="image", content=object()),
                UGInterleavedMessage(type="text", content="draw then explain"),
            ],
            session_id="adapter-session",
        )
        marker = runtime.decode_next_segment(handle)
        self.assertEqual(marker.type, "image_marker")

        latents = torch.zeros(1, 2, 4)
        for step in range(2):
            response = runtime.predict_velocity(
                UGVelocityRequest(
                    session=handle,
                    latent_tokens=latents,
                    timestep=torch.tensor([1.0 - step * 0.5]),
                    latent_position_ids=torch.arange(2),
                    sampling_params=None,
                )
            )
            latents = response.velocity

        handle_after_image = runtime.append_generated_image(handle, image=object())
        post_image = runtime.decode_next_segment(handle_after_image)
        runtime.close_session(handle_after_image)

        self.assertEqual(post_image.type, "text")
        self.assertEqual(post_image.text, "adapter_text_after_image")
        self.assertEqual(
            model_adapter.events,
            [
                ("prefill", "adapter-session"),
                ("decode", "adapter-session"),
                ("velocity", "adapter-session"),
                ("velocity", "adapter-session"),
                ("append_image", "adapter-session"),
                ("decode", "adapter-session"),
                ("close", "adapter-session"),
            ],
        )

        prefill_view = model_adapter.session_views[0]
        append_view = model_adapter.session_views[4]
        self.assertEqual(prefill_view.srt_request_count, 1)
        self.assertEqual(prefill_view.srt_last_request_id, handle.anchor_request_id)
        self.assertEqual(prefill_view.srt_mm_offsets, ((1, 3),))
        self.assertEqual(append_view.srt_request_count, 2)
        self.assertEqual(
            append_view.srt_last_request_id,
            handle_after_image.anchor_request_id,
        )
        self.assertEqual(append_view.srt_mm_offsets, ((1, 3), (6, 8)))

        counters = runtime.get_debug_counters("adapter-session")
        self.assertTrue(counters["closed"])
        self.assertEqual(counters["prefill_count"], 1)
        self.assertEqual(counters["velocity_count"], 2)
        self.assertEqual(counters["append_image_count"], 1)
        self.assertEqual(counters["decode_count"], 2)
        self.assertEqual(counters["srt_request_count"], 2)
        self.assertEqual(tree_cache.released_sessions, ["adapter-session"])

    def test_adapter_decode_view_includes_srt_u_decode_metadata(self):
        model_adapter = RecordingUGModelAdapter()
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(model_adapter),
            session_controller=SessionController(FakeTreeCache()),
            srt_u_decode_max_new_tokens=1,
        )

        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="draw a boat")],
            session_id="adapter-u-decode-session",
        )
        runtime.decode_next_segment(handle)

        decode_view = model_adapter.session_views[1]
        self.assertEqual(decode_view.srt_request_count, 2)
        self.assertEqual(
            decode_view.srt_last_request_id,
            "adapter-u-decode-session:d1",
        )
        self.assertEqual(
            decode_view.metadata["srt_u_decode_request_count"],
            1,
        )
        self.assertEqual(
            decode_view.metadata["srt_last_u_decode_request_id"],
            "adapter-u-decode-session:d1",
        )
        self.assertIn("srt_last_u_decode_text", decode_view.metadata)


if __name__ == "__main__":
    unittest.main()
