# SPDX-License-Identifier: Apache-2.0

import unittest
from dataclasses import fields

import torch

from sglang.srt.session.session_controller import SessionController
from sglang.srt.ug.context import UGSessionHandle
from sglang.srt.ug.runtime import (
    FakeUGModelRunner,
    UGInterleavedMessage,
    UGSegmentState,
    UGSessionRuntime,
    UGVelocityRequest,
)


class FakeTreeCache:
    def __init__(self):
        self.released_sessions = []

    def release_session(self, session_id):
        self.released_sessions.append(session_id)


class TestUGSessionRuntime(unittest.TestCase):
    def test_handle_does_not_expose_kv_allocator_details(self):
        names = {field.name for field in fields(UGSessionHandle)}

        self.assertEqual(
            names,
            {"session_id", "anchor_request_id", "context_length", "context_version"},
        )
        self.assertFalse(any("kv" in name.lower() for name in names))
        self.assertFalse(any("slot" in name.lower() for name in names))
        self.assertFalse(any("page" in name.lower() for name in names))

    def test_prefill_reuses_existing_srt_session(self):
        runtime = UGSessionRuntime(
            model_runner=FakeUGModelRunner(),
            session_controller=SessionController(FakeTreeCache()),
        )

        first = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="hello")],
            session_id="ug-session",
        )
        second = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="again")],
            session_id=first.session_id,
        )

        self.assertEqual(second.session_id, "ug-session")
        self.assertEqual(runtime.get_debug_counters(second)["prefill_count"], 2)
        self.assertEqual(runtime.get_debug_counters(second)["session_id"], "ug-session")

    def test_state_machine_rejects_invalid_g_transition(self):
        runtime = UGSessionRuntime(
            model_runner=FakeUGModelRunner(),
            session_controller=SessionController(FakeTreeCache()),
        )
        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="hello")]
        )

        self.assertEqual(runtime.records[handle.session_id].state, UGSegmentState.U_DECODE)
        with self.assertRaisesRegex(ValueError, "Cannot enter G_DENOISE"):
            runtime.predict_velocity(
                UGVelocityRequest(
                    session=handle,
                    latent_tokens=torch.zeros(1, 2, 4),
                    timestep=torch.tensor([1.0]),
                    latent_position_ids=torch.arange(2),
                    sampling_params=None,
                )
            )

    def test_velocity_does_not_repeat_prefill(self):
        runtime = UGSessionRuntime(
            model_runner=FakeUGModelRunner(),
            session_controller=SessionController(FakeTreeCache()),
        )
        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="draw an image")]
        )
        self.assertEqual(runtime.decode_next_segment(handle).type, "image_marker")

        velocity = runtime.predict_velocity(
            UGVelocityRequest(
                session=handle,
                latent_tokens=torch.ones(1, 3, 4),
                timestep=torch.tensor([0.5]),
                latent_position_ids=torch.arange(3),
                sampling_params=None,
            )
        )

        self.assertEqual(tuple(velocity.velocity.shape), (1, 3, 4))
        counters = runtime.get_debug_counters(handle)
        self.assertEqual(counters["prefill_count"], 1)
        self.assertEqual(counters["velocity_count"], 1)

    def test_close_releases_session(self):
        tree_cache = FakeTreeCache()
        runtime = UGSessionRuntime(
            model_runner=FakeUGModelRunner(),
            session_controller=SessionController(tree_cache),
        )
        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="hello")]
        )

        runtime.close_session(handle)

        self.assertEqual(tree_cache.released_sessions, [handle.session_id])
        self.assertTrue(runtime.records[handle.session_id].closed)


if __name__ == "__main__":
    unittest.main()
