# SPDX-License-Identifier: Apache-2.0

import unittest
from dataclasses import fields

import torch

from sglang.srt.session.session_controller import SessionController
from sglang.srt.ug.context import UGSessionHandle
from sglang.srt.ug.denoiser import SRTBackedUGDenoiserBridge
from sglang.srt.ug.interleaved import DEFAULT_UG_TEXT_MAX_NEW_TOKENS
from sglang.srt.ug.runtime import (
    FakeUGModelRunner,
    UGInterleavedMessage,
    UGSegmentState,
    UGSessionRuntime,
    UGVelocityRequest,
    UGVLMTextGenerationResult,
)


class FakeTreeCache:
    def __init__(self):
        self.released_sessions = []

    def release_session(self, session_id):
        self.released_sessions.append(session_id)


class RecordingThinkRunner(FakeUGModelRunner):
    def __init__(self):
        self.think_max_new_tokens = []

    def decode_vlm_text(self, *, runtime, session, max_new_tokens):
        del runtime
        self.think_max_new_tokens.append(max_new_tokens)
        return UGVLMTextGenerationResult(session=session, text="thinking")


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

        self.assertEqual(runtime.get_state(handle), UGSegmentState.U_DECODE)
        with self.assertRaisesRegex(ValueError, "Cannot predict UG velocity"):
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
        self.assertTrue(runtime.get_debug_counters(handle)["closed"])

    def test_think_default_max_new_tokens_is_not_smoke_sized(self):
        runner = RecordingThinkRunner()
        runtime = UGSessionRuntime(
            model_runner=runner,
            session_controller=SessionController(FakeTreeCache()),
        )
        bridge = SRTBackedUGDenoiserBridge(runtime)

        contexts = bridge.build_contexts(prompt="draw a cup", image=None, think=True)
        bridge.release_contexts(contexts)

        self.assertEqual(
            runner.think_max_new_tokens,
            [DEFAULT_UG_TEXT_MAX_NEW_TOKENS],
        )


if __name__ == "__main__":
    unittest.main()
