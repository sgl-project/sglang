# SPDX-License-Identifier: Apache-2.0

import unittest
from dataclasses import fields

import torch

from sglang.srt.session.session_controller import SessionController
from sglang.srt.ug.context import UGSessionHandle
from sglang.srt.ug.denoiser import SRTBackedUGMiddleBridge
from sglang.srt.ug.interleaved import DEFAULT_UG_TEXT_MAX_NEW_TOKENS
from sglang.srt.ug.runtime import (
    UGDecodeResult,
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


class RecordingUGModelRunner:
    def prefill_interleaved(self, *, record, messages):
        del record
        token_count = 0
        for message in messages:
            if message.type == "text":
                token_count += len(str(message.content).split())
            elif message.type == "image":
                token_count += 2
        return token_count

    def decode_next_segment(self, *, record):
        if record.append_image_count == 0 and record.decode_count == 0:
            return UGDecodeResult(type="image_marker")
        if record.append_image_count > 0 and record.decode_count == 1:
            return UGDecodeResult(type="text", text="generated_text_after_image")

        return UGDecodeResult(type="done")

    def predict_velocity_from_session(self, *, request, record):
        scale = 1.0 + record.context_length * 0.01 + record.context_version * 0.001
        return request.latent_tokens + scale * request.timestep.reshape(-1, 1, 1).to(
            request.latent_tokens
        )

    def prepare_latents_from_session(self, *, request, record):
        del request, record
        return None

    def append_generated_image(self, *, record, image):
        del record, image
        return 2

    def decode_latents_to_image(self, *, request, record):
        del request, record
        return None

    def close_session(self, *, session_id):
        del session_id


class RecordingThinkRunner(RecordingUGModelRunner):
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
            model_runner=RecordingUGModelRunner(),
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
            model_runner=RecordingUGModelRunner(),
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
            model_runner=RecordingUGModelRunner(),
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
            model_runner=RecordingUGModelRunner(),
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
        bridge = SRTBackedUGMiddleBridge(runtime)

        contexts = bridge.prepare_u_context(prompt="draw a cup", image=None, think=True)
        bridge.release(contexts)

        self.assertEqual(
            runner.think_max_new_tokens,
            [DEFAULT_UG_TEXT_MAX_NEW_TOKENS],
        )


if __name__ == "__main__":
    unittest.main()
