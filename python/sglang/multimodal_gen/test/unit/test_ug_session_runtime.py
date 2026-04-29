# SPDX-License-Identifier: Apache-2.0

import unittest
from dataclasses import fields
from types import SimpleNamespace

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


class RecordingSRTRequestExecutor:
    def __init__(self):
        self.events = []

    def execute_ug_request(self, *, record, req, state):
        self.events.append(
            {
                "session_id": record.session_id,
                "state": state.value,
                "rid": req.rid,
                "origin_input_ids": list(req.origin_input_ids),
                "mm_offsets": (
                    [
                        offset
                        for item in getattr(req.multimodal_inputs, "mm_items", [])
                        for offset in getattr(item, "offsets", [])
                    ]
                    if req.multimodal_inputs is not None
                    else []
                ),
                "finished_reason": req.finished_reason,
                "max_new_tokens": req.sampling_params.max_new_tokens,
                "output_ids": list(req.output_ids),
            }
        )


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

    def test_runtime_uses_existing_srt_session_lifecycle(self):
        class FakeSessionController:
            def __init__(self):
                self.opened = []
                self.closed = []
                self.sessions = set()

            def __contains__(self, session_id):
                return session_id in self.sessions

            def open(self, req):
                self.opened.append(req.session_id)
                self.sessions.add(req.session_id)
                return SimpleNamespace(success=True)

            def close(self, req):
                self.closed.append(req.session_id)
                self.sessions.remove(req.session_id)

        controller = FakeSessionController()
        runtime = UGSessionRuntime(
            model_runner=FakeUGModelRunner(), session_controller=controller
        )

        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="hello")],
            session_id="srt-session",
        )
        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="again")],
            session_id=handle.session_id,
        )

        self.assertEqual(handle.session_id, "srt-session")
        self.assertEqual(controller.opened, ["srt-session"])
        self.assertEqual(runtime.get_debug_counters(handle)["prefill_count"], 2)

        runtime.close_session(handle)
        self.assertEqual(controller.closed, ["srt-session"])

    def test_prefill_uses_srt_create_req_and_appends_to_same_session(self):
        controller = SessionController(FakeTreeCache())
        runtime = UGSessionRuntime(
            model_runner=FakeUGModelRunner(), session_controller=controller
        )

        first = runtime.prefill_interleaved(
            [
                UGInterleavedMessage(type="image", content=object()),
                UGInterleavedMessage(type="text", content="hello"),
            ],
            session_id="srt-backed-prefill",
        )
        first_debug = runtime.get_debug_counters(first)

        self.assertEqual(first_debug["srt_request_count"], 1)
        self.assertEqual(first_debug["srt_last_request_id"], first.anchor_request_id)
        self.assertEqual(first_debug["srt_last_origin_input_len"], 4)
        self.assertEqual(first_debug["srt_mm_offsets"], [(1, 3)])

        second = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="again")],
            session_id=first.session_id,
        )
        second_debug = runtime.get_debug_counters(second)

        self.assertEqual(second.session_id, first.session_id)
        self.assertEqual(controller.sessions.keys(), {"srt-backed-prefill"})
        self.assertEqual(second_debug["prefill_count"], 2)
        self.assertEqual(second_debug["srt_request_count"], 2)
        self.assertEqual(second_debug["srt_last_request_id"], second.anchor_request_id)
        self.assertEqual(second_debug["srt_last_origin_input_len"], 5)
        self.assertEqual(second_debug["srt_mm_offsets"], [(1, 3)])
        self.assertEqual(len(controller.get(second.session_id).req_nodes), 2)

    def test_prefill_uses_real_tokenizer_when_available(self):
        class EncodedTokenizer:
            bos_token_id = None
            eos_token_id = 9

            def encode(self, text, add_special_tokens=False):
                self.last_encode = (text, add_special_tokens)
                return [11, 12]

        tokenizer = EncodedTokenizer()
        controller = SessionController(FakeTreeCache())
        runtime = UGSessionRuntime(
            model_runner=FakeUGModelRunner(),
            session_controller=controller,
            tokenizer=tokenizer,
            vocab_size=128,
        )

        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="hello world")],
            session_id="encoded-tokenizer",
        )
        debug = runtime.get_debug_counters(handle)

        self.assertEqual(tokenizer.last_encode, ("hello world", False))
        self.assertEqual(debug["srt_last_origin_input_ids"], [9, 11, 12])
        self.assertEqual(debug["srt_last_origin_input_len"], 3)

    def test_append_generated_image_uses_srt_session_offset_shift(self):
        controller = SessionController(FakeTreeCache())
        runtime = UGSessionRuntime(
            model_runner=FakeUGModelRunner(), session_controller=controller
        )

        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="hello world")],
            session_id="srt-backed-image-append",
        )
        self.assertEqual(runtime.get_debug_counters(handle)["srt_mm_offsets"], [])

        decode = runtime.decode_next_segment(handle)
        self.assertEqual(decode.type, "image_marker")
        handle = runtime.append_generated_image(handle, image=object())
        debug = runtime.get_debug_counters(handle)

        self.assertEqual(debug["srt_request_count"], 2)
        self.assertEqual(debug["srt_last_request_id"], handle.anchor_request_id)
        self.assertEqual(debug["srt_last_origin_input_len"], 5)
        self.assertEqual(debug["srt_mm_offsets"], [(3, 5)])
        self.assertEqual(debug["append_image_count"], 1)
        self.assertEqual(debug["state"], "u_decode")

        post_image = runtime.decode_next_segment(handle)
        self.assertEqual(post_image.type, "text")
        self.assertEqual(post_image.text, "generated_text_after_image")

    def test_srt_request_executor_receives_materialized_prefill_and_append_reqs(self):
        executor = RecordingSRTRequestExecutor()
        runtime = UGSessionRuntime(
            model_runner=FakeUGModelRunner(),
            session_controller=SessionController(FakeTreeCache()),
            srt_request_executor=executor,
        )

        handle = runtime.prefill_interleaved(
            [
                UGInterleavedMessage(type="image", content=object()),
                UGInterleavedMessage(type="text", content="hello"),
            ],
            session_id="srt-executor-session",
        )
        runtime.decode_next_segment(handle)
        handle = runtime.append_generated_image(handle, image=object())

        self.assertEqual(
            [event["state"] for event in executor.events],
            ["u_prefill", "append_image"],
        )
        self.assertEqual(
            [event["rid"] for event in executor.events],
            ["srt-executor-session:u1", "srt-executor-session:u2"],
        )
        self.assertEqual(executor.events[0]["mm_offsets"], [(1, 3)])
        self.assertEqual(executor.events[1]["mm_offsets"], [(1, 3), (4, 6)])
        self.assertIsNone(executor.events[0]["finished_reason"])
        self.assertIsNone(executor.events[1]["finished_reason"])

        debug = runtime.get_debug_counters(handle)
        self.assertEqual(debug["srt_executed_request_count"], 2)
        self.assertEqual(
            debug["srt_last_executed_request_id"], handle.anchor_request_id
        )
        self.assertEqual(debug["srt_last_executed_state"], "append_image")

    def test_u_decode_can_materialize_srt_session_decode_request(self):
        executor = RecordingSRTRequestExecutor()
        runtime = UGSessionRuntime(
            model_runner=FakeUGModelRunner(),
            session_controller=SessionController(FakeTreeCache()),
            srt_request_executor=executor,
            srt_u_decode_max_new_tokens=1,
        )

        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="draw a cat")],
            session_id="srt-u-decode-session",
        )
        marker = runtime.decode_next_segment(handle)
        debug = runtime.get_debug_counters(handle)

        self.assertEqual(marker.type, "image_marker")
        self.assertEqual(
            [event["state"] for event in executor.events],
            ["u_prefill", "u_decode"],
        )
        self.assertEqual(
            [event["rid"] for event in executor.events],
            ["srt-u-decode-session:u1", "srt-u-decode-session:d1"],
        )
        self.assertEqual([event["max_new_tokens"] for event in executor.events], [0, 1])
        self.assertEqual(debug["srt_request_count"], 2)
        self.assertEqual(debug["srt_u_decode_request_count"], 1)
        self.assertEqual(
            debug["srt_last_u_decode_request_id"], "srt-u-decode-session:d1"
        )
        self.assertEqual(debug["srt_last_executed_state"], "u_decode")
        self.assertEqual(debug["state"], "g_denoise")

    def test_close_session_releases_srt_multimodal_features(self):
        tree_cache = FakeTreeCache()
        controller = SessionController(tree_cache)
        runtime = UGSessionRuntime(
            model_runner=FakeUGModelRunner(), session_controller=controller
        )

        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="image", content=object())],
            session_id="srt-backed-close",
        )
        self.assertFalse(runtime.get_debug_counters(handle)["srt_mm_features_released"])

        runtime.close_session(handle)
        debug = runtime.get_debug_counters("srt-backed-close")

        self.assertTrue(debug["closed"])
        self.assertEqual(debug["state"], "done")
        self.assertTrue(debug["srt_mm_features_released"])
        self.assertEqual(tree_cache.released_sessions, ["srt-backed-close"])
        self.assertNotIn("srt-backed-close", controller.sessions)

    def test_u_g_u_minimal_loop_keeps_one_session(self):
        runtime = UGSessionRuntime(model_runner=FakeUGModelRunner())
        events = ["input_text", "input_image"]

        handle = runtime.prefill_interleaved(
            [
                UGInterleavedMessage(type="image", content=object()),
                UGInterleavedMessage(type="text", content="draw then explain"),
            ],
            session_id="ug-test-session",
        )
        self.assertEqual(runtime.get_state(handle), UGSegmentState.U_DECODE)

        decode = runtime.decode_next_segment(handle)
        self.assertEqual(decode.type, "image_marker")
        self.assertEqual(runtime.get_state(handle), UGSegmentState.G_DENOISE)

        latents = torch.zeros(1, 4, 8)
        for step in range(3):
            response = runtime.predict_velocity(
                UGVelocityRequest(
                    session=handle,
                    latent_tokens=latents,
                    timestep=torch.tensor([1.0 - step * 0.25]),
                    latent_position_ids=torch.arange(4),
                    sampling_params=None,
                )
            )
            latents = response.velocity

        events.append("generated_image")
        handle_after_image = runtime.append_generated_image(handle, image=object())
        self.assertEqual(handle_after_image.session_id, handle.session_id)
        self.assertGreater(handle_after_image.context_version, handle.context_version)
        self.assertEqual(runtime.get_state(handle_after_image), UGSegmentState.U_DECODE)

        decode_after_image = runtime.decode_next_segment(handle_after_image)
        self.assertEqual(decode_after_image.type, "text")
        self.assertEqual(decode_after_image.text, "generated_text_after_image")
        events.append("generated_text_after_image")

        self.assertEqual(
            events,
            [
                "input_text",
                "input_image",
                "generated_image",
                "generated_text_after_image",
            ],
        )

        counters = runtime.get_debug_counters(handle_after_image)
        self.assertEqual(counters["session_id"], "ug-test-session")
        self.assertEqual(counters["prefill_count"], 1)
        self.assertEqual(counters["velocity_count"], 3)
        self.assertEqual(counters["append_image_count"], 1)

    def test_illegal_transitions_fail_early(self):
        runtime = UGSessionRuntime(model_runner=FakeUGModelRunner())
        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="hello")],
            session_id="illegal",
        )

        with self.assertRaisesRegex(ValueError, "Cannot predict UG velocity"):
            runtime.predict_velocity(
                UGVelocityRequest(
                    session=handle,
                    latent_tokens=torch.zeros(1, 1, 1),
                    timestep=torch.tensor([1.0]),
                    latent_position_ids=torch.arange(1),
                    sampling_params=None,
                )
            )

        with self.assertRaisesRegex(ValueError, "Cannot append generated image"):
            runtime.append_generated_image(handle, image=object())

        runtime.decode_next_segment(handle)
        handle_after_image = runtime.append_generated_image(handle, image=object())
        with self.assertRaisesRegex(ValueError, "Stale UG session handle"):
            runtime.decode_next_segment(handle)

        self.assertEqual(runtime.get_state(handle_after_image), UGSegmentState.U_DECODE)


if __name__ == "__main__":
    unittest.main()
