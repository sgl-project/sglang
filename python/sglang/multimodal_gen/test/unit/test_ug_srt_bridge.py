# SPDX-License-Identifier: Apache-2.0

import unittest
from dataclasses import fields

import torch
from PIL import Image

from sglang.srt.ug.context import UGSessionHandle
from sglang.srt.ug.denoiser import FakeUGDenoiserBridge, SRTBackedUGDenoiserBridge
from sglang.srt.ug.runtime import UGDecodeResult, UGSessionRuntime


class TestFakeUGDenoiserBridge(unittest.TestCase):
    def test_build_contexts_splits_full_text_and_image_cfg(self):
        bridge = FakeUGDenoiserBridge()
        image = Image.new("RGB", (8, 8), color="white")

        contexts = bridge.build_contexts(prompt="a small cat", image=image)

        self.assertEqual(contexts.full.request_id, "full")
        self.assertEqual(contexts.full.token_count, 5)
        self.assertEqual(contexts.text_cfg.token_count, 2)
        self.assertEqual(contexts.image_cfg.token_count, 3)

    def test_predict_velocity_depends_on_full_context(self):
        bridge = FakeUGDenoiserBridge()
        contexts = bridge.build_contexts(prompt="hello world", image=None)
        latents = torch.zeros(1, 2, 4)
        timestep = torch.tensor([0.5])

        velocity = bridge.predict_velocity(
            contexts=contexts,
            latent_tokens=latents,
            timestep=timestep,
            latent_position_ids=torch.arange(2),
            sampling_params=None,
        )

        self.assertTrue(torch.allclose(velocity, torch.full_like(latents, 0.51)))


class TestSRTBackedUGDenoiserBridge(unittest.TestCase):
    def test_bridge_uses_session_handle_without_kv_details(self):
        bridge = SRTBackedUGDenoiserBridge()
        image = Image.new("RGB", (8, 8), color="white")

        contexts = bridge.build_contexts(prompt="a small cat", image=image)

        self.assertIsInstance(contexts.full.session, UGSessionHandle)
        handle_fields = {field.name for field in fields(contexts.full.session)}
        self.assertFalse(any("kv" in name.lower() for name in handle_fields))
        self.assertFalse(any("slot" in name.lower() for name in handle_fields))
        self.assertFalse(any("page" in name.lower() for name in handle_fields))
        self.assertEqual(contexts.full.token_count, 5)
        self.assertEqual(contexts.text_cfg.token_count, 2)
        self.assertEqual(contexts.image_cfg.token_count, 3)

    def test_bridge_velocity_reuses_one_prefill_context(self):
        bridge = SRTBackedUGDenoiserBridge()
        contexts = bridge.build_contexts(prompt="hello world", image=None)
        latents = torch.zeros(1, 2, 4)

        for i in range(3):
            latents = bridge.predict_velocity(
                contexts=contexts,
                latent_tokens=latents,
                timestep=torch.tensor([1.0 - i * 0.25]),
                latent_position_ids=torch.arange(2),
                sampling_params=None,
            )

        counters = bridge.runtime.get_debug_counters(contexts.full.session)
        self.assertEqual(counters["prefill_count"], 1)
        self.assertEqual(counters["velocity_count"], 3)
        self.assertEqual(counters["decode_count"], 1)
        self.assertEqual(counters["state"], "g_denoise")

    def test_bridge_appends_generated_image_then_returns_to_u_decode(self):
        bridge = SRTBackedUGDenoiserBridge()
        contexts = bridge.build_contexts(prompt="hello world", image=None)
        session_before_image = contexts.full.session

        bridge.predict_velocity(
            contexts=contexts,
            latent_tokens=torch.zeros(1, 2, 4),
            timestep=torch.tensor([1.0]),
            latent_position_ids=torch.arange(2),
            sampling_params=None,
        )
        bridge.append_generated_image(contexts=contexts, image=object())
        post_image_segment = bridge.decode_next_segment(contexts=contexts)

        self.assertEqual(
            contexts.full.session.session_id, session_before_image.session_id
        )
        self.assertGreater(
            contexts.full.session.context_version,
            session_before_image.context_version,
        )
        self.assertEqual(contexts.full.token_count, 4)
        self.assertEqual(post_image_segment.type, "text")
        self.assertEqual(post_image_segment.text, "generated_text_after_image")

        counters = bridge.runtime.get_debug_counters(contexts.full.session)
        self.assertEqual(counters["prefill_count"], 1)
        self.assertEqual(counters["velocity_count"], 1)
        self.assertEqual(counters["append_image_count"], 1)
        self.assertEqual(counters["decode_count"], 2)
        self.assertEqual(counters["state"], "u_decode")

    def test_bridge_bounds_pre_image_decode_loop_and_closes_session(self):
        runner = TextOnlyUGModelRunner()
        bridge = SRTBackedUGDenoiserBridge(
            UGSessionRuntime(model_runner=runner),
            max_pre_image_decode_steps=2,
        )

        with self.assertRaisesRegex(ValueError, "image marker"):
            bridge.build_contexts(prompt="never image", image=None)

        self.assertEqual(runner.decode_count, 2)
        self.assertEqual(len(runner.closed_sessions), 1)


class TextOnlyUGModelRunner:
    def __init__(self):
        self.decode_count = 0
        self.closed_sessions = []

    def prefill_interleaved(self, *, record, messages):
        del record
        return sum(len(str(message.content).split()) for message in messages)

    def decode_next_segment(self, *, record):
        del record
        self.decode_count += 1
        return UGDecodeResult(type="text", text="still text")

    def predict_velocity_from_session(self, *, request, record):
        del request, record
        raise AssertionError("velocity should not be requested before image_marker")

    def prepare_latents_from_session(self, *, request, record):
        del request, record
        return None

    def append_generated_image(self, *, record, image):
        del record, image
        return 0

    def decode_latents_to_image(self, *, request, record):
        del request, record
        return None

    def close_session(self, *, session_id):
        self.closed_sessions.append(session_id)


if __name__ == "__main__":
    unittest.main()
