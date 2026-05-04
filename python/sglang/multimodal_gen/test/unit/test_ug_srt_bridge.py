# SPDX-License-Identifier: Apache-2.0

import unittest
from dataclasses import fields
from pathlib import Path

import torch
from PIL import Image

from sglang.srt.ug.context import UGSessionHandle
from sglang.srt.ug.denoiser import SRTBackedUGMiddleBridge
from sglang.srt.ug.interleaved import UGGSegmentResult
from sglang.srt.ug.runtime import UGDecodeResult, UGSessionRuntime


class TestSRTBackedUGMiddleBridge(unittest.TestCase):
    def test_prepare_u_context_uses_session_handle_without_kv_details(self):
        bridge = _make_bridge()
        image = Image.new("RGB", (8, 8), color="white")

        contexts = bridge.prepare_u_context(prompt="a small cat", image=image)

        self.assertIsInstance(contexts.full.session, UGSessionHandle)
        handle_fields = {field.name for field in fields(contexts.full.session)}
        self.assertFalse(any("kv" in name.lower() for name in handle_fields))
        self.assertFalse(any("slot" in name.lower() for name in handle_fields))
        self.assertFalse(any("page" in name.lower() for name in handle_fields))
        self.assertEqual(contexts.full.token_count, 5)
        self.assertEqual(contexts.text_cfg.token_count, 2)
        self.assertEqual(contexts.image_cfg.token_count, 3)

    def test_g_velocity_reuses_one_prefill_context(self):
        bridge = _make_bridge()
        contexts = bridge.prepare_u_context(prompt="hello world", image=None)
        latents = torch.zeros(1, 2, 4)

        for i in range(3):
            latents = bridge.predict_g_velocity(
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

    def test_commit_generated_segment_then_returns_to_u_decode(self):
        bridge = _make_bridge()
        contexts = bridge.prepare_u_context(prompt="hello world", image=None)
        session_before_image = contexts.full.session

        bridge.predict_g_velocity(
            contexts=contexts,
            latent_tokens=torch.zeros(1, 2, 4),
            timestep=torch.tensor([1.0]),
            latent_position_ids=torch.arange(2),
            sampling_params=None,
        )
        bridge.commit_generated_segment(
            contexts=contexts,
            segment=UGGSegmentResult(type="image", image=object()),
        )
        post_image_segment = bridge.continue_u_decode(contexts=contexts)

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

    def test_run_g_segment_wraps_model_specific_executor(self):
        bridge = _make_bridge()
        contexts = bridge.prepare_u_context(prompt="hello world", image=None)
        image = Image.new("RGB", (8, 8), color="black")

        segment = bridge.run_g_segment(
            contexts=contexts,
            executor=lambda _: UGGSegmentResult(
                type="image",
                image=image,
                metadata={"g_kind": "pixel_flow"},
            ),
        )

        counters = bridge.runtime.get_debug_counters(contexts.full.session)
        self.assertIs(segment.image, image)
        self.assertEqual(segment.metadata["g_kind"], "pixel_flow")
        self.assertEqual(counters["append_image_count"], 0)
        self.assertEqual(counters["state"], "g_denoise")

    def test_bridge_bounds_pre_image_decode_loop_and_closes_session(self):
        runner = TextOnlyUGModelRunner()
        bridge = SRTBackedUGMiddleBridge(
            UGSessionRuntime(model_runner=runner),
            max_pre_image_decode_steps=2,
        )

        with self.assertRaisesRegex(ValueError, "image marker"):
            bridge.prepare_u_context(prompt="never image", image=None)

        self.assertEqual(runner.decode_count, 2)
        self.assertEqual(len(runner.closed_sessions), 1)

    def test_test_only_pixel_flow_backend_runs_text_image_text_protocol(self):
        bridge = TestOnlyPixelFlowBridge()
        contexts = bridge.prepare_u_context(prompt="draw a cup", image=None)

        segment = bridge.run_g_segment(contexts=contexts, executor=None)
        bridge.commit_generated_segment(contexts=contexts, segment=segment)
        text = bridge.continue_u_decode(contexts=contexts)

        self.assertEqual(bridge.g_kind, "pixel_flow")
        self.assertEqual(segment.type, "image")
        self.assertEqual(text.type, "text")
        self.assertEqual(text.text, "pixel_flow_text_after_image")
        self.assertEqual(contexts.full.session.session_id, "pixel-flow-session")

    def test_common_middle_protocol_files_do_not_embed_bagel_g_mechanics(self):
        package_root = Path(__file__).resolve().parents[3]
        common_files = [
            package_root / "srt/ug/context.py",
            package_root / "srt/ug/denoiser.py",
            package_root / "srt/ug/interleaved.py",
            package_root / "srt/ug/runtime.py",
            package_root / "multimodal_gen/runtime/pipelines_core/stages/ug.py",
        ]
        forbidden = ("bagel", "vae", "ug_srt_bagel", "build_bagel")

        for path in common_files:
            text = path.read_text(encoding="utf-8").lower()
            for token in forbidden:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, text)


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


class BridgeUGModelRunner:
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


def _make_bridge():
    return SRTBackedUGMiddleBridge(UGSessionRuntime(model_runner=BridgeUGModelRunner()))


class TestOnlyPixelFlowBridge:
    g_kind = "pixel_flow"

    def __init__(self):
        self.committed = False

    def prepare_u_context(
        self, *, prompt, image, think=False, think_max_new_tokens=None
    ):
        del prompt, image, think, think_max_new_tokens
        session = UGSessionHandle(
            session_id="pixel-flow-session",
            anchor_request_id="pixel-flow-session:u1",
            context_length=3,
            context_version=1,
        )
        from sglang.srt.ug.context import UGContextBundle, UGContextHandle

        return UGContextBundle(
            full=UGContextHandle("pixel-flow-session:u1", 3, session=session),
            text_cfg=UGContextHandle("text-cfg", 0, session=session),
            image_cfg=UGContextHandle("image-cfg", 0, session=session),
        )

    def run_g_segment(self, *, contexts, executor):
        del contexts, executor
        return UGGSegmentResult(
            type="image",
            image=Image.new("RGB", (4, 4), color="blue"),
            metadata={"g_kind": self.g_kind},
        )

    def commit_generated_segment(self, *, contexts, segment):
        self.committed = True
        session = UGSessionHandle(
            session_id=contexts.full.session.session_id,
            anchor_request_id="pixel-flow-session:u2",
            context_length=5,
            context_version=2,
        )
        contexts.full.session = session
        contexts.full.request_id = session.anchor_request_id
        contexts.full.token_count = session.context_length

    def continue_u_decode(self, *, contexts):
        del contexts
        if not self.committed:
            return UGDecodeResult(type="done")
        return UGDecodeResult(type="text", text="pixel_flow_text_after_image")

    def release(self, contexts):
        del contexts


if __name__ == "__main__":
    unittest.main()
