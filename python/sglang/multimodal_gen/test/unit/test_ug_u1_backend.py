# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch
import unittest

from sglang.multimodal_gen.configs.pipeline_configs.ug import UGPipelineConfig
from sglang.multimodal_gen.runtime.pipelines.ug import (
    _build_ug_g_segment_executor,
    _load_ug_bridge,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ug_bagel import (
    BAGELLatentFlowGSegmentExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ug_u1 import (
    U1PixelFlowGSegmentExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.ug import UGGSegmentStage
from sglang.srt.session.session_controller import SessionController
from sglang.srt.ug.adapter import UGModelRunnerAdapter
from sglang.srt.ug.context import UGContextBundle, UGContextHandle, UGSessionHandle
from sglang.srt.ug.runtime import (
    UGInterleavedMessage,
    UGSegmentState,
    UGSessionRuntime,
)
from sglang.srt.ug.u1 import U1UGModelAdapter


class TestU1UGBackendShell(unittest.TestCase):
    def test_u1_adapter_declares_pixel_flow_kind(self):
        adapter = U1UGModelAdapter()

        self.assertEqual(adapter.g_kind, "pixel_flow")

    def test_u1_shell_does_not_expose_bagel_latent_api(self):
        adapter = U1UGModelAdapter()

        self.assertFalse(hasattr(adapter, "prepare_latents_from_session"))
        self.assertFalse(hasattr(adapter, "predict_velocity_from_session"))
        self.assertFalse(hasattr(adapter, "decode_latents_to_image"))

    def test_unwired_generation_methods_fail_clearly_until_wired(self):
        adapter = U1UGModelAdapter()

        with self.assertRaisesRegex(NotImplementedError, "SenseNova U1"):
            adapter.decode_next_segment(session=None)
        with self.assertRaisesRegex(NotImplementedError, "SenseNova U1"):
            adapter.decode_vlm_text(runtime=None, session=None, max_new_tokens=1)

    def test_u1_text_prepared_input_carries_causal_rows(self):
        adapter = U1UGModelAdapter()

        prepared = adapter.prepare_srt_u_message_inputs(
            session=None,
            message=UGInterleavedMessage(type="text", content="hello U1"),
            state=UGSegmentState.U_PREFILL,
        )

        self.assertEqual(len(prepared), 1)
        metadata = prepared[0].adapter_metadata["u1"]
        self.assertEqual(metadata["segment_type"], "text")
        self.assertEqual(metadata["attention_rows"][0]["attention"], "causal")
        self.assertEqual(prepared[0].mot_text_token_indices, metadata["token_indices"])
        self.assertFalse(prepared[0].non_causal_query_attention)

    def test_u1_image_prepared_input_carries_hybrid_rows(self):
        adapter = U1UGModelAdapter()

        prepared = adapter.prepare_srt_u_message_inputs(
            session=None,
            message=UGInterleavedMessage(type="image", content=object()),
            state=UGSegmentState.U_PREFILL,
        )

        self.assertEqual(len(prepared), 1)
        metadata = prepared[0].adapter_metadata["u1"]
        self.assertEqual(metadata["segment_type"], "image")
        self.assertEqual(metadata["attention_rows"][0]["attention"], "hybrid")
        self.assertEqual(prepared[0].mot_image_token_indices, metadata["token_indices"])
        self.assertTrue(prepared[0].non_causal_query_attention)
        self.assertFalse(metadata["generated_image_commit"])

    def test_u1_generated_image_commit_uses_u_path_metadata(self):
        adapter = U1UGModelAdapter()

        prepared = adapter.prepare_srt_u_message_inputs(
            session=None,
            message=UGInterleavedMessage(type="image", content=object()),
            state=UGSegmentState.APPEND_IMAGE,
        )

        metadata = prepared[0].adapter_metadata["u1"]
        self.assertTrue(metadata["generated_image_commit"])
        self.assertEqual(metadata["source"], "generated_image")

    def test_u1_runtime_prefill_records_u_context_state_without_kv_details(self):
        adapter = U1UGModelAdapter()
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=SessionController(FakeTreeCache()),
        )

        handle = runtime.prefill_interleaved(
            [
                UGInterleavedMessage(type="text", content="describe this"),
                UGInterleavedMessage(type="image", content=object()),
            ],
            session_id="u1-session",
        )

        counters = runtime.get_debug_counters(handle)
        u1_state = counters["ug_model_state"]["u1"]
        self.assertEqual(handle.session_id, "u1-session")
        self.assertEqual(counters["prefill_count"], 1)
        self.assertEqual(u1_state["segments"][0]["segment_type"], "text")
        self.assertEqual(u1_state["segments"][1]["segment_type"], "image")
        self.assertEqual(
            u1_state["segments"][1]["attention_rows"][0]["attention"], "hybrid"
        )
        self.assertFalse(_has_kv_allocator_detail(u1_state))
        self.assertGreaterEqual(len(adapter.observed_u_forwards), 2)
        self.assertFalse(_has_kv_allocator_detail(adapter.observed_u_forwards[-1]))

    def test_u1_runtime_append_generated_image_records_commit_intent(self):
        adapter = U1UGModelAdapter()
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=SessionController(FakeTreeCache()),
        )
        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="draw a cup")],
            session_id="u1-commit-session",
        )
        g_handle = runtime.begin_g_denoise(handle)

        updated = runtime.append_generated_image(g_handle, image=object())

        counters = runtime.get_debug_counters(updated)
        u1_state = counters["ug_model_state"]["u1"]
        self.assertEqual(updated.session_id, handle.session_id)
        self.assertEqual(counters["append_image_count"], 1)
        self.assertTrue(u1_state["segments"][-1]["generated_image_commit"])
        self.assertEqual(u1_state["segments"][-1]["source"], "generated_image")
        self.assertFalse(_has_kv_allocator_detail(u1_state))

    def test_u1_g_executor_declares_pixel_flow_requirement(self):
        executor = U1PixelFlowGSegmentExecutor()

        self.assertEqual(executor.required_g_kind, "pixel_flow")

    def test_u1_g_executor_is_accepted_by_pixel_flow_bridge(self):
        bridge = PixelFlowBridge()
        server_args = _make_ug_server_args()

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.base."
            "get_global_server_args",
            return_value=server_args,
        ):
            UGGSegmentStage(bridge, U1PixelFlowGSegmentExecutor())

    def test_u1_g_executor_rejects_latent_flow_bridge(self):
        bridge = LatentFlowBridge()
        server_args = _make_ug_server_args()

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.base."
            "get_global_server_args",
            return_value=server_args,
        ):
            with self.assertRaisesRegex(ValueError, "pixel_flow"):
                UGGSegmentStage(bridge, U1PixelFlowGSegmentExecutor())

    def test_u1_g_executor_call_fails_clearly_until_wired(self):
        executor = U1PixelFlowGSegmentExecutor()

        with self.assertRaisesRegex(NotImplementedError, "pixel-flow executor"):
            executor(
                bridge=PixelFlowBridge(),
                contexts=_make_contexts(),
                batch=SimpleNamespace(),
                server_args=_make_ug_server_args(),
            )

    def test_pipeline_dispatch_selects_u1_executor_for_pixel_flow(self):
        executor = _build_ug_g_segment_executor(PixelFlowBridge())

        self.assertIsInstance(executor, U1PixelFlowGSegmentExecutor)

    def test_pipeline_dispatch_selects_bagel_executor_for_latent_flow(self):
        executor = _build_ug_g_segment_executor(LatentFlowBridge())

        self.assertIsInstance(executor, BAGELLatentFlowGSegmentExecutor)

    def test_load_u1_bridge_shell_from_model_path(self):
        bridge = _load_ug_bridge("sensenova/SenseNova-U1-8B-MoT")

        self.assertEqual(bridge.g_kind, "pixel_flow")
        self.assertFalse(hasattr(bridge, "prepare_g_latents"))
        with self.assertRaisesRegex(NotImplementedError, "SenseNova U1"):
            bridge.prepare_u_context(prompt="draw a cup", image=None)


class PixelFlowBridge:
    g_kind = "pixel_flow"


class LatentFlowBridge:
    g_kind = "latent_flow"


class FakeTreeCache:
    def __init__(self):
        self.released_sessions = []

    def release_session(self, session_id):
        self.released_sessions.append(session_id)


def _make_contexts():
    session = UGSessionHandle(
        session_id="u1-test-session",
        anchor_request_id="u1-test-session:0",
        context_length=1,
        context_version=1,
    )
    return UGContextBundle(
        full=UGContextHandle("u1-test-session:0", 1, session=session),
        text_cfg=UGContextHandle("u1-test-session:text", 0, session=session),
        image_cfg=UGContextHandle("u1-test-session:image", 0, session=session),
    )


def _make_ug_server_args():
    return SimpleNamespace(
        pipeline_config=UGPipelineConfig(default_height=4, default_width=4),
        num_gpus=1,
        enable_cfg_parallel=False,
        disagg_mode=False,
        comfyui_mode=True,
    )


def _has_kv_allocator_detail(value):
    forbidden = ("allocator", "page", "slot")
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if "kv" in key_text or any(word in key_text for word in forbidden):
                return True
            if _has_kv_allocator_detail(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_has_kv_allocator_detail(item) for item in value)
    return False


if __name__ == "__main__":
    unittest.main()
