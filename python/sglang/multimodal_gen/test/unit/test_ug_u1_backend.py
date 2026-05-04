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
from sglang.srt.ug.context import UGContextBundle, UGContextHandle, UGSessionHandle
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

    def test_true_weight_adapter_methods_fail_clearly_until_wired(self):
        adapter = U1UGModelAdapter()

        with self.assertRaisesRegex(NotImplementedError, "SenseNova U1"):
            adapter.prefill_interleaved(session=None, messages=[])
        with self.assertRaisesRegex(NotImplementedError, "SenseNova U1"):
            adapter.decode_next_segment(session=None)
        with self.assertRaisesRegex(NotImplementedError, "SenseNova U1"):
            adapter.append_generated_image(session=None, image=None)

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


if __name__ == "__main__":
    unittest.main()
