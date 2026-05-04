# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch
import unittest

from sglang.multimodal_gen.configs.pipeline_configs.ug import UGPipelineConfig
from sglang.multimodal_gen.configs.sample.ug import UGSamplingParams
from sglang.multimodal_gen.runtime.pipelines.ug import (
    _build_ug_g_segment_executor,
    _load_ug_bridge,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ug_bagel import (
    BAGELLatentFlowGSegmentExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ug_u1 import (
    U1PixelFlowGSegmentExecutor,
    _u1_guidance_branch,
    _u1_patch_grid,
    _u1_timesteps,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.ug import (
    UGContextStage,
    UGDecodeStage,
    UGGSegmentStage,
)
from sglang.srt.session.session_controller import SessionController
from sglang.srt.ug.adapter import UGModelRunnerAdapter
from sglang.srt.ug.context import UGContextBundle, UGContextHandle, UGSessionHandle
from sglang.srt.ug.runtime import (
    UGInterleavedMessage,
    UGSegmentState,
    UGSessionRuntime,
)
from sglang.srt.ug.u1 import U1UGModelAdapter, U1VLMBackendResult


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
            adapter.decode_vlm_text(runtime=None, session=None, max_new_tokens=1)

    def test_u1_vlm_backend_generates_real_text_through_adapter(self):
        adapter = U1UGModelAdapter(vlm_backend=FakeU1VLMBackend("real answer"))
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=SessionController(FakeTreeCache()),
        )
        handle = runtime.prefill_interleaved(
            [
                UGInterleavedMessage(type="image", content="/tmp/u1-input.png"),
                UGInterleavedMessage(type="text", content="what is in this image?"),
            ],
            session_id="u1-vlm-session",
        )

        result = adapter.decode_vlm_text(
            runtime=runtime,
            session=handle,
            max_new_tokens=4,
        )

        self.assertEqual(result.text, "real answer")
        self.assertEqual(adapter.vlm_backend.calls[0]["max_new_tokens"], 4)
        self.assertEqual(
            [message.type for message in adapter.vlm_backend.calls[0]["messages"]],
            ["image", "text"],
        )

    def test_u1_bridge_vlm_uses_real_backend_text(self):
        adapter = U1UGModelAdapter(vlm_backend=FakeU1VLMBackend("bridge answer"))
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=SessionController(FakeTreeCache()),
        )
        bridge = _load_u1_bridge(runtime)

        result = bridge.generate_vlm_text(
            messages=[
                UGInterleavedMessage(type="image", content="/tmp/u1-input.png"),
                UGInterleavedMessage(type="text", content="describe"),
            ],
            max_new_tokens=3,
        )

        self.assertEqual(result.text, "bridge answer")
        self.assertEqual(runtime.get_debug_counters(result.session)["prefill_count"], 1)
        bridge.runtime.close_session(result.session)

    def test_u1_vlm_without_external_backend_uses_srt_decode(self):
        adapter = U1UGModelAdapter()
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=SessionController(FakeTreeCache()),
            srt_request_executor=FakeSRTDecodeExecutor([910, 911]),
            tokenizer=FakeTokenizer(),
        )
        bridge = _load_u1_bridge(runtime)

        result = bridge.generate_vlm_text(
            messages=[
                UGInterleavedMessage(type="image", content="/tmp/u1-input.png"),
                UGInterleavedMessage(type="text", content="describe"),
            ],
            max_new_tokens=2,
        )

        counters = runtime.get_debug_counters(result.session)
        self.assertEqual(result.text, "native:910 911")
        self.assertEqual(result.next_token_ids, (910, 911))
        self.assertEqual(counters["prefill_count"], 1)
        self.assertEqual(counters["srt_u_decode_request_count"], 1)
        self.assertEqual(counters["srt_request_count"], 3)
        bridge.runtime.close_session(result.session)

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

    def test_u1_patch_grid_ceil_aligns_to_patch_size(self):
        self.assertEqual(_u1_patch_grid(height=33, width=65, patch_size=16), (3, 5))

    def test_u1_timesteps_use_num_inference_steps(self):
        timesteps = _u1_timesteps(num_inference_steps=4, timestep_shift=2.0)

        self.assertEqual(len(timesteps), 4)
        self.assertGreater(timesteps[0], timesteps[-1])

    def test_u1_guidance_branch_selection(self):
        self.assertEqual(_u1_guidance_branch(_sampling()), "none")
        self.assertEqual(
            _u1_guidance_branch(_sampling(cfg_text_scale=2.0)),
            "text",
        )
        self.assertEqual(
            _u1_guidance_branch(_sampling(cfg_img_scale=2.0)),
            "image",
        )
        self.assertEqual(
            _u1_guidance_branch(_sampling(cfg_text_scale=2.0, cfg_img_scale=2.0)),
            "text_image",
        )

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

    def test_u1_g_executor_returns_image_and_metadata(self):
        executor = U1PixelFlowGSegmentExecutor()
        batch = Req(
            sampling_params=_sampling(
                height=8,
                width=8,
                num_inference_steps=3,
                cfg_text_scale=2.0,
            ),
            prompt="draw a cup",
            seed=7,
        )

        result = executor(
            bridge=PixelFlowBridge(),
            contexts=_make_contexts(),
            batch=batch,
            server_args=_make_ug_server_args(),
        )

        self.assertEqual(result.type, "image")
        self.assertEqual(result.image.size, (8, 8))
        self.assertEqual(result.metadata["g_kind"], "pixel_flow")
        self.assertEqual(result.metadata["grid"], (1, 1))
        self.assertEqual(result.metadata["timesteps"], 3)
        self.assertEqual(result.metadata["guidance"], "text")
        self.assertFalse(result.metadata["temporary_g_kv"])

    def test_u1_common_stages_run_text_image_text_light_smoke(self):
        bridge = _load_ug_bridge("sensenova/SenseNova-U1-8B-MoT")
        server_args = _make_ug_server_args()
        batch = Req(
            sampling_params=_sampling(
                height=8,
                width=8,
                num_inference_steps=2,
            ),
            prompt="draw a cup",
            seed=11,
            extra={"ug_request_metadata": {"mode": "interleave"}},
        )

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.base."
            "get_global_server_args",
            return_value=server_args,
        ):
            batch = UGContextStage(bridge).forward(batch, server_args)
            session = batch.extra["ug_contexts"].full.session
            session_id = session.session_id
            batch = UGGSegmentStage(bridge, U1PixelFlowGSegmentExecutor()).forward(
                batch,
                server_args,
            )
            self.assertEqual(
                bridge.runtime.get_debug_counters(session_id)["append_image_count"],
                0,
            )
            batch = UGDecodeStage(bridge).forward(batch, server_args)

        outputs = batch.extra["ug_output_segments"]
        self.assertEqual([segment["type"] for segment in outputs], ["image", "text"])
        self.assertIn("u1_pixel_flow", outputs[1]["text"])
        self.assertEqual(
            bridge.runtime.get_debug_counters(session_id)["append_image_count"],
            1,
        )
        bridge.release(batch.extra["ug_contexts"])

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
        contexts = bridge.prepare_u_context(prompt="draw a cup", image=None)
        self.assertEqual(contexts.full.session.context_version, 1)
        bridge.release(contexts)

    def test_load_u1_bridge_accepts_scheduler_for_native_srt_decode(self):
        scheduler = FakeScheduler()

        bridge = _load_ug_bridge(
            "sensenova/SenseNova-U1-8B-MoT",
            scheduler=scheduler,
            srt_u_decode_max_new_tokens=2,
        )

        self.assertEqual(bridge.g_kind, "pixel_flow")
        self.assertIs(bridge.runtime.srt_request_executor.scheduler, scheduler)
        self.assertEqual(bridge.runtime.srt_u_decode_max_new_tokens, 2)


class PixelFlowBridge:
    g_kind = "pixel_flow"


class LatentFlowBridge:
    g_kind = "latent_flow"


class FakeTreeCache:
    def __init__(self):
        self.released_sessions = []

    def release_session(self, session_id):
        self.released_sessions.append(session_id)


class FakeSRTDecodeExecutor:
    finish_request_after_execute = True

    def __init__(self, output_ids):
        self.output_ids = list(output_ids)
        self.requests = []

    def execute_ug_request(self, *, record, req, state):
        del record, state
        self.requests.append(req.rid)
        if req.sampling_params.max_new_tokens > 0:
            req.output_ids = self.output_ids[: req.sampling_params.max_new_tokens]


class FakeTokenizer:
    bos_token_id = 1

    def decode(self, token_ids):
        return "native:" + " ".join(str(token_id) for token_id in token_ids)


class FakeScheduler:
    def __init__(self):
        self.session_controller = SessionController(FakeTreeCache())
        self.tokenizer = FakeTokenizer()
        self.model_config = FakeModelConfig()
        self.model_worker = type(
            "Worker",
            (),
            {"model_runner": type("ModelRunner", (), {})()},
        )()


class FakeModelConfig:
    vocab_size = 32000


class FakeU1VLMBackend:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate_text(self, *, messages, max_new_tokens):
        self.calls.append(
            {
                "messages": list(messages),
                "max_new_tokens": max_new_tokens,
            }
        )
        return U1VLMBackendResult(text=self.text)


def _load_u1_bridge(runtime):
    from sglang.srt.ug.u1 import U1SRTBackedUGMiddleBridge

    return U1SRTBackedUGMiddleBridge(runtime)


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
    return FakeServerArgs(
        pipeline_config=UGPipelineConfig(default_height=4, default_width=4),
        num_gpus=1,
        enable_cfg_parallel=False,
        disagg_mode=False,
        comfyui_mode=True,
    )


def _sampling(**kwargs):
    values = {
        "height": 8,
        "width": 8,
        "num_inference_steps": 2,
        "cfg_text_scale": 1.0,
        "cfg_img_scale": 1.0,
    }
    values.update(kwargs)
    return UGSamplingParams(**values)


class FakeServerArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


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
