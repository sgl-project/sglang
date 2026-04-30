# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.ug import UGPipelineConfig
from sglang.multimodal_gen.configs.sample.ug import UGSamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    UGInterleavedGenerateReq,
    build_ug_interleaved_generate_reqs,
    build_ug_vlm_generate_reqs,
    serialize_ug_interleaved_output,
)
from sglang.multimodal_gen.runtime.managers.gpu_worker import (
    GPUWorker,
    _maybe_attach_ug_srt_scheduler,
    _should_attach_ug_srt_scheduler,
)
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines.ug import UGPipeline
from sglang.multimodal_gen.runtime.pipelines_core.executors.sync_executor import (
    SyncExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.srt.session.session_controller import SessionController
from sglang.srt.ug.context import UGContextBundle, UGContextHandle
from sglang.srt.ug.interleaved import (
    UGInputSegment,
    UGInterleavedRequest,
    UGInterleavedResponse,
    UGRuntimeStats,
)
from sglang.srt.ug.runtime import UGDecodeResult, UGLatentPrepareResult
from sglang.srt.ug.sampling import build_bagel_denoise_schedule
from sglang.srt.ug.srt_server import build_bagel_language_model_view

_GLOBAL_ARGS_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args"
)


def _make_server_args() -> SimpleNamespace:
    return SimpleNamespace(
        pipeline_config=UGPipelineConfig(
            default_height=32,
            default_width=32,
            latent_downsample=16,
            latent_patch_size=2,
            latent_channel=16,
        ),
        num_gpus=1,
        enable_cfg_parallel=False,
        disagg_mode=False,
        disagg_role=RoleType.MONOLITHIC,
        comfyui_mode=True,
    )


class TestUGDiffusionPipeline(unittest.TestCase):
    def test_bagel_language_model_view_contains_native_srt_config(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            with tempfile.TemporaryDirectory() as output_dir:
                checkpoint = Path(checkpoint_dir)
                output = Path(output_dir)
                (checkpoint / "llm_config.json").write_text(
                    json.dumps({"model_type": "bagel"}),
                    encoding="utf-8",
                )
                (checkpoint / "ema.safetensors").write_bytes(b"")

                view_path = build_bagel_language_model_view(checkpoint, output)
                config = json.loads((view_path / "config.json").read_text())

                self.assertEqual(
                    config["architectures"],
                    ["BAGELQwen2MoTForCausalLM"],
                )
                self.assertEqual(config["bagel_checkpoint_dir"], str(checkpoint))
                self.assertTrue((view_path / "model.safetensors").is_symlink())

    def test_real_bagel_worker_attaches_native_srt_scheduler(self):
        server_args = SimpleNamespace(
            model_path="/models/BAGEL-7B-MoT",
            model_id=None,
            pipeline_class_name="UGPipeline",
            num_gpus=1,
            enable_cfg_parallel=False,
            ug_srt_mem_fraction_static=0.25,
            ug_srt_chunked_prefill_size=128,
            ug_srt_attention_backend="torch_native",
            ug_srt_log_level="warning",
        )
        handle = SimpleNamespace(scheduler=object())
        calls = []

        def fake_create_bagel_srt_scheduler(**kwargs):
            calls.append(kwargs)
            return handle

        self.assertTrue(_should_attach_ug_srt_scheduler(server_args))
        with patch(
            "sglang.srt.ug.srt_server.create_bagel_srt_scheduler",
            fake_create_bagel_srt_scheduler,
        ):
            attached = _maybe_attach_ug_srt_scheduler(server_args, local_rank=0)

        self.assertIs(attached, handle)
        self.assertIs(server_args.ug_srt_scheduler, handle.scheduler)
        self.assertIs(server_args.ug_srt_scheduler_handle, handle)
        self.assertEqual(
            calls,
            [
                {
                    "checkpoint_dir": "/models/BAGEL-7B-MoT",
                    "gpu_id": 0,
                    "mem_fraction_static": 0.25,
                    "chunked_prefill_size": 128,
                    "attention_backend": "torch_native",
                    "log_level": "warning",
                }
            ],
        )

    def test_mock_bagel_worker_does_not_attach_native_srt_scheduler(self):
        server_args = SimpleNamespace(
            model_path="sglang-internal/mock-bagel",
            model_id=None,
            pipeline_class_name="UGPipeline",
        )

        self.assertFalse(_should_attach_ug_srt_scheduler(server_args))

    def test_fake_pipeline_runs_g_denoise_path(self):
        server_args = _make_server_args()
        with patch(_GLOBAL_ARGS_PATCH, return_value=server_args):
            pipeline = UGPipeline(
                "sglang-internal/fake-ug",
                server_args,
                executor=SyncExecutor(server_args),
            )

        self.assertEqual(
            [stage.__class__.__name__ for stage in pipeline.stages],
            ["UGContextStage", "UGLatentStage", "UGDenoiseStage", "UGDecodeStage"],
        )

        batch = Req(
            sampling_params=UGSamplingParams(
                prompt="text and image",
                width=32,
                height=32,
                seed=123,
                num_inference_steps=4,
                return_trajectory_latents=True,
                suppress_logs=True,
            ),
            condition_image=Image.new("RGB", (16, 16), color="white"),
        )

        result = pipeline.forward(batch, server_args)

        self.assertEqual(result.output.shape, (1, 32, 32, 3))
        self.assertEqual(result.latents.shape, (1, 4, 64))
        self.assertEqual(result.extra["ug_contexts"].full.token_count, 5)
        self.assertIsNotNone(result.extra["ug_contexts"].full.session)
        self.assertEqual(
            [segment["type"] for segment in result.extra["ug_output_segments"]],
            ["image"],
        )
        self.assertEqual(result.trajectory_latents.shape[0], 3)
        self.assertEqual(result.trajectory_timesteps.shape[0], 3)

        bridge = pipeline.get_module("ug_bridge")
        session = result.extra["ug_contexts"].full.session
        counters = bridge.runtime.get_debug_counters(session)
        self.assertEqual(counters["prefill_count"], 1)
        self.assertEqual(counters["velocity_count"], 3)
        self.assertEqual(counters["append_image_count"], 0)
        self.assertEqual(counters["decode_count"], 1)
        self.assertEqual(counters["srt_request_count"], 1)
        self.assertEqual(counters["srt_last_request_id"], session.anchor_request_id)
        self.assertGreater(counters["srt_last_origin_input_len"], 0)
        self.assertEqual(counters["srt_mm_offsets"], [(1, 3)])
        self.assertEqual(counters["srt_executed_request_count"], 1)
        self.assertEqual(
            counters["srt_last_executed_request_id"], session.anchor_request_id
        )
        self.assertEqual(counters["srt_last_executed_state"], "u_prefill")
        self.assertEqual(counters["state"], "g_denoise")

        bridge.release_contexts(result.extra["ug_contexts"])
        closed = bridge.runtime.get_debug_counters(session.session_id)
        self.assertTrue(closed["closed"])
        self.assertEqual(closed["state"], "done")
        self.assertEqual(
            bridge.runtime.session_controller.tree_cache.released_sessions,
            [session.session_id],
        )

    def test_experimental_interleaved_api_runs_fake_ug_pipeline(self):
        server_args = _make_server_args()
        with patch(_GLOBAL_ARGS_PATCH, return_value=server_args):
            pipeline = UGPipeline(
                "sglang-internal/fake-ug",
                server_args,
                executor=SyncExecutor(server_args),
            )

        response = pipeline.forward_interleaved(
            [
                {"type": "image", "image": Image.new("RGB", (16, 16), "white")},
                {"type": "text", "text": "draw a small lamp and describe it"},
            ],
            UGSamplingParams(
                width=32,
                height=32,
                seed=123,
                num_inference_steps=2,
                suppress_logs=True,
            ),
            server_args,
        )
        segments = response.to_legacy_segments()

        self.assertEqual([segment["type"] for segment in segments], ["image", "text"])
        self.assertIsInstance(segments[0]["image"], Image.Image)
        self.assertEqual(segments[0]["image"].size, (32, 32))
        self.assertEqual(segments[1]["text"], "generated_text_after_image")
        self.assertEqual([segment["type"] for segment in response], ["image", "text"])
        self.assertEqual(
            [segment.type for segment in response.segments], ["image", "text"]
        )
        self.assertIsNotNone(response.segments[0].image)
        self.assertEqual(response.stats.prefill_count, 1)
        self.assertEqual(response.stats.velocity_count, 1)
        self.assertEqual(response.stats.append_image_count, 1)
        self.assertEqual(response.stats.decode_count, 2)
        self.assertGreater(response.stats.context_length, 0)
        self.assertGreaterEqual(response.stats.context_version, 2)
        self.assertEqual(response.stats.srt_request_count, 2)
        self.assertEqual(response.stats.srt_executed_request_count, 2)
        self.assertEqual(response.stats.srt_sidecar_request_count, 0)
        self.assertEqual(response.stats.srt_u_decode_request_count, 0)
        bridge = pipeline.get_module("ug_bridge")
        self.assertEqual(
            len(bridge.runtime.session_controller.tree_cache.released_sessions),
            1,
        )

    def test_ug_runtime_stats_serializes_srt_entrypoint_counters(self):
        stats = UGRuntimeStats.from_debug_counters(
            {
                "session_id": "entry-smoke",
                "state": "u_decode",
                "context_length": 11,
                "context_version": 3,
                "prefill_count": 1,
                "velocity_count": 2,
                "append_image_count": 1,
                "decode_count": 2,
                "srt_request_count": 5,
                "srt_executed_request_count": 5,
                "srt_sidecar_request_count": 1,
                "srt_u_decode_request_count": 2,
            }
        )
        response = UGInterleavedResponse.from_legacy_segments(
            [{"type": "text", "text": "ok"}],
            stats=stats,
        )

        serialized = serialize_ug_interleaved_output(response)

        self.assertEqual(serialized["stats"]["context_length"], 11)
        self.assertEqual(serialized["stats"]["context_version"], 3)
        self.assertEqual(serialized["stats"]["srt_sidecar_request_count"], 1)
        self.assertEqual(serialized["stats"]["srt_u_decode_request_count"], 2)

    def test_experimental_vlm_api_runs_fake_ug_text_only(self):
        server_args = _make_server_args()
        with patch(_GLOBAL_ARGS_PATCH, return_value=server_args):
            pipeline = UGPipeline(
                "sglang-internal/fake-ug",
                server_args,
                executor=SyncExecutor(server_args),
            )

        response = pipeline.forward_vlm(
            [
                {
                    "type": "image",
                    "image": {
                        "image": Image.new("RGB", (16, 16), "white"),
                        "vae": False,
                        "vit": True,
                    },
                },
                {"type": "text", "text": "describe this image"},
            ],
            server_args=server_args,
            max_new_tokens=3,
        )
        segments = response.to_legacy_segments()

        self.assertEqual([segment["type"] for segment in segments], ["text"])
        self.assertEqual(segments[0]["text"], "generated_text")
        self.assertEqual(segments[0]["metadata"]["token_ids"], [1, 2, 3])
        self.assertEqual(response.metadata["mode"], "vlm")
        self.assertEqual(response.stats.prefill_count, 1)
        self.assertEqual(response.stats.velocity_count, 0)
        self.assertEqual(response.stats.append_image_count, 0)
        bridge = pipeline.get_module("ug_bridge")
        self.assertEqual(
            len(bridge.runtime.session_controller.tree_cache.released_sessions),
            1,
        )

    def test_build_ug_vlm_request_sets_metadata_mode(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
            Image.new("RGB", (4, 4), "white").save(image_file.name)
            req = build_ug_vlm_generate_reqs(
                {
                    "messages": [
                        {"type": "image", "image": image_file.name},
                        {"type": "text", "text": "describe"},
                    ],
                    "sampling_params": {
                        "width": 32,
                        "height": 32,
                        "max_new_tokens": 4,
                    },
                }
            )

        self.assertIsInstance(req, UGInterleavedGenerateReq)
        self.assertEqual(req.request.metadata["mode"], "vlm")
        self.assertEqual(req.request.metadata["max_new_tokens"], 4)
        self.assertIsInstance(req.request.messages[0].image, Image.Image)

    def test_experimental_interleaved_api_can_return_text_image_text(self):
        server_args = _make_server_args()
        bridge = RecordingUGBridge(
            pre_image_segments=[{"type": "text", "text": "text_before_image"}]
        )
        with patch(_GLOBAL_ARGS_PATCH, return_value=server_args):
            pipeline = UGPipeline(
                "recording-ug",
                server_args,
                loaded_modules={"ug_bridge": bridge},
                executor=SyncExecutor(server_args),
            )

        response = pipeline.forward_interleaved(
            [
                {"type": "image", "image": Image.new("RGB", (16, 16), "white")},
                {"type": "text", "text": "draw then explain"},
            ],
            UGSamplingParams(
                width=32,
                height=32,
                seed=123,
                num_inference_steps=2,
                suppress_logs=True,
            ),
            server_args,
        )
        segments = response.to_segments()

        self.assertEqual(
            [segment["type"] for segment in segments], ["text", "image", "text"]
        )
        self.assertEqual(segments[0]["text"], "text_before_image")
        self.assertIsInstance(segments[1]["image"], Image.Image)
        self.assertEqual(segments[2]["text"], "after_image")
        self.assertEqual(
            [message.type for message in bridge.interleaved_messages],
            ["image", "text"],
        )
        self.assertIsNone(response.stats)

    def test_experimental_interleaved_api_accepts_dict_sampling_params(self):
        server_args = _make_server_args()
        with patch(_GLOBAL_ARGS_PATCH, return_value=server_args):
            pipeline = UGPipeline(
                "sglang-internal/fake-ug",
                server_args,
                executor=SyncExecutor(server_args),
            )

        response = pipeline.forward_interleaved(
            [
                {"type": "image", "image": Image.new("RGB", (16, 16), "white")},
                {"type": "text", "text": "draw a small lamp and describe it"},
            ],
            {
                "width": 32,
                "height": 32,
                "seed": 123,
                "num_inference_steps": 2,
                "suppress_logs": True,
            },
        )
        segments = response.to_legacy_segments()

        self.assertEqual([segment["type"] for segment in segments], ["image", "text"])
        self.assertIsInstance(segments[0]["image"], Image.Image)
        self.assertEqual(segments[0]["image"].size, (32, 32))
        self.assertEqual(segments[1]["text"], "generated_text_after_image")

    def test_experimental_interleaved_api_accepts_request_schema(self):
        server_args = _make_server_args()
        with patch(_GLOBAL_ARGS_PATCH, return_value=server_args):
            pipeline = UGPipeline(
                "sglang-internal/fake-ug",
                server_args,
                executor=SyncExecutor(server_args),
            )

        request = UGInterleavedRequest.from_segments(
            [
                UGInputSegment.from_image(Image.new("RGB", (16, 16), "white")),
                UGInputSegment.from_text("draw a small lamp and describe it"),
            ],
            sampling_params={
                "width": 32,
                "height": 32,
                "seed": 123,
                "num_inference_steps": 2,
                "suppress_logs": True,
            },
            metadata={"request_id": "schema-test"},
        )
        response = pipeline.forward_interleaved(request)

        self.assertEqual(
            response.metadata, {"request_id": "schema-test", "mode": "interleave"}
        )
        self.assertEqual(
            [segment.type for segment in response.segments], ["image", "text"]
        )
        self.assertEqual(response.stats.prefill_count, 1)
        self.assertEqual(response.stats.velocity_count, 1)

    def test_experimental_interleaved_api_isolates_two_sessions(self):
        server_args = _make_server_args()
        with patch(_GLOBAL_ARGS_PATCH, return_value=server_args):
            pipeline = UGPipeline(
                "sglang-internal/fake-ug",
                server_args,
                executor=SyncExecutor(server_args),
            )

        first = pipeline.forward_interleaved(
            [
                {"type": "image", "image": Image.new("RGB", (16, 16), "white")},
                {"type": "text", "text": "draw a red kite"},
            ],
            {
                "width": 32,
                "height": 32,
                "seed": 123,
                "num_inference_steps": 2,
                "suppress_logs": True,
            },
        )
        second = pipeline.forward_interleaved(
            [
                {"type": "image", "image": Image.new("RGB", (16, 16), "black")},
                {"type": "text", "text": "draw a blue boat"},
            ],
            {
                "width": 32,
                "height": 32,
                "seed": 456,
                "num_inference_steps": 2,
                "suppress_logs": True,
            },
        )

        self.assertNotEqual(first.stats.session_id, second.stats.session_id)
        self.assertEqual(first.stats.prefill_count, 1)
        self.assertEqual(second.stats.prefill_count, 1)
        self.assertEqual(first.stats.velocity_count, 1)
        self.assertEqual(second.stats.velocity_count, 1)
        bridge = pipeline.get_module("ug_bridge")
        self.assertEqual(
            bridge.runtime.session_controller.tree_cache.released_sessions,
            [first.stats.session_id, second.stats.session_id],
        )

    def test_experimental_interleaved_batch_api_isolates_sessions(self):
        server_args = _make_server_args()
        with patch(_GLOBAL_ARGS_PATCH, return_value=server_args):
            pipeline = UGPipeline(
                "sglang-internal/fake-ug",
                server_args,
                executor=SyncExecutor(server_args),
            )

        requests = [
            UGInterleavedRequest.from_segments(
                [
                    {"type": "image", "image": Image.new("RGB", (16, 16), color)},
                    {"type": "text", "text": text},
                ],
                sampling_params={
                    "width": 32,
                    "height": 32,
                    "seed": seed,
                    "num_inference_steps": 2,
                    "suppress_logs": True,
                },
            )
            for color, text, seed in (
                ("white", "draw a red kite", 123),
                ("black", "draw a blue boat", 456),
            )
        ]

        responses = pipeline.forward_interleaved_batch(requests, server_args)

        self.assertEqual(len(responses), 2)
        self.assertNotEqual(
            responses[0].stats.session_id, responses[1].stats.session_id
        )
        self.assertEqual(
            [response.stats.prefill_count for response in responses], [1, 1]
        )
        self.assertEqual(
            [response.stats.velocity_count for response in responses], [1, 1]
        )

    def test_ug_interleaved_worker_executes_batched_transport_requests(self):
        server_args = _make_server_args()
        with patch(_GLOBAL_ARGS_PATCH, return_value=server_args):
            pipeline = UGPipeline(
                "sglang-internal/fake-ug",
                server_args,
                executor=SyncExecutor(server_args),
            )
        worker = GPUWorker.__new__(GPUWorker)
        worker.pipeline = pipeline
        worker.server_args = server_args

        reqs = build_ug_interleaved_generate_reqs(
            {
                "requests": [
                    {
                        "messages": [
                            {
                                "type": "image",
                                "image": Image.new("RGB", (16, 16), "white"),
                            },
                            {"type": "text", "text": "draw a red kite"},
                        ],
                        "sampling_params": {
                            "width": 32,
                            "height": 32,
                            "seed": 123,
                            "num_inference_steps": 2,
                            "suppress_logs": True,
                        },
                    },
                    {
                        "messages": [
                            {
                                "type": "image",
                                "image": Image.new("RGB", (16, 16), "black"),
                            },
                            {"type": "text", "text": "draw a blue boat"},
                        ],
                        "sampling_params": {
                            "width": 32,
                            "height": 32,
                            "seed": 456,
                            "num_inference_steps": 2,
                            "suppress_logs": True,
                        },
                    },
                ]
            }
        )

        output = worker.execute_ug_interleaved(reqs)

        self.assertIsNone(output.error)
        self.assertEqual(len(output.output), 2)
        self.assertIsInstance(
            serialize_ug_interleaved_output(output.output[0])["segments"][0]["image"],
            str,
        )
        self.assertNotEqual(
            output.output[0].stats.session_id, output.output[1].stats.session_id
        )

    def test_worker_routes_vlm_mode_to_vlm_pipeline(self):
        worker = GPUWorker.__new__(GPUWorker)
        worker.pipeline = RecordingUGPipeline()
        worker.server_args = _make_server_args()
        reqs = [
            UGInterleavedGenerateReq(
                UGInterleavedRequest.from_segments(
                    [{"type": "text", "text": "describe"}],
                    metadata={"mode": "vlm"},
                )
            )
        ]

        output = worker.execute_ug_interleaved(reqs)

        self.assertIsNone(output.error)
        self.assertEqual(output.output, "vlm-ok")
        self.assertEqual(worker.pipeline.vlm_batch, [reqs[0].request])
        self.assertIsNone(worker.pipeline.interleaved_batch)

    def test_scheduler_routes_ug_interleaved_list_to_worker(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.worker = RecordingUGWorker()
        reqs = [
            UGInterleavedGenerateReq(
                UGInterleavedRequest.from_segments(
                    [{"type": "text", "text": "draw"}],
                    sampling_params={
                        "width": 32,
                        "height": 32,
                        "num_inference_steps": 2,
                    },
                )
            ),
            UGInterleavedGenerateReq(
                UGInterleavedRequest.from_segments(
                    [{"type": "text", "text": "describe"}],
                    sampling_params={
                        "width": 32,
                        "height": 32,
                        "num_inference_steps": 2,
                    },
                )
            ),
        ]

        output = scheduler._handle_list_request([reqs])

        self.assertIsNone(output.error)
        self.assertEqual(scheduler.worker.executed, reqs)
        self.assertEqual(output.output, ["ug-ok"])

    def test_experimental_interleaved_api_rejects_kwargs_with_params_object(self):
        server_args = _make_server_args()
        with patch(_GLOBAL_ARGS_PATCH, return_value=server_args):
            pipeline = UGPipeline(
                "sglang-internal/fake-ug",
                server_args,
                executor=SyncExecutor(server_args),
            )

        with self.assertRaisesRegex(ValueError, "keyword overrides"):
            pipeline.forward_interleaved(
                [{"type": "text", "text": "draw"}],
                UGSamplingParams(
                    width=32,
                    height=32,
                    num_inference_steps=2,
                    suppress_logs=True,
                ),
                width=64,
            )

    def test_experimental_interleaved_api_rejects_sampling_overrides_on_request(self):
        server_args = _make_server_args()
        with patch(_GLOBAL_ARGS_PATCH, return_value=server_args):
            pipeline = UGPipeline(
                "sglang-internal/fake-ug",
                server_args,
                executor=SyncExecutor(server_args),
            )

        request = UGInterleavedRequest.from_segments(
            [{"type": "text", "text": "draw"}],
            sampling_params=UGSamplingParams(
                width=32,
                height=32,
                num_inference_steps=2,
                suppress_logs=True,
            ),
        )
        with self.assertRaisesRegex(ValueError, "already contains sampling_params"):
            pipeline.forward_interleaved(request, width=64)

    def test_fake_pipeline_can_use_injected_srt_scheduler_executor(self):
        server_args = _make_server_args()
        scheduler = RecordingSRTScheduler()
        server_args.ug_srt_scheduler = scheduler
        with patch(_GLOBAL_ARGS_PATCH, return_value=server_args):
            pipeline = UGPipeline(
                "sglang-internal/fake-ug",
                server_args,
                executor=SyncExecutor(server_args),
            )

        result = pipeline.forward(
            Req(
                sampling_params=UGSamplingParams(
                    prompt="scheduler executor",
                    width=32,
                    height=32,
                    seed=123,
                    num_inference_steps=2,
                    suppress_logs=True,
                )
            ),
            server_args,
        )

        bridge = pipeline.get_module("ug_bridge")
        session = result.extra["ug_contexts"].full.session
        self.assertIs(bridge.runtime.session_controller, scheduler.session_controller)
        self.assertEqual(
            [req.rid for req in scheduler.enqueued_requests],
            [
                f"{session.session_id}:u1",
                f"{session.session_id}:d1",
            ],
        )
        self.assertEqual(
            scheduler.finished_reason_at_enqueue,
            [None, None],
        )
        self.assertTrue(all(req.finished() for req in scheduler.enqueued_requests))
        self.assertEqual(
            [req.sampling_params.max_new_tokens for req in scheduler.enqueued_requests],
            [0, 1],
        )
        self.assertEqual(scheduler.init_req_max_new_tokens_calls, 2)
        self.assertEqual(scheduler.run_batch_calls, 2)
        self.assertEqual(scheduler.process_batch_result_calls, 2)
        self.assertEqual(scheduler.pending_queue, [])
        self.assertIsNone(scheduler.last_batch)
        self.assertEqual(
            bridge.runtime.srt_request_executor.events,
            [
                ("u_prefill", f"{session.session_id}:u1", 3),
                ("u_decode", f"{session.session_id}:d1", 3),
            ],
        )
        self.assertEqual(bridge.runtime.srt_request_executor.sync_step_count, 2)
        counters = bridge.runtime.get_debug_counters(session)
        self.assertEqual(counters["srt_executed_request_count"], 2)
        self.assertEqual(counters["srt_u_decode_request_count"], 1)
        self.assertEqual(counters["srt_last_executed_state"], "u_decode")

    def test_runtime_guard_rejects_cfg_parallel(self):
        server_args = _make_server_args()
        server_args.enable_cfg_parallel = True
        with patch(_GLOBAL_ARGS_PATCH, return_value=server_args):
            pipeline = UGPipeline(
                "sglang-internal/fake-ug",
                server_args,
                executor=SyncExecutor(server_args),
            )

        batch = Req(
            sampling_params=UGSamplingParams(
                prompt="guard",
                width=32,
                height=32,
                num_inference_steps=2,
                suppress_logs=True,
            )
        )

        with self.assertRaisesRegex(ValueError, "enable_cfg_parallel"):
            pipeline.forward(batch, server_args)

    def test_interleave_decode_stage_appends_single_pil_image_to_ug_session(self):
        server_args = _make_server_args()
        bridge = RecordingUGBridge()
        with patch(_GLOBAL_ARGS_PATCH, return_value=server_args):
            pipeline = UGPipeline(
                "recording-ug",
                server_args,
                loaded_modules={"ug_bridge": bridge},
                executor=SyncExecutor(server_args),
            )

        response = pipeline.forward_interleaved(
            [{"type": "text", "text": "draw then explain"}],
            UGSamplingParams(
                width=32,
                height=32,
                seed=123,
                num_inference_steps=2,
                suppress_logs=True,
            ),
            server_args,
        )
        segments = response.to_segments()

        self.assertEqual(bridge.velocity_calls, 1)
        self.assertIsInstance(bridge.appended_image, Image.Image)
        self.assertEqual(bridge.appended_image.size, (32, 32))
        self.assertEqual([segment["type"] for segment in segments], ["image", "text"])
        self.assertEqual(segments[1]["text"], "after_image")

    def test_t2i_mode_does_not_append_generated_image(self):
        server_args = _make_server_args()
        bridge = RecordingUGBridge()
        with patch(_GLOBAL_ARGS_PATCH, return_value=server_args):
            pipeline = UGPipeline(
                "recording-ug",
                server_args,
                loaded_modules={"ug_bridge": bridge},
                executor=SyncExecutor(server_args),
            )

        result = pipeline.forward(
            Req(
                sampling_params=UGSamplingParams(
                    prompt="draw without interleave",
                    width=32,
                    height=32,
                    seed=123,
                    num_inference_steps=2,
                    suppress_logs=True,
                )
            ),
            server_args,
        )

        self.assertEqual(bridge.velocity_calls, 1)
        self.assertIsNone(bridge.appended_image)
        self.assertEqual(
            [segment["type"] for segment in result.extra["ug_output_segments"]],
            ["image"],
        )

    def test_think_is_t2i_sampling_switch_not_separate_mode(self):
        server_args = _make_server_args()
        bridge = RecordingUGBridge()
        with patch(_GLOBAL_ARGS_PATCH, return_value=server_args):
            pipeline = UGPipeline(
                "recording-ug",
                server_args,
                loaded_modules={"ug_bridge": bridge},
                executor=SyncExecutor(server_args),
            )

        result = pipeline.forward(
            Req(
                sampling_params=UGSamplingParams(
                    prompt="think before drawing",
                    width=32,
                    height=32,
                    seed=123,
                    num_inference_steps=2,
                    think=True,
                    think_max_new_tokens=5,
                    suppress_logs=True,
                )
            ),
            server_args,
        )

        self.assertEqual(result.extra["ug_mode"], "t2i")
        self.assertTrue(bridge.think)
        self.assertEqual(bridge.think_max_new_tokens, 5)
        self.assertIsNone(bridge.appended_image)

    def test_latent_stage_prefers_bridge_supplied_model_latents(self):
        server_args = _make_server_args()
        bridge = RecordingUGBridge(
            prepared_latents=UGLatentPrepareResult(
                latent_tokens=torch.ones(4, 64),
                latent_position_ids=torch.arange(4),
                latent_shape=(2, 2, 64),
            )
        )
        with patch(_GLOBAL_ARGS_PATCH, return_value=server_args):
            pipeline = UGPipeline(
                "recording-ug",
                server_args,
                loaded_modules={"ug_bridge": bridge},
                executor=SyncExecutor(server_args),
            )

        result = pipeline.forward(
            Req(
                sampling_params=UGSamplingParams(
                    prompt="model-shaped latents",
                    width=32,
                    height=32,
                    seed=321,
                    num_inference_steps=2,
                    suppress_logs=True,
                )
            ),
            server_args,
        )

        self.assertEqual(bridge.prepare_latents_seed, 321)
        self.assertEqual(result.latents.shape, (4, 64))
        self.assertTrue(
            torch.equal(result.extra["ug_latent_position_ids"], torch.arange(4))
        )
        self.assertEqual(result.extra["ug_latent_shape"], (2, 2, 64))

    def test_denoise_stage_uses_bagel_official_shifted_schedule(self):
        server_args = _make_server_args()
        bridge = RecordingUGBridge(
            prepared_latents=UGLatentPrepareResult(
                latent_tokens=torch.ones(4, 64, dtype=torch.float64),
                latent_position_ids=torch.arange(4),
                latent_shape=(2, 2, 64),
            )
        )
        with patch(_GLOBAL_ARGS_PATCH, return_value=server_args):
            pipeline = UGPipeline(
                "recording-ug",
                server_args,
                loaded_modules={"ug_bridge": bridge},
                executor=SyncExecutor(server_args),
            )

        params = UGSamplingParams(
            prompt="official schedule",
            width=32,
            height=32,
            seed=321,
            num_inference_steps=4,
            timestep_shift=3.0,
            return_trajectory_latents=True,
            suppress_logs=True,
        )
        result = pipeline.forward(Req(sampling_params=params), server_args)

        expected = build_bagel_denoise_schedule(
            num_inference_steps=4,
            timestep_shift=3.0,
        )
        observed = torch.stack(
            [timestep.flatten()[0].cpu() for timestep in bridge.velocity_timesteps]
        )
        self.assertEqual(observed.dtype, expected.timesteps.dtype)
        self.assertTrue(torch.allclose(observed, expected.timesteps))
        self.assertTrue(torch.allclose(result.trajectory_timesteps, expected.timesteps))

    def test_ug_sampling_params_rejects_zero_timestep_shift(self):
        with self.assertRaisesRegex(ValueError, "timestep_shift must be positive"):
            UGSamplingParams(
                prompt="bad schedule",
                timestep_shift=0.0,
                suppress_logs=True,
            )


class RecordingUGBridge:
    def __init__(self, prepared_latents=None, pre_image_segments=None):
        self.prepared_latents = prepared_latents
        self.pre_image_segments = pre_image_segments or []
        self.prepare_latents_seed = None
        self.think = None
        self.think_max_new_tokens = None
        self.appended_image = None
        self.velocity_calls = 0
        self.velocity_timesteps = []
        self.velocity_sampling_params = []
        self.interleaved_messages = []

    def build_contexts(self, *, prompt, image, think=False, think_max_new_tokens=None):
        del prompt, image
        self.think = think
        self.think_max_new_tokens = think_max_new_tokens
        return self._make_contexts()

    def build_contexts_from_messages(
        self, *, messages, think=False, think_max_new_tokens=None
    ):
        self.interleaved_messages = list(messages)
        self.think = think
        self.think_max_new_tokens = think_max_new_tokens
        return self._make_contexts()

    def _make_contexts(self):
        return UGContextBundle(
            full=UGContextHandle(
                "full",
                1,
                metadata={"pre_image_segments": list(self.pre_image_segments)},
            ),
            text_cfg=UGContextHandle("text_cfg", 0),
            image_cfg=UGContextHandle("image_cfg", 1),
        )

    def predict_velocity(
        self,
        *,
        contexts,
        latent_tokens,
        timestep,
        latent_position_ids,
        sampling_params,
    ):
        del contexts, latent_position_ids
        self.velocity_calls += 1
        self.velocity_timesteps.append(timestep.detach().clone())
        self.velocity_sampling_params.append(sampling_params)
        return torch.zeros_like(latent_tokens)

    def release_contexts(self, contexts):
        del contexts

    def prepare_latents(self, *, contexts, sampling_params, seed):
        del contexts, sampling_params
        self.prepare_latents_seed = seed
        return self.prepared_latents

    def append_generated_image(self, *, contexts, image):
        del contexts
        self.appended_image = image

    def decode_latents(self, *, contexts, latent_tokens, sampling_params):
        del contexts, latent_tokens, sampling_params
        return None

    def decode_next_segment(self, *, contexts):
        del contexts
        return UGDecodeResult(type="text", text="after_image")


class RecordingUGWorker:
    def __init__(self):
        self.executed = None

    def execute_ug_interleaved(self, reqs):
        self.executed = reqs
        return OutputBatch(output="ug-ok")


class RecordingUGPipeline:
    def __init__(self):
        self.vlm_batch = None
        self.interleaved_batch = None

    def forward_vlm_batch(self, requests, server_args=None):
        del server_args
        self.vlm_batch = list(requests)
        return ["vlm-ok" for _ in requests]

    def forward_interleaved_batch(self, requests, server_args=None):
        del server_args
        self.interleaved_batch = list(requests)
        return ["interleaved-ok" for _ in requests]


class FakeTreeCache:
    def __init__(self):
        self.released_sessions = []

    def release_session(self, session_id):
        self.released_sessions.append(session_id)


class RecordingSRTScheduler:
    def __init__(self):
        self.session_controller = SessionController(FakeTreeCache())
        self.enqueued_requests = []
        self.pending_queue = []
        self.finished_reason_at_enqueue = []
        self.init_req_max_new_tokens_calls = 0
        self.run_batch_calls = 0
        self.process_batch_result_calls = 0
        self.idle_calls = 0
        self.cur_batch = None
        self.last_batch = None

    def init_req_max_new_tokens(self, req):
        self.init_req_max_new_tokens_calls += 1
        req.ug_init_req_max_new_tokens_seen = True

    def _add_request_to_queue(self, req):
        self.finished_reason_at_enqueue.append(req.finished_reason)
        self.enqueued_requests.append(req)
        self.pending_queue.append(req)

    def is_fully_idle(self):
        return not self.pending_queue and self.last_batch is None

    def get_next_batch_to_run(self):
        if self.last_batch is not None:
            self.last_batch = None
            return None
        if not self.pending_queue:
            return None
        return RecordingSRTBatch(self.pending_queue.pop(0))

    def run_batch(self, batch):
        self.run_batch_calls += 1
        self.assert_unfinished_batch(batch)
        return RecordingSRTBatchResult()

    def process_batch_result(self, batch, result):
        del result
        self.process_batch_result_calls += 1
        for req in batch.reqs:
            if req.sampling_params.max_new_tokens > 0:
                req.output_ids.append(1000 + len(req.output_ids))
            req.check_finished(new_accepted_len=0)

    def on_idle(self):
        self.idle_calls += 1

    @staticmethod
    def assert_unfinished_batch(batch):
        for req in batch.reqs:
            if req.finished():
                raise AssertionError(f"Request {req.rid} finished before run_batch")


class RecordingSRTBatch:
    def __init__(self, req):
        self.reqs = [req]

    def __bool__(self):
        return True


class RecordingSRTBatchResult:
    pass


if __name__ == "__main__":
    unittest.main()
