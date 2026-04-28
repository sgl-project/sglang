# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.ug import UGPipelineConfig
from sglang.multimodal_gen.configs.sample.ug import UGSamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines.ug import UGPipeline
from sglang.multimodal_gen.runtime.pipelines_core.executors.sync_executor import (
    SyncExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.srt.session.session_controller import SessionController
from sglang.srt.ug.context import UGContextBundle, UGContextHandle
from sglang.srt.ug.runtime import UGDecodeResult, UGLatentPrepareResult

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
        self.assertEqual(result.extra["ug_contexts"].full.token_count, 7)
        self.assertIsNotNone(result.extra["ug_contexts"].full.session)
        self.assertEqual(result.extra["ug_post_image_segment"].type, "text")
        self.assertEqual(
            result.extra["ug_post_image_segment"].text,
            "generated_text_after_image",
        )
        self.assertEqual(result.trajectory_latents.shape[0], 3)
        self.assertEqual(result.trajectory_timesteps.shape[0], 3)

        bridge = pipeline.get_module("ug_bridge")
        session = result.extra["ug_contexts"].full.session
        counters = bridge.runtime.get_debug_counters(session)
        self.assertEqual(counters["prefill_count"], 1)
        self.assertEqual(counters["velocity_count"], 3)
        self.assertEqual(counters["append_image_count"], 1)
        self.assertEqual(counters["decode_count"], 2)
        self.assertEqual(counters["srt_request_count"], 2)
        self.assertEqual(counters["srt_last_request_id"], session.anchor_request_id)
        self.assertEqual(counters["srt_last_origin_input_len"], 8)
        self.assertEqual(counters["srt_mm_offsets"], [(1, 3), (6, 8)])
        self.assertEqual(counters["srt_executed_request_count"], 2)
        self.assertEqual(
            counters["srt_last_executed_request_id"], session.anchor_request_id
        )
        self.assertEqual(counters["srt_last_executed_state"], "append_image")
        self.assertEqual(counters["state"], "u_decode")

        bridge.release_contexts(result.extra["ug_contexts"])
        closed = bridge.runtime.get_debug_counters(session.session_id)
        self.assertTrue(closed["closed"])
        self.assertEqual(closed["state"], "done")
        self.assertEqual(
            bridge.runtime.session_controller.tree_cache.released_sessions,
            [session.session_id],
        )

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
            [f"{session.session_id}:u1", f"{session.session_id}:u2"],
        )
        self.assertEqual(
            scheduler.finished_reason_at_enqueue,
            [None, None],
        )
        self.assertTrue(all(req.finished() for req in scheduler.enqueued_requests))
        self.assertEqual(scheduler.init_req_max_new_tokens_calls, 2)
        self.assertEqual(scheduler.run_batch_calls, 2)
        self.assertEqual(scheduler.process_batch_result_calls, 2)
        self.assertEqual(scheduler.pending_queue, [])
        self.assertIsNone(scheduler.last_batch)
        self.assertEqual(
            bridge.runtime.srt_request_executor.events,
            [
                ("u_prefill", f"{session.session_id}:u1", 3),
                ("append_image", f"{session.session_id}:u2", 5),
            ],
        )
        self.assertEqual(bridge.runtime.srt_request_executor.sync_step_count, 2)
        counters = bridge.runtime.get_debug_counters(session)
        self.assertEqual(counters["srt_executed_request_count"], 2)
        self.assertEqual(counters["srt_last_executed_state"], "append_image")

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

    def test_decode_stage_appends_single_pil_image_to_ug_session(self):
        server_args = _make_server_args()
        bridge = RecordingUGBridge()
        with patch(_GLOBAL_ARGS_PATCH, return_value=server_args):
            pipeline = UGPipeline(
                "recording-ug",
                server_args,
                loaded_modules={"ug_bridge": bridge},
                executor=SyncExecutor(server_args),
            )

        batch = Req(
            sampling_params=UGSamplingParams(
                prompt="draw then explain",
                width=32,
                height=32,
                seed=123,
                num_inference_steps=2,
                suppress_logs=True,
            )
        )

        result = pipeline.forward(batch, server_args)

        self.assertEqual(bridge.velocity_calls, 1)
        self.assertIsInstance(bridge.appended_image, Image.Image)
        self.assertEqual(bridge.appended_image.size, (32, 32))
        self.assertEqual(result.extra["ug_post_image_segment"].type, "text")
        self.assertEqual(result.extra["ug_post_image_segment"].text, "after_image")

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


class RecordingUGBridge:
    def __init__(self, prepared_latents=None):
        self.prepared_latents = prepared_latents
        self.prepare_latents_seed = None
        self.appended_image = None
        self.velocity_calls = 0

    def build_contexts(self, *, prompt, image):
        del prompt, image
        return UGContextBundle(
            full=UGContextHandle("full", 1),
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
        del contexts, timestep, latent_position_ids, sampling_params
        self.velocity_calls += 1
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
