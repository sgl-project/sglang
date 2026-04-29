# SPDX-License-Identifier: Apache-2.0
"""Manual live smoke for BAGEL Qwen2-MoT native SRT U-forward.

Usage:
CUDA_VISIBLE_DEVICES=0 \
SGLANG_TEST_BAGEL_QWEN2_MOT_MODEL=/data/models/BAGEL-7B-MoT \
python3 test/registered/scheduler/test_bagel_qwen2_mot_native_live.py
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=420,
    suite="stage-b-test-1-gpu-large",
    disabled=(
        "Manual BAGEL Qwen2-MoT native live smoke; requires "
        "SGLANG_TEST_BAGEL_QWEN2_MOT_MODEL"
    ),
)

_MODEL_ENV = "SGLANG_TEST_BAGEL_QWEN2_MOT_MODEL"
_OFFICIAL_REPO_ENV = "SGLANG_TEST_BAGEL_OFFICIAL_REPO"
_GLOBAL_ARGS_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base." "get_global_server_args"
)


class _NoopSender:
    def send_output(self, *args, **kwargs):
        del args, kwargs


class _RecordingUGModelRunner:
    def __init__(self, inner):
        self.inner = inner
        self.srt_request_views = []

    def observe_srt_u_forward(self, *, record, request, messages):
        del record, messages
        self.srt_request_views.append(request)

    def prefill_interleaved(self, *, record, messages):
        return self.inner.prefill_interleaved(record=record, messages=messages)

    def decode_next_segment(self, *, record):
        return self.inner.decode_next_segment(record=record)

    def predict_velocity_from_session(self, *, request, record):
        return self.inner.predict_velocity_from_session(request=request, record=record)

    def prepare_latents_from_session(self, *, request, record):
        return self.inner.prepare_latents_from_session(request=request, record=record)

    def append_generated_image(self, *, record, image):
        return self.inner.append_generated_image(record=record, image=image)

    def decode_latents_to_image(self, *, request, record):
        return self.inner.decode_latents_to_image(request=request, record=record)

    def close_session(self, *, session_id):
        self.inner.close_session(session_id=session_id)


def _replace_sender_with_noop(scheduler, name: str) -> None:
    sender = getattr(scheduler, name, None)
    socket = getattr(sender, "socket", None)
    if socket is not None:
        socket.close(linger=0)
    setattr(scheduler, name, _NoopSender())


def _write_language_model_view(checkpoint_dir: Path, output_dir: Path) -> Path:
    config_path = checkpoint_dir / "llm_config.json"
    weight_path = checkpoint_dir / "ema.safetensors"
    if not config_path.exists() or not weight_path.exists():
        raise FileNotFoundError(
            "BAGEL Qwen2-MoT live smoke requires llm_config.json and "
            f"ema.safetensors under {checkpoint_dir}"
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config.update(
        {
            "architectures": ["BAGELQwen2MoTForCausalLM"],
            "bagel_checkpoint_dir": str(checkpoint_dir),
            "bagel_enable_visual_feature_extractors": True,
            "bagel_connector_act": "gelu_pytorch_tanh",
            "bagel_latent_patch_size": 2,
            "bagel_max_latent_size": 64,
            "bagel_max_latent_tokens": 64 * 64,
            "bagel_vit_max_num_patch_per_side": 70,
            "layer_module": "Qwen2MoTDecoderLayer",
            "qk_norm": True,
            "tie_word_embeddings": False,
        }
    )
    (output_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    os.symlink(weight_path, output_dir / "model.safetensors")
    return output_dir


def _maybe_add_official_bagel_repo_to_path() -> None:
    candidates = []
    configured = os.getenv(_OFFICIAL_REPO_ENV)
    if configured:
        candidates.append(Path(configured))
    candidates.append(Path("/data/BAGEL"))

    for candidate in candidates:
        if not candidate.exists():
            continue
        if not (candidate / "inferencer.py").exists():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        return


def _make_ug_pipeline_server_args(scheduler) -> SimpleNamespace:
    from sglang.multimodal_gen.configs.pipeline_configs.ug import UGPipelineConfig
    from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

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
        ug_srt_scheduler=scheduler,
        ug_srt_u_decode_max_new_tokens=1,
    )


@unittest.skipUnless(os.getenv(_MODEL_ENV), f"Set {_MODEL_ENV} for live smoke")
class TestBAGELQwen2MoTNativeLive(CustomTestCase):
    def test_real_bagel_language_weights_run_srt_u_forward(self):
        import importlib.util

        import torch
        import torch.distributed as dist

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the BAGEL native live smoke")

        _maybe_add_official_bagel_repo_to_path()
        if importlib.util.find_spec("inferencer") is None:
            self.skipTest(
                "Set SGLANG_TEST_BAGEL_OFFICIAL_REPO to the official BAGEL repo "
                "for the BAGEL visual feature extractor live smoke"
            )

        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.models.bagel_qwen2_mot import (
            _iter_bagel_language_model_weights,
        )
        from sglang.srt.server_args import (
            PortArgs,
            ServerArgs,
            set_global_server_args_for_scheduler,
        )
        from sglang.srt.ug.bagel import BAGELNativeSRTPreparedDenoise
        from sglang.srt.ug.bagel_checkpoint import (
            load_bagel_checkpoint_keys,
            summarize_bagel_checkpoint_keys,
        )
        from sglang.srt.ug.runtime import (
            FakeUGModelRunner,
            UGInterleavedMessage,
            UGSessionRuntime,
        )
        from sglang.srt.ug.srt_executor import UGSRTSchedulerExecutor

        checkpoint_dir = Path(os.environ[_MODEL_ENV])
        keys = load_bagel_checkpoint_keys(checkpoint_dir)
        summary = summarize_bagel_checkpoint_keys(keys)
        language_keys = list(
            _iter_bagel_language_model_weights((key, None) for key in keys)
        )
        language_key_names = {name for name, _ in language_keys}
        self.assertGreaterEqual(
            len(language_keys),
            summary.counts["qwen2_shared"] + summary.counts["mot_gen_branch"],
        )
        self.assertGreater(summary.counts["mot_gen_branch"], 0)
        self.assertIn("vae2llm.weight", language_key_names)
        self.assertIn("llm2vae.weight", language_key_names)
        self.assertIn("time_embedder.mlp.0.weight", language_key_names)
        self.assertIn("latent_pos_embed.pos_embed", language_key_names)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _write_language_model_view(checkpoint_dir, Path(tmpdir))
            server_args = ServerArgs(
                model_path=str(model_path),
                tokenizer_path=str(checkpoint_dir),
                trust_remote_code=True,
                dtype="bfloat16",
                tp_size=1,
                pp_size=1,
                dp_size=1,
                disable_cuda_graph=True,
                disable_piecewise_cuda_graph=True,
                disable_overlap_schedule=True,
                skip_server_warmup=True,
                mem_fraction_static=float(
                    os.getenv("SGLANG_TEST_BAGEL_QWEN2_MOT_MEM_FRACTION", "0.35")
                ),
                chunked_prefill_size=int(
                    os.getenv("SGLANG_TEST_BAGEL_QWEN2_MOT_CHUNKED_PREFILL", "256")
                ),
                log_level="error",
            )
            server_args.check_server_args()
            set_global_server_args_for_scheduler(server_args)

            scheduler = Scheduler(
                server_args,
                PortArgs.init_new(server_args),
                gpu_id=int(os.getenv("SGLANG_TEST_BAGEL_QWEN2_MOT_GPU_ID", "0")),
                tp_rank=0,
                moe_ep_rank=0,
                pp_rank=0,
                attn_cp_rank=0,
                moe_dp_rank=0,
                dp_rank=None,
            )
            _replace_sender_with_noop(scheduler, "send_to_tokenizer")
            _replace_sender_with_noop(scheduler, "send_to_detokenizer")

            try:
                executor = UGSRTSchedulerExecutor(scheduler, max_sync_steps=16)
                model_runner = _RecordingUGModelRunner(FakeUGModelRunner())
                runtime = UGSessionRuntime(
                    model_runner=model_runner,
                    session_controller=scheduler.session_controller,
                    srt_request_executor=executor,
                    tokenizer=scheduler.tokenizer,
                    vocab_size=scheduler.model_config.vocab_size,
                    srt_u_decode_max_new_tokens=1,
                )

                handle = runtime.prefill_interleaved(
                    [UGInterleavedMessage(type="text", content="hello from bagel")],
                    session_id="bagel-native-srt-u-forward",
                )
                marker = runtime.decode_next_segment(handle)
                counters = runtime.get_debug_counters(handle)

                self.assertEqual(marker.type, "image_marker")
                self.assertEqual(counters["prefill_count"], 1)
                self.assertEqual(counters["decode_count"], 1)
                self.assertEqual(counters["srt_request_count"], 2)
                self.assertEqual(counters["srt_executed_request_count"], 2)
                self.assertEqual(counters["srt_u_decode_request_count"], 1)
                self.assertEqual(len(counters["srt_last_u_decode_output_ids"]), 1)
                self.assertEqual(
                    counters["srt_model_runner_forward_request_ids"],
                    ["bagel-native-srt-u-forward:u1"],
                )
                self.assertEqual(executor.sync_step_count, 2)
                self.assertTrue(scheduler.is_fully_idle())

                token_bindings = [
                    request.metadata.get("srt_kv_token_binding")
                    for request in model_runner.srt_request_views
                ]
                self.assertTrue(token_bindings)
                self.assertTrue(all(binding is not None for binding in token_bindings))
                self.assertTrue(
                    all(binding.token_count > 0 for binding in token_bindings)
                )

                latest_binding = token_bindings[-1]
                req_slots_before = scheduler.req_to_token_pool.available_size()
                kv_slots_before = scheduler.token_to_kv_pool_allocator.available_size()
                native_denoise_executor = (
                    executor.create_bagel_native_srt_denoise_executor()
                )
                prepared = BAGELNativeSRTPreparedDenoise(
                    generation_input={
                        "packed_text_ids": torch.tensor([1, 2], dtype=torch.long),
                        "packed_text_indexes": torch.tensor([0, 3], dtype=torch.long),
                        "packed_vae_token_indexes": torch.tensor(
                            [1, 2], dtype=torch.long
                        ),
                        "packed_vae_position_ids": torch.tensor(
                            [0, 1], dtype=torch.long
                        ),
                        "packed_seqlens": torch.tensor([4], dtype=torch.int32),
                        "packed_position_ids": torch.arange(
                            latest_binding.token_count,
                            latest_binding.token_count + 4,
                            dtype=torch.long,
                        ),
                        "packed_indexes": torch.arange(4, dtype=torch.long),
                        "key_values_lens": torch.tensor(
                            [latest_binding.token_count], dtype=torch.int32
                        ),
                        "packed_key_value_indexes": torch.arange(
                            latest_binding.token_count, dtype=torch.long
                        ),
                    },
                    srt_kv_token_binding=latest_binding,
                )
                velocity = native_denoise_executor.predict_velocity(
                    prepared=prepared,
                    latent_tokens=torch.zeros(
                        2,
                        64,
                        dtype=torch.bfloat16,
                        device=torch.device("cuda"),
                    ),
                    timestep=torch.tensor([0.5], device=torch.device("cuda")),
                )

                self.assertEqual(tuple(velocity.shape), (2, 64))
                self.assertEqual(executor.temp_g_forward_count, 1)
                self.assertEqual(executor.temp_g_allocated_token_count, 4)
                self.assertEqual(
                    scheduler.req_to_token_pool.available_size(), req_slots_before
                )
                self.assertGreaterEqual(
                    scheduler.token_to_kv_pool_allocator.available_size(),
                    kv_slots_before,
                )
                self.assertTrue(scheduler.is_fully_idle())

                runtime.close_session(handle)
                self.assertNotIn(
                    handle.session_id, scheduler.session_controller.sessions
                )
            finally:
                if dist.is_available() and dist.is_initialized():
                    dist.destroy_process_group()
                torch.cuda.empty_cache()

    def test_full_ug_pipeline_uses_native_srt_g_velocity(self):
        import importlib.util

        import torch
        import torch.distributed as dist
        from PIL import Image

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the BAGEL native pipeline live smoke")

        _maybe_add_official_bagel_repo_to_path()
        if importlib.util.find_spec("inferencer") is None:
            self.skipTest(
                "Set SGLANG_TEST_BAGEL_OFFICIAL_REPO to the official BAGEL repo "
                "for the full BAGEL pipeline smoke"
            )

        from sglang.multimodal_gen.configs.sample.ug import UGSamplingParams
        from sglang.multimodal_gen.runtime.pipelines.ug import UGPipeline
        from sglang.multimodal_gen.runtime.pipelines_core.executors.sync_executor import (
            SyncExecutor,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.server_args import (
            PortArgs,
            ServerArgs,
            set_global_server_args_for_scheduler,
        )

        checkpoint_dir = Path(os.environ[_MODEL_ENV])
        contexts = None
        bridge = None
        scheduler = None
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _write_language_model_view(checkpoint_dir, Path(tmpdir))
            server_args = ServerArgs(
                model_path=str(model_path),
                tokenizer_path=str(checkpoint_dir),
                trust_remote_code=True,
                dtype="bfloat16",
                tp_size=1,
                pp_size=1,
                dp_size=1,
                disable_cuda_graph=True,
                disable_piecewise_cuda_graph=True,
                disable_overlap_schedule=True,
                skip_server_warmup=True,
                mem_fraction_static=float(
                    os.getenv("SGLANG_TEST_BAGEL_QWEN2_MOT_MEM_FRACTION", "0.35")
                ),
                chunked_prefill_size=int(
                    os.getenv("SGLANG_TEST_BAGEL_QWEN2_MOT_CHUNKED_PREFILL", "256")
                ),
                log_level="error",
            )
            server_args.check_server_args()
            set_global_server_args_for_scheduler(server_args)

            scheduler = Scheduler(
                server_args,
                PortArgs.init_new(server_args),
                gpu_id=int(os.getenv("SGLANG_TEST_BAGEL_QWEN2_MOT_GPU_ID", "0")),
                tp_rank=0,
                moe_ep_rank=0,
                pp_rank=0,
                attn_cp_rank=0,
                moe_dp_rank=0,
                dp_rank=None,
            )
            _replace_sender_with_noop(scheduler, "send_to_tokenizer")
            _replace_sender_with_noop(scheduler, "send_to_detokenizer")

            try:
                diffusion_args = _make_ug_pipeline_server_args(scheduler)
                with patch(_GLOBAL_ARGS_PATCH, return_value=diffusion_args):
                    pipeline = UGPipeline(
                        str(checkpoint_dir),
                        diffusion_args,
                        executor=SyncExecutor(diffusion_args),
                    )

                bridge = pipeline.get_module("ug_bridge")
                runtime = bridge.runtime
                srt_executor = runtime.srt_request_executor
                native_executor = (
                    runtime.model_runner.adapter.backend.native_srt_denoise_executor
                )
                req_slots_before = scheduler.req_to_token_pool.available_size()
                num_inference_steps = 3

                result = pipeline.forward(
                    Req(
                        sampling_params=UGSamplingParams(
                            prompt="draw a small lantern, then describe it",
                            width=32,
                            height=32,
                            seed=123,
                            num_inference_steps=num_inference_steps,
                            cfg_text_scale=1.0,
                            cfg_img_scale=1.0,
                            cfg_interval=[0.0, 1.0],
                            return_trajectory_latents=True,
                            suppress_logs=True,
                        ),
                        condition_image=Image.new("RGB", (16, 16), color="white"),
                    ),
                    diffusion_args,
                )
                contexts = result.extra["ug_contexts"]
                session = contexts.full.session
                counters = runtime.get_debug_counters(session)

                self.assertEqual(tuple(result.output.shape), (1, 32, 32, 3))
                self.assertEqual(result.extra["ug_post_image_segment"].type, "text")
                self.assertEqual(contexts.full.session.session_id, session.session_id)
                self.assertEqual(counters["prefill_count"], 1)
                self.assertEqual(counters["velocity_count"], num_inference_steps - 1)
                self.assertEqual(counters["append_image_count"], 1)
                self.assertEqual(counters["decode_count"], 2)
                self.assertEqual(counters["srt_request_count"], 7)
                self.assertEqual(counters["srt_u_decode_request_count"], 2)
                self.assertEqual(counters["srt_executed_request_count"], 7)
                self.assertEqual(
                    srt_executor.temp_g_forward_count,
                    num_inference_steps - 1,
                )
                self.assertEqual(
                    native_executor.velocity_count,
                    num_inference_steps - 1,
                )
                self.assertGreater(srt_executor.temp_g_allocated_token_count, 0)
                self.assertTrue(scheduler.is_fully_idle())

                bridge.release_contexts(contexts)
                contexts = None
                closed = runtime.get_debug_counters(session.session_id)
                self.assertTrue(closed["closed"])
                self.assertEqual(closed["state"], "done")
                self.assertGreaterEqual(
                    scheduler.req_to_token_pool.available_size(),
                    req_slots_before,
                )
                self.assertTrue(scheduler.is_fully_idle())
            finally:
                if bridge is not None and contexts is not None:
                    bridge.release_contexts(contexts)
                if dist.is_available() and dist.is_initialized():
                    dist.destroy_process_group()
                torch.cuda.empty_cache()

    def test_forward_interleaved_api_runs_full_native_pipeline(self):
        import importlib.util

        import torch
        import torch.distributed as dist
        from PIL import Image

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the BAGEL interleaved API smoke")

        _maybe_add_official_bagel_repo_to_path()
        if importlib.util.find_spec("inferencer") is None:
            self.skipTest(
                "Set SGLANG_TEST_BAGEL_OFFICIAL_REPO to the official BAGEL repo "
                "for the BAGEL interleaved API smoke"
            )

        from sglang.multimodal_gen.runtime.pipelines.ug import UGPipeline
        from sglang.multimodal_gen.runtime.pipelines_core.executors.sync_executor import (
            SyncExecutor,
        )
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.server_args import (
            PortArgs,
            ServerArgs,
            set_global_server_args_for_scheduler,
        )

        checkpoint_dir = Path(os.environ[_MODEL_ENV])
        scheduler = None
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _write_language_model_view(checkpoint_dir, Path(tmpdir))
            server_args = ServerArgs(
                model_path=str(model_path),
                tokenizer_path=str(checkpoint_dir),
                trust_remote_code=True,
                dtype="bfloat16",
                tp_size=1,
                pp_size=1,
                dp_size=1,
                disable_cuda_graph=True,
                disable_piecewise_cuda_graph=True,
                disable_overlap_schedule=True,
                skip_server_warmup=True,
                mem_fraction_static=float(
                    os.getenv("SGLANG_TEST_BAGEL_QWEN2_MOT_MEM_FRACTION", "0.35")
                ),
                chunked_prefill_size=int(
                    os.getenv("SGLANG_TEST_BAGEL_QWEN2_MOT_CHUNKED_PREFILL", "256")
                ),
                log_level="error",
            )
            server_args.check_server_args()
            set_global_server_args_for_scheduler(server_args)

            scheduler = Scheduler(
                server_args,
                PortArgs.init_new(server_args),
                gpu_id=int(os.getenv("SGLANG_TEST_BAGEL_QWEN2_MOT_GPU_ID", "0")),
                tp_rank=0,
                moe_ep_rank=0,
                pp_rank=0,
                attn_cp_rank=0,
                moe_dp_rank=0,
                dp_rank=None,
            )
            _replace_sender_with_noop(scheduler, "send_to_tokenizer")
            _replace_sender_with_noop(scheduler, "send_to_detokenizer")

            try:
                diffusion_args = _make_ug_pipeline_server_args(scheduler)
                with patch(_GLOBAL_ARGS_PATCH, return_value=diffusion_args):
                    pipeline = UGPipeline(
                        str(checkpoint_dir),
                        diffusion_args,
                        executor=SyncExecutor(diffusion_args),
                    )

                segments = pipeline.forward_interleaved(
                    [
                        {
                            "type": "image",
                            "image": Image.new("RGB", (16, 16), color="white"),
                        },
                        {
                            "type": "text",
                            "text": "draw a small lantern, then describe it",
                        },
                    ],
                    {
                        "width": 32,
                        "height": 32,
                        "seed": 123,
                        "num_inference_steps": 3,
                        "cfg_text_scale": 1.0,
                        "cfg_img_scale": 1.0,
                        "cfg_interval": [0.0, 1.0],
                        "suppress_logs": True,
                    },
                )

                self.assertEqual(
                    [segment["type"] for segment in segments],
                    ["image", "text"],
                )
                self.assertIsInstance(segments[0]["image"], Image.Image)
                self.assertEqual(segments[0]["image"].size, (32, 32))
                self.assertIsInstance(segments[1]["text"], str)
                self.assertTrue(scheduler.is_fully_idle())
                self.assertEqual(scheduler.session_controller.sessions, {})
            finally:
                if dist.is_available() and dist.is_initialized():
                    dist.destroy_process_group()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
