"""Unit tests for the --enable-cfg-parallel warmup fix and guard.

Covers warmup and cfg-parallel guard paths introduced alongside this file:
- build_warmup_reqs synthesizes warmup Reqs that actually enable
  classifier-free guidance when cfg-parallel is on.
- DiffGenerator sends explicit warmup resolutions through the scheduler client.
- InputValidationStage.forward ACCEPTS non-CFG requests when the server
  has cfg-parallel on, and the branch dispatcher serves them.
- Server-based warmup can opt into model-default negative prompts so warmup
  populates the negative text embedding cache.
- Req-based warmup remains available only through the lazy legacy path.

All tests are CPU-only; no model loading, no distributed init.
"""

import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.flux_finetuned import (
    Flux2FinetunedPipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.longlive2 import (
    LongLive2T2VConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.ltx_2_5 import LTX25PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.sana_wm import SanaWMPipelineConfig
from sglang.multimodal_gen.configs.sample.longlive2 import LongLive2SamplingParams
from sglang.multimodal_gen.configs.sample.ltx_2_5 import LTX25SamplingParams
from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.sana_wm import SanaWMSamplingParams
from sglang.multimodal_gen.runtime.entrypoints.control_requests import (
    SetLoraReq,
    UnmergeLoraWeightsReq,
)
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageVAEEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.server_warmup import (
    format_warmup_req,
    should_run_explicit_client_warmup,
    should_run_synthetic_server_warmup,
)
from sglang.multimodal_gen.runtime.warmup_request_builder import (
    DEFAULT_PLACEHOLDER_PROMPT,
    SERVER_WARMUP_IMAGE_FALLBACK_RESOLUTION,
    _apply_warmup_sampling_overrides,
    _resolve_warmup_num_frames,
    build_warmup_reqs,
    should_include_warmup_image,
    supports_synthetic_warmup,
)
from sglang.multimodal_gen.test.server.gpu_cases import ONE_GPU_CASES, TWO_GPU_CASES
from sglang.multimodal_gen.test.server.testcase_configs import _get_extra_arg_value


def _make_bare_scheduler(enable_cfg_parallel: bool) -> Scheduler:
    """
    Build a minimal Scheduler without calling __init__ (which requires
    distributed init, ZMQ sockets, pipeline load, etc.). Populates only
    the attributes req-based warmup reads/writes.
    """
    scheduler = object.__new__(Scheduler)

    server_args = MagicMock()
    server_args.warmup_mode = "request"
    server_args.warmup_steps = 1
    server_args.warmup_resolutions = ["512x512"]
    server_args.enable_cfg_parallel = enable_cfg_parallel
    server_args.enable_torch_compile = False
    server_args.is_arg_explicitly_set.return_value = False

    server_args.pipeline_config.task_type = ModelTaskType.T2I

    scheduler.server_args = server_args
    scheduler.req_based_warmup_scheduled = False
    scheduler.waiting_queue = deque()
    return scheduler


def _make_input_validation_stage() -> InputValidationStage:
    return InputValidationStage()


def _make_generation_req() -> Req:
    return Req(
        data_type=ModelTaskType.T2I.data_type(),
        prompt="prompt",
        width=512,
        height=512,
        num_inference_steps=20,
    )


def _make_validation_server_args(enable_cfg_parallel: bool) -> MagicMock:
    sa = MagicMock()
    sa.enable_cfg_parallel = enable_cfg_parallel
    sa.pipeline_config.task_type = ModelTaskType.T2I
    return sa


class TestWarmupReqCfgParallel(unittest.TestCase):
    """Warmup request construction and req-based warmup guards."""

    def test_sampling_workload_override_accepts_json(self):
        defaults = SamplingParams(
            width=1024,
            height=1024,
            num_frames=81,
            num_inference_steps=35,
        )
        server_args = SimpleNamespace(
            warmup_sampling_params=(
                '{"width":832,"height":480,"num_frames":9,"num_inference_steps":4}'
            )
        )

        overridden = _apply_warmup_sampling_overrides(server_args, defaults)

        self.assertEqual(
            (
                overridden.width,
                overridden.height,
                overridden.num_frames,
                overridden.num_inference_steps,
            ),
            (832, 480, 9, 4),
        )
        self.assertEqual((defaults.width, defaults.height), (1024, 1024))

    def test_sampling_workload_override_rejects_unknown_field(self):
        server_args = SimpleNamespace(
            warmup_sampling_params={"not_a_sampling_field": 1}
        )

        with self.assertRaisesRegex(ValueError, "invalid --warmup-sampling-params"):
            _apply_warmup_sampling_overrides(server_args, SamplingParams())

    def test_sampling_workload_override_supports_fixed_model_fields(self):
        defaults = MiniMaxH3SamplingParams()
        server_args = SimpleNamespace(
            warmup_sampling_params={"num_frames": 49, "fps": 12}
        )

        overridden = _apply_warmup_sampling_overrides(server_args, defaults)

        self.assertEqual((overridden.num_frames, overridden.fps), (49, 12))
        self.assertEqual((defaults.num_frames, defaults.fps), (1, 24))

    def test_warmup_req_cfg_parallel_sets_do_cfg(self):
        server_args = _make_bare_scheduler(enable_cfg_parallel=True).server_args
        sampling_defaults = SamplingParams()
        with patch(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            return_value=sampling_defaults,
        ):
            req = build_warmup_reqs(
                server_args,
                warmup_resolutions=["512x512"],
                server_based_warmup=True,
            )[0]
        self.assertIs(req.do_classifier_free_guidance, True)
        self.assertEqual(req.negative_prompt, sampling_defaults.negative_prompt)

    def test_warmup_req_cfg_parallel_fills_missing_negative_prompt(self):
        server_args = _make_bare_scheduler(enable_cfg_parallel=True).server_args
        with patch(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            return_value=SamplingParams(negative_prompt=None),
        ):
            req = build_warmup_reqs(
                server_args,
                warmup_resolutions=["512x512"],
                server_based_warmup=True,
            )[0]
        self.assertIs(req.do_classifier_free_guidance, True)
        self.assertEqual(req.negative_prompt, DEFAULT_PLACEHOLDER_PROMPT)

    def test_warmup_req_no_cfg_parallel_unchanged(self):
        # Regression guard: the cfg-parallel=on fix must not bleed into
        # the cfg-parallel=off path. Key invariant is do_cfg stays False
        # AND the synthesized Req is not using the cfg-parallel-specific
        # "warmup" placeholder for negative_prompt (which would indicate
        # the fix's kwargs leaked into this branch).
        server_args = _make_bare_scheduler(enable_cfg_parallel=False).server_args
        with patch(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            return_value=SamplingParams(),
        ):
            req = build_warmup_reqs(
                server_args,
                warmup_resolutions=["512x512"],
                server_based_warmup=True,
            )[0]
        self.assertIs(req.do_classifier_free_guidance, False)
        self.assertNotEqual(req.negative_prompt, DEFAULT_PLACEHOLDER_PROMPT)

    def test_server_warmup_keeps_minimum_image_steps_without_compile(self):
        server_args = _make_bare_scheduler(enable_cfg_parallel=False).server_args
        with patch(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            return_value=SamplingParams(num_inference_steps=9),
        ):
            req = build_warmup_reqs(
                server_args,
                warmup_resolutions=["512x512"],
                server_based_warmup=True,
            )[0]
        self.assertEqual(req.num_inference_steps, 2)

    def test_torch_compile_respects_explicit_server_warmup_steps(self):
        server_args = _make_bare_scheduler(enable_cfg_parallel=False).server_args
        server_args.enable_torch_compile = True
        server_args.is_arg_explicitly_set.side_effect = lambda name: (
            name == "warmup_steps"
        )
        with patch(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            return_value=SamplingParams(num_inference_steps=9),
        ):
            req = build_warmup_reqs(
                server_args,
                warmup_resolutions=["512x512"],
                server_based_warmup=True,
            )[0]
        self.assertIn("(512x512, 1/9 steps)", format_warmup_req(req))

    def test_torch_compile_server_warmup_repeats_each_bucket(self):
        server_args = _make_bare_scheduler(enable_cfg_parallel=False).server_args
        server_args.enable_torch_compile = True
        with patch(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            return_value=SamplingParams(num_inference_steps=9),
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=["512x512", "1024x1024"],
                server_based_warmup=True,
            )
        self.assertEqual(len(reqs), 4)
        self.assertEqual(
            [(req.width, req.height) for req in reqs],
            [(512, 512)] * 2 + [(1024, 1024)] * 2,
        )
        self.assertEqual([req.is_warmup for req in reqs], [True, False] * 2)
        self.assertEqual([req.num_inference_steps for req in reqs], [2] * 4)
        self.assertEqual(
            [req.extra.get("server_internal_prewarm", False) for req in reqs],
            [False, True] * 2,
        )
        self.assertEqual([req.save_output for req in reqs], [False] * 4)
        self.assertIsNot(reqs[0].sampling_params, reqs[1].sampling_params)
        self.assertEqual(reqs[1].sampling_params.num_inference_steps, 2)
        reqs[1].sampling_params.num_inference_steps = 123
        self.assertEqual(reqs[0].sampling_params.num_inference_steps, 2)

    def test_lightweight_warmup_result_ignores_control_requests(self):
        scheduler = _make_bare_scheduler(enable_cfg_parallel=False)

        self.assertFalse(
            scheduler._should_return_lightweight_warmup_result(SetLoraReq("test"))
        )
        self.assertFalse(
            scheduler._should_return_lightweight_warmup_result(UnmergeLoraWeightsReq())
        )

    def test_lightweight_warmup_result_returns_internal_prewarm(self):
        scheduler = _make_bare_scheduler(enable_cfg_parallel=False)
        req = _make_generation_req()
        req.extra["server_internal_prewarm"] = True

        self.assertTrue(scheduler._should_return_lightweight_warmup_result(req))

    def test_req_based_warmup_remains_explicit_legacy_entry(self):
        scheduler = _make_bare_scheduler(enable_cfg_parallel=False)
        scheduler.server_args.warmup_resolutions = None
        scheduler.server_args.warmup_mode = "request"

        req = _make_generation_req()
        recv_reqs = [(b"0", req)]
        processed = scheduler.process_received_reqs_with_req_based_warmup(recv_reqs)

        self.assertEqual(len(processed), 2)
        self.assertIs(processed[1][1], req)
        self.assertIsNot(processed[0][1], req)
        self.assertTrue(processed[0][1].is_warmup)
        self.assertTrue(processed[0][1].metrics.suppress_stage_breakdown)
        self.assertEqual(processed[0][1].num_inference_steps, 1)
        self.assertEqual(processed[0][1].extra["cache_dit_num_inference_steps"], 20)
        self.assertTrue(scheduler.req_based_warmup_scheduled)

    def test_req_based_warmup_skips_default_server_warmup_path(self):
        scheduler = _make_bare_scheduler(enable_cfg_parallel=False)
        scheduler.server_args.warmup_resolutions = None
        scheduler.server_args.warmup_mode = "server"

        recv_reqs = [(b"0", _make_generation_req())]
        processed = scheduler.process_received_reqs_with_req_based_warmup(recv_reqs)

        self.assertIs(processed, recv_reqs)
        self.assertEqual(len(processed), 1)
        self.assertFalse(scheduler.req_based_warmup_scheduled)

    def test_diff_generator_runs_explicit_warmup_through_scheduler_client(self):
        generator = object.__new__(DiffGenerator)
        server_args = MagicMock()
        server_args.warmup_mode = "request"
        server_args.warmup_resolutions = ["832x480"]
        server_args.warmup_steps = 1
        server_args.enable_cfg_parallel = False
        server_args.enable_torch_compile = False
        server_args.num_gpus = 1

        server_args.pipeline_config.task_type = ModelTaskType.T2V
        server_args.pipeline_config.adjust_num_frames.side_effect = lambda value: value
        generator.server_args = server_args

        sampling_defaults = SamplingParams(num_frames=81, num_inference_steps=50)
        with (
            patch(
                "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
                return_value=sampling_defaults,
            ),
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.diffusion_generator.sync_scheduler_client.forward",
                return_value=OutputBatch(error=None),
            ) as forward,
        ):
            generator._run_client_warmup_if_needed()

        forward.assert_called_once()
        req = forward.call_args.args[0]
        self.assertTrue(req.is_warmup)
        self.assertEqual((req.width, req.height), (832, 480))
        self.assertEqual(req.num_frames, 17)
        self.assertEqual(req.num_inference_steps, 2)
        self.assertTrue(req.extra["return_warmup_result"])
        self.assertTrue(req.extra["server_based_warmup"])
        self.assertEqual(req.extra["warmup_total"], 1)

    def test_server_based_warmup_uses_model_default_negative_prompt(self):
        server_args = MagicMock()
        server_args.warmup_steps = 1
        server_args.enable_cfg_parallel = False
        server_args.enable_torch_compile = False

        server_args.pipeline_config.task_type = ModelTaskType.T2I

        sampling_defaults = SamplingParams(
            negative_prompt="model default negative",
            guidance_scale=4.0,
            num_inference_steps=20,
        )
        with patch(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            return_value=sampling_defaults,
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=None,
                return_warmup_result=True,
                server_based_warmup=True,
            )

        self.assertEqual(len(reqs), 1)
        req = reqs[0]
        self.assertTrue(req.is_warmup)
        self.assertTrue(req.metrics.suppress_stage_breakdown)
        self.assertEqual(req.num_inference_steps, 2)
        self.assertEqual(req.extra["cache_dit_num_inference_steps"], 20)
        self.assertEqual(req.negative_prompt, "model default negative")
        self.assertEqual(req.prompt, DEFAULT_PLACEHOLDER_PROMPT)
        self.assertIs(req.do_classifier_free_guidance, True)
        self.assertTrue(req.extra["return_warmup_result"])
        self.assertTrue(req.extra["server_based_warmup"])

    def test_auto_residency_uses_one_full_serving_shape_probe(self):
        server_args = SimpleNamespace(
            warmup_steps=1,
            enable_cfg_parallel=False,
            enable_torch_compile=False,
            enable_breakable_cuda_graph=False,
            pipeline_class_name=None,
            num_gpus=1,
            pipeline_config=SimpleNamespace(
                task_type=ModelTaskType.T2V,
                adjust_num_frames=lambda value: value,
                vae_stride=None,
                vae_scale_factor=None,
                vae_config=SimpleNamespace(arch_config=None),
            ),
            is_arg_explicitly_set=lambda _name: False,
        )
        sampling_defaults = SamplingParams(
            width=1280,
            height=720,
            num_frames=81,
            num_inference_steps=35,
            adjust_frames=False,
            supported_resolutions=[(1280, 720), (832, 480)],
        )
        with (
            patch(
                "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
                return_value=sampling_defaults,
            ),
            patch(
                "sglang.multimodal_gen.runtime.warmup_request_builder.auto_residency_args_skip_reason",
                return_value=None,
            ),
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=None,
                server_based_warmup=True,
            )

        # the bounded warmup runs first so the worker can size the probe, and
        # once more after it so serving starts from a serving-shaped pool
        self.assertEqual(len(reqs), 3)
        self.assertFalse(reqs[0].extra.get("auto_residency_full_shape_probe"))
        self.assertFalse(reqs[2].extra.get("auto_residency_full_shape_probe"))
        self.assertEqual(
            (reqs[2].width, reqs[2].height, reqs[2].num_frames),
            (reqs[0].width, reqs[0].height, reqs[0].num_frames),
        )
        self.assertEqual(
            (reqs[1].width, reqs[1].height, reqs[1].num_frames),
            (1280, 720, 81),
        )
        self.assertTrue(reqs[1].extra["auto_residency_full_shape_probe"])
        self.assertFalse(reqs[1].metrics.suppress_stage_breakdown)
        self.assertEqual(reqs[1].num_inference_steps, 4)
        self.assertIn(
            "auto residency probe (1280x720x81f, 4/35 steps)",
            format_warmup_req(reqs[1]),
        )

    def test_server_based_warmup_uses_model_default_resolution(self):
        server_args = MagicMock()
        server_args.warmup_steps = 1
        server_args.enable_cfg_parallel = False
        server_args.enable_torch_compile = False

        server_args.pipeline_config.task_type = ModelTaskType.T2I

        sampling_defaults = SamplingParams(width=640, height=640)
        with patch(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            return_value=sampling_defaults,
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=None,
                server_based_warmup=True,
            )

        req = reqs[0]
        self.assertEqual(req.width, 640)
        self.assertEqual(req.height, 640)

    def test_server_based_warmup_resolutions_keep_sampling_defaults_and_caps(self):
        server_args = MagicMock()
        server_args.warmup_steps = 1
        server_args.enable_cfg_parallel = False
        server_args.enable_torch_compile = False
        server_args.num_gpus = 1

        server_args.pipeline_config.task_type = ModelTaskType.T2V
        server_args.pipeline_config.adjust_num_frames.side_effect = lambda value: value

        sampling_defaults = SamplingParams(
            negative_prompt="model default negative",
            guidance_scale=3.5,
            num_frames=81,
            num_inference_steps=50,
        )
        with patch(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            return_value=sampling_defaults,
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=["832x480"],
                return_warmup_result=True,
                server_based_warmup=True,
            )

        req = reqs[0]
        self.assertEqual((req.width, req.height), (832, 480))
        self.assertEqual(req.num_frames, 17)
        self.assertEqual(req.num_inference_steps, 2)
        self.assertEqual(req.extra["cache_dit_num_inference_steps"], 50)
        self.assertEqual(req.negative_prompt, "model default negative")
        self.assertIs(req.do_classifier_free_guidance, True)

    def test_server_based_image_warmup_uses_model_default_over_supported(self):
        """Server-based image warmup uses the model's default resolution so it
        warms up at the real inference shape (avoiding a residual
        cudagraph/compile gap), rather than shrinking to the smallest supported
        resolution within an area budget."""
        server_args = MagicMock()
        server_args.warmup_steps = 1
        server_args.enable_cfg_parallel = False
        server_args.enable_torch_compile = False

        server_args.pipeline_config.task_type = ModelTaskType.T2I

        sampling_defaults = SamplingParams(
            width=1024,
            height=1024,
            supported_resolutions=[(512, 512), (1024, 1024)],
        )
        with patch(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            return_value=sampling_defaults,
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=None,
                server_based_warmup=True,
            )

        self.assertEqual((reqs[0].width, reqs[0].height), (1024, 1024))

    def test_server_based_image_warmup_uses_full_model_default(self):
        """Server-based image warmup keeps the model's full default resolution
        instead of scaling down to a server-warmup area budget, so warmup hits
        the real inference shape."""
        server_args = MagicMock()
        server_args.warmup_steps = 1
        server_args.enable_cfg_parallel = False
        server_args.enable_torch_compile = False
        server_args.backend = "auto"

        server_args.pipeline_config.task_type = ModelTaskType.T2I

        sampling_defaults = SamplingParams(width=1024, height=1024)
        with patch(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            return_value=sampling_defaults,
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=None,
                server_based_warmup=True,
            )

        self.assertEqual((reqs[0].width, reqs[0].height), (1024, 1024))

    def test_server_based_image_warmup_diffusers_uses_model_default(self):
        """Even on the diffusers backend, server-based image warmup uses the
        model default resolution rather than the diffusers image area budget."""
        server_args = MagicMock()
        server_args.warmup_steps = 1
        server_args.enable_cfg_parallel = False
        server_args.enable_torch_compile = False
        server_args.backend = "diffusers"

        server_args.pipeline_config.task_type = ModelTaskType.T2I

        sampling_defaults = SamplingParams(width=1024, height=1024)
        with patch(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            return_value=sampling_defaults,
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=None,
                server_based_warmup=True,
            )

        self.assertEqual((reqs[0].width, reqs[0].height), (1024, 1024))

    def test_server_based_warmup_keeps_video_warmup_lightweight(self):
        server_args = MagicMock()
        server_args.warmup_steps = 1
        server_args.enable_cfg_parallel = False
        server_args.enable_torch_compile = False
        server_args.num_gpus = 1

        server_args.pipeline_config.task_type = ModelTaskType.T2V
        server_args.pipeline_config.adjust_num_frames.side_effect = lambda value: value

        sampling_defaults = SamplingParams(
            width=832,
            height=480,
            num_frames=81,
            num_inference_steps=50,
        )
        with patch(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            return_value=sampling_defaults,
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=None,
                server_based_warmup=True,
            )

        self.assertEqual(reqs[0].num_inference_steps, 2)
        self.assertEqual(reqs[0].num_frames, 17)

    def test_video_warmup_preserves_model_frame_alignment(self):
        pipeline_config = LongLive2T2VConfig()
        server_args = SimpleNamespace(
            pipeline_config=pipeline_config,
            enable_breakable_cuda_graph=False,
            pipeline_class_name=None,
            num_gpus=1,
        )

        num_frames = _resolve_warmup_num_frames(
            server_args,
            LongLive2SamplingParams(),
            server_based_warmup=True,
        )

        temporal_scale = pipeline_config.vae_config.arch_config.scale_factor_temporal
        latent_frames = (num_frames - 1) // temporal_scale + 1
        self.assertEqual(num_frames, 29)
        self.assertEqual(
            latent_frames % pipeline_config.dit_config.arch_config.num_frames_per_block,
            0,
        )

    def test_breakable_cuda_graph_uses_explicit_warmup_num_frames(self):
        pipeline_config = MagicMock()
        pipeline_config.task_type = ModelTaskType.T2V
        pipeline_config.adjust_num_frames.side_effect = lambda value: value
        server_args = SimpleNamespace(
            pipeline_config=pipeline_config,
            enable_breakable_cuda_graph=True,
            pipeline_class_name=None,
            num_gpus=1,
            warmup_num_frames=17,
        )

        num_frames = _resolve_warmup_num_frames(
            server_args,
            SamplingParams(num_frames=81),
            server_based_warmup=True,
        )

        self.assertEqual(num_frames, 17)
        pipeline_config.adjust_num_frames.assert_called_once_with(17)

    def test_server_warmup_preserves_explicit_frames_without_cuda_graphs(self):
        server_args = SimpleNamespace(
            pipeline_config=LTX25PipelineConfig(),
            enable_breakable_cuda_graph=False,
            pipeline_class_name="LTX2Pipeline",
            num_gpus=2,
            warmup_num_frames=49,
        )

        num_frames = _resolve_warmup_num_frames(
            server_args, LTX25SamplingParams(), server_based_warmup=True
        )

        self.assertEqual(num_frames, 57)

    def test_sana_ci_warmup_matches_formal_shape(self):
        case = next(case for case in ONE_GPU_CASES if case.id == "sana_wm_ti2v")
        resolution = _get_extra_arg_value(
            case.server_args.extras, "--warmup-resolutions"
        )
        server_args = SimpleNamespace(
            pipeline_config=SanaWMPipelineConfig(),
            pipeline_class_name=None,
            model_path=case.server_args.model_path,
            model_id=None,
            backend="sglang",
            num_gpus=1,
            warmup_steps=1,
            warmup_num_frames=None,
            warmup_sampling_params=None,
            enable_breakable_cuda_graph=False,
            enable_torch_compile=False,
            enable_cfg_parallel=False,
        )
        with patch.object(
            SamplingParams, "from_pretrained", return_value=SanaWMSamplingParams()
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=[resolution],
                warmup_input_path="synthetic-warmup.png",
                server_based_warmup=True,
            )
        self.assertEqual(resolution, case.sampling_params.output_size)
        self.assertEqual(len(reqs), 1)
        self.assertEqual((reqs[0].width, reqs[0].height), (384, 640))
        self.assertEqual(reqs[0].num_frames, case.sampling_params.num_frames)

    def test_ltx25_ci_warmup_matches_formal_decoder_and_shape(self):
        case = next(
            case
            for case in TWO_GPU_CASES
            if case.id == "ltx_2_5_diffusion_decoder_2gpus"
        )
        extras = case.server_args.extras
        server_args = SimpleNamespace(
            pipeline_config=LTX25PipelineConfig(),
            pipeline_class_name=None,
            model_path=case.server_args.model_path,
            model_id=None,
            backend="sglang",
            num_gpus=2,
            warmup_steps=1,
            warmup_num_frames=int(_get_extra_arg_value(extras, "--warmup-num-frames")),
            warmup_sampling_params=_get_extra_arg_value(
                extras, "--warmup-sampling-params"
            ),
            enable_breakable_cuda_graph=False,
            enable_torch_compile=False,
            enable_cfg_parallel=False,
        )
        resolution = _get_extra_arg_value(extras, "--warmup-resolutions")
        with patch.object(
            SamplingParams, "from_pretrained", return_value=LTX25SamplingParams()
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=[resolution],
                warmup_input_path="synthetic-warmup.png",
                server_based_warmup=True,
            )

        self.assertEqual(len(reqs), 1)
        req = reqs[0]
        self.assertEqual(resolution, case.sampling_params.output_size)
        self.assertEqual(server_args.warmup_num_frames, case.sampling_params.num_frames)
        self.assertEqual((req.width, req.height, req.num_frames), (768, 448, 57))
        self.assertEqual(
            req.sampling_params.use_diffusion_decoder,
            case.sampling_params.extras["use_diffusion_decoder"],
        )
        self.assertEqual(req.num_inference_steps, 2)

    def test_server_based_warmup_uses_video_supported_resolution_budget(self):
        server_args = MagicMock()
        server_args.warmup_steps = 1
        server_args.enable_cfg_parallel = False
        server_args.enable_torch_compile = False
        server_args.num_gpus = 1

        server_args.pipeline_config.task_type = ModelTaskType.T2V
        server_args.pipeline_config.adjust_num_frames.side_effect = lambda value: value

        sampling_defaults = SamplingParams(
            width=1280,
            height=720,
            num_frames=81,
            num_inference_steps=35,
            supported_resolutions=[
                (1280, 720),
                (720, 1280),
                (832, 480),
                (480, 832),
                (1024, 1024),
            ],
        )
        with patch(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            return_value=sampling_defaults,
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=None,
                server_based_warmup=True,
            )

        self.assertEqual((reqs[0].width, reqs[0].height), (832, 480))
        self.assertEqual(reqs[0].num_frames, 17)
        self.assertEqual(reqs[0].num_inference_steps, 2)

    def test_ltx2_two_stage_warmup_uses_pipeline_alignment(self):
        server_args = MagicMock()
        server_args.warmup_steps = 1
        server_args.enable_cfg_parallel = False
        server_args.enable_torch_compile = False
        server_args.num_gpus = 1
        server_args.pipeline_class_name = "LTX2TwoStageHQPipeline"

        server_args.pipeline_config.task_type = ModelTaskType.T2V
        server_args.pipeline_config.vae_scale_factor = 32
        server_args.pipeline_config.vae_config = SimpleNamespace(
            use_temporal_scaling_frames=True,
            arch_config=SimpleNamespace(temporal_compression_ratio=8),
        )
        server_args.pipeline_config.adjust_num_frames.return_value = 25
        server_args.num_gpus = 2

        sampling_defaults = SamplingParams(
            width=1920,
            height=1088,
            num_frames=121,
            num_inference_steps=15,
        )
        with patch(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            return_value=sampling_defaults,
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=None,
                server_based_warmup=True,
            )

        self.assertEqual((reqs[0].width, reqs[0].height), (832, 448))
        self.assertEqual(reqs[0].width % 64, 0)
        self.assertEqual(reqs[0].height % 64, 0)
        self.assertEqual(reqs[0].num_frames, 25)
        server_args.pipeline_config.adjust_num_frames.assert_called_once_with(25)

    def test_ltx2_two_stage_single_gpu_keeps_generic_frame_cap(self):
        server_args = MagicMock()
        server_args.pipeline_class_name = "LTX2TwoStagePipeline"
        server_args.num_gpus = 1
        server_args.pipeline_config.task_type = ModelTaskType.T2V
        server_args.pipeline_config.adjust_num_frames.side_effect = lambda value: value

        num_frames = _resolve_warmup_num_frames(
            server_args,
            SamplingParams(num_frames=121),
            server_based_warmup=True,
        )

        self.assertEqual(num_frames, 17)

    def test_server_based_warmup_uses_representative_image_fallback(self):
        server_args = MagicMock()
        server_args.warmup_steps = 1
        server_args.enable_cfg_parallel = False
        server_args.enable_torch_compile = False

        server_args.pipeline_config.task_type = ModelTaskType.T2I

        with patch(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            return_value=SamplingParams(),
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=None,
                server_based_warmup=True,
            )

        req = reqs[0]
        self.assertEqual(
            (req.width, req.height),
            SERVER_WARMUP_IMAGE_FALLBACK_RESOLUTION,
        )

    def test_warmup_image_inclusion_policy_all_task_types(self):
        server_based_expected = {
            ModelTaskType.T2I: False,
            ModelTaskType.T2V: False,
            ModelTaskType.TI2I: True,
            ModelTaskType.TI2V: True,
            ModelTaskType.I2I: True,
            ModelTaskType.I2V: True,
            ModelTaskType.I2M: True,
            ModelTaskType.VLA_ACTION: False,
        }
        request_based_expected = {
            task_type: task_type.accepts_image_input() for task_type in ModelTaskType
        }
        request_based_expected[ModelTaskType.VLA_ACTION] = False

        for task_type in ModelTaskType:
            server_args = MagicMock()
            server_args.pipeline_config.task_type = task_type

            self.assertEqual(
                should_include_warmup_image(server_args, server_based_warmup=True),
                server_based_expected[task_type],
                task_type.name,
            )
            self.assertEqual(
                should_include_warmup_image(server_args, server_based_warmup=False),
                request_based_expected[task_type],
                task_type.name,
            )

    def test_action_pipeline_skips_synthetic_warmup_before_sampling_defaults(self):
        server_args = MagicMock()
        server_args.pipeline_config.task_type = ModelTaskType.VLA_ACTION

        with (
            patch(
                "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults"
            ) as get_defaults,
            patch(
                "sglang.multimodal_gen.runtime.warmup_request_builder._resolve_default_warmup_resolution"
            ) as resolve_resolution,
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=None,
                server_based_warmup=True,
            )

        self.assertEqual(reqs, [])
        get_defaults.assert_not_called()
        resolve_resolution.assert_not_called()

    def test_action_pipeline_disables_synthetic_warmup(self):
        server_args = MagicMock()
        server_args.warmup_mode = "server"
        server_args.warmup_resolutions = ["512x512"]
        server_args.pipeline_config.task_type = ModelTaskType.VLA_ACTION

        self.assertFalse(supports_synthetic_warmup(server_args))
        self.assertFalse(should_run_synthetic_server_warmup(server_args))
        self.assertFalse(should_run_explicit_client_warmup(server_args))

    def test_mesh_pipeline_builds_image_conditioned_warmup(self):
        server_args = MagicMock()
        server_args.warmup_mode = "server"
        server_args.warmup_steps = 1
        server_args.warmup_resolutions = None
        server_args.enable_cfg_parallel = False
        server_args.enable_torch_compile = False
        server_args.enable_breakable_cuda_graph = False
        server_args.backend = "native"
        server_args.pipeline_class_name = None
        server_args.is_arg_explicitly_set.return_value = False
        server_args.pipeline_config = SimpleNamespace(
            task_type=ModelTaskType.I2M,
            supports_auto_residency=True,
            vae_stride=None,
            vae_scale_factor=None,
            vae_config=None,
        )

        with (
            patch(
                "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
                return_value=SamplingParams(width=512, height=512),
            ),
            patch(
                "sglang.multimodal_gen.runtime.server_warmup.is_realtime_serving",
                return_value=False,
            ),
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=None,
                warmup_input_path="/tmp/warmup.png",
                server_based_warmup=True,
            )
            self.assertTrue(should_run_synthetic_server_warmup(server_args))

        self.assertEqual(len(reqs), 1)
        self.assertEqual(reqs[0].data_type, ModelTaskType.I2M.data_type())
        self.assertEqual(reqs[0].image_path, ["/tmp/warmup.png"])

    def test_server_based_warmup_keeps_ti2i_image_input(self):
        server_args = MagicMock()
        server_args.warmup_steps = 1
        server_args.enable_cfg_parallel = False
        server_args.enable_torch_compile = False
        server_args.pipeline_config.task_type = ModelTaskType.TI2I

        with patch(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            return_value=SamplingParams(width=512, height=512),
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=None,
                warmup_input_path="/tmp/warmup.png",
                server_based_warmup=True,
            )

        self.assertEqual(reqs[0].image_path, ["/tmp/warmup.png"])

    def test_server_based_warmup_keeps_image_input_count(self):
        server_args = MagicMock()
        server_args.warmup_steps = 1
        server_args.enable_cfg_parallel = False
        server_args.enable_torch_compile = False
        server_args.pipeline_config.task_type = ModelTaskType.TI2I

        with patch(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            return_value=SamplingParams(
                width=512,
                height=512,
                image_path=["first.png", "second.png"],
            ),
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=None,
                warmup_input_path="/tmp/warmup.png",
                server_based_warmup=True,
            )

        self.assertEqual(
            reqs[0].image_path,
            ["/tmp/warmup.png", "/tmp/warmup.png"],
        )

    def test_server_based_warmup_keeps_required_image_input(self):
        server_args = MagicMock()
        server_args.warmup_steps = 1
        server_args.enable_cfg_parallel = False
        server_args.enable_torch_compile = False
        server_args.pipeline_config.task_type = ModelTaskType.I2I

        with patch(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            return_value=SamplingParams(width=512, height=512),
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=None,
                warmup_input_path="/tmp/warmup.png",
                server_based_warmup=True,
            )

        self.assertEqual(reqs[0].image_path, ["/tmp/warmup.png"])

    def test_server_based_warmup_keeps_ti2v_image_input(self):
        server_args = MagicMock()
        server_args.warmup_steps = 1
        server_args.enable_cfg_parallel = False
        server_args.enable_torch_compile = False
        server_args.num_gpus = 1
        server_args.pipeline_config.task_type = ModelTaskType.TI2V

        with patch(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            return_value=SamplingParams(width=512, height=512),
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=None,
                warmup_input_path="/tmp/warmup.png",
                server_based_warmup=True,
            )

        self.assertEqual(reqs[0].image_path, ["/tmp/warmup.png"])


class TestFlux2FinetunedVaeEncodePreprocess(unittest.TestCase):
    def test_single_frame_custom_vae_encode_input_is_4d(self):
        config = Flux2FinetunedPipelineConfig()
        vae = MagicMock()
        vae.bn = None

        image = torch.zeros(1, 3, 1, 32, 32)
        output = config.preprocess_vae_encode(image, vae)

        self.assertEqual(tuple(output.shape), (1, 3, 32, 32))

    def test_standard_flux2_vae_encode_input_stays_5d(self):
        config = Flux2FinetunedPipelineConfig()
        vae = MagicMock()
        vae.bn = object()

        image = torch.zeros(1, 3, 1, 32, 32)
        output = config.preprocess_vae_encode(image, vae)

        self.assertIs(output, image)

    def test_custom_vae_already_patchified_encode_latents_stay_128_channels(self):
        config = Flux2FinetunedPipelineConfig()
        config.dit_config.arch_config.in_channels = 128
        vae = MagicMock()
        vae.bn = None

        image_latents = torch.zeros(1, config.dit_config.arch_config.in_channels, 8, 8)
        output = config.postprocess_vae_encode(image_latents, vae)

        self.assertIs(output, image_latents)

    def test_standard_flux2_vae_encode_latents_are_patchified(self):
        config = Flux2FinetunedPipelineConfig()
        vae = MagicMock()
        vae.bn = object()

        image_latents = torch.zeros(1, 32, 8, 8)
        output = config.postprocess_vae_encode(image_latents, vae)

        self.assertEqual(
            tuple(output.shape),
            (1, image_latents.shape[1] * 4, 4, 4),
        )


class TestImageVaeEncodingLatentRetrieval(unittest.TestCase):
    def test_encode_scale_and_shift_allows_missing_shift(self):
        latents = torch.ones(1, 4, 2, 2)
        scaling_factor = torch.full((1, 1, 1, 1), 2.0)

        output = ImageVAEEncodingStage.scale_and_shift_encode_latents(
            latents, scaling_factor, None
        )

        self.assertTrue(torch.equal(output, torch.full_like(latents, 2.0)))

    def test_retrieve_latents_accepts_encoder_output_latent(self):
        stage = object.__new__(ImageVAEEncodingStage)
        latents = torch.zeros(1, 32, 8, 8)
        encoder_output = SimpleNamespace(latent=latents)

        self.assertIs(
            stage.retrieve_latents(encoder_output, sample_mode="argmax"),
            latents,
        )
        self.assertIs(
            stage.retrieve_latents(encoder_output, sample_mode="sample"),
            latents,
        )

    def test_retrieve_latents_accepts_encoder_output_latents(self):
        stage = object.__new__(ImageVAEEncodingStage)
        latents = torch.zeros(1, 32, 8, 8)
        encoder_output = SimpleNamespace(latents=latents)

        self.assertIs(
            stage.retrieve_latents(encoder_output, sample_mode="argmax"),
            latents,
        )
        self.assertIs(
            stage.retrieve_latents(encoder_output, sample_mode="sample"),
            latents,
        )


class TestInputValidationCfgParallelSingleBranch(unittest.TestCase):
    """A request that turns CFG off must still be served under cfg-parallel.

    This used to raise. The guard came from a warmup hang (#23198, 2026-04-23);
    two weeks later the multi-branch refactor (#23736) taught the dispatcher to
    handle a single branch, and the warmup builder grew its own fix (it forces
    CFG on whenever cfg-parallel is enabled). What the guard still did was refuse
    live traffic the runtime could serve -- and because cfg-parallel is
    AUTO-enabled from the model's default sampling params, a plain
    `sglang serve --num-gpus 2` on a CFG-defaulting model rejected every
    guidance_scale=1.0 request, citing a flag the user never passed.

    Both tests patch _generate_seeds (the first statement of
    InputValidationStage.forward) to sidestep its device-lookup / generator
    creation, keeping the suite CPU-only. num_inference_steps must be set because
    the "num_inference_steps <= 0" check raises TypeError on None first.
    """

    def _single_branch_req(self) -> Req:
        # negative_prompt="" (non-None) keeps the negative_prompt-is-None check
        # from firing first, so this isolates the cfg-parallel path.
        return Req(
            prompt="test",
            negative_prompt="",
            guidance_scale=1.0,
            true_cfg_scale=None,
            num_inference_steps=4,
            num_outputs_per_prompt=1,
            width=512,
            height=512,
        )

    def test_input_validation_accepts_cfg_parallel_without_cfg(self):
        req = self._single_branch_req()
        self.assertIs(
            req.do_classifier_free_guidance,
            False,
            "Sanity: the setup must leave do_cfg=False, or this tests nothing.",
        )

        stage = _make_input_validation_stage()
        server_args = _make_validation_server_args(enable_cfg_parallel=True)

        with patch.object(InputValidationStage, "_generate_seeds"):
            try:
                stage.forward(req, server_args)
            except ValueError as e:
                self.fail(
                    "forward() rejected a single-branch request under "
                    f"cfg-parallel; the dispatcher can serve it: {e}"
                )

    def test_input_validation_passes_cfg_parallel_with_cfg(self):
        req = Req(
            prompt="test",
            negative_prompt="bad",
            guidance_scale=4.0,
            true_cfg_scale=4.0,
            num_inference_steps=4,
            num_outputs_per_prompt=1,
            width=512,
            height=512,
        )
        self.assertIs(
            req.do_classifier_free_guidance,
            True,
            "Sanity: req must enable CFG for this positive-case test.",
        )

        stage = _make_input_validation_stage()
        server_args = _make_validation_server_args(enable_cfg_parallel=True)

        with patch.object(InputValidationStage, "_generate_seeds"):
            try:
                stage.forward(req, server_args)
            except ValueError as e:
                self.fail(f"forward() raised ValueError on a valid CFG request: {e}")


class TestCfgParallelServesOneBranch(unittest.TestCase):
    """The property that makes accepting a single-branch request safe.

    Dropping the validation guard is only correct because the dispatcher already
    handles n_branches=1 on a 2-rank CFG group: rank 0 owns the branch, every
    other rank runs it too so the all-gather has shapes, and the reorder step
    hands both ranks the owner's prediction. Pin it from the rank that owns
    nothing -- that is the rank the old comment said returned None and hung a
    gloo broadcast for half an hour.
    """

    def _run_on_rank(self, cfg_rank: int, n_branches: int = 1, world_size: int = 2):
        from sglang.multimodal_gen.runtime.distributed.cfg_policy import (
            CFGBranch,
            CFGPolicy,
        )

        mod = "sglang.multimodal_gen.runtime.distributed.cfg_parallel_utils"
        branches = [CFGBranch(f"b{i}", i == 0, {"tag": i}) for i in range(n_branches)]
        policy = CFGPolicy(branches=branches)
        seen: list[int] = []

        def predict_fn(branch):
            seen.append(branch.kwargs["tag"])
            return torch.full((1, 2), float(branch.kwargs["tag"]))

        # A real 2-rank gather returns one tensor per rank. Both ranks ran the
        # same branch here, so both contributions carry the same values.
        def fake_all_gather(t, dim=0, separate_tensors=False):
            return [t.clone() for _ in range(world_size)]

        with (
            patch(f"{mod}.get_classifier_free_guidance_rank", return_value=cfg_rank),
            patch(
                f"{mod}.get_classifier_free_guidance_world_size",
                return_value=world_size,
            ),
            patch(f"{mod}.get_local_torch_device", return_value=torch.device("cpu")),
            patch(f"{mod}.cfg_model_parallel_all_gather", side_effect=fake_all_gather),
        ):
            from sglang.multimodal_gen.runtime.distributed.cfg_parallel_utils import (
                run_cfg_parallel,
            )

            return run_cfg_parallel(policy, predict_fn), seen

    def test_branch_owner_gets_the_single_prediction(self):
        preds, seen = self._run_on_rank(cfg_rank=0)
        self.assertEqual(len(preds), 1)
        self.assertEqual(seen, [0], "the owning rank runs branch 0 once")
        self.assertTrue(torch.equal(preds[0], torch.zeros(1, 2)))

    def test_rank_without_a_branch_still_returns_the_owners_prediction(self):
        preds, seen = self._run_on_rank(cfg_rank=1)
        self.assertEqual(
            seen,
            [0],
            "the rank that owns no branch must still run one, or the "
            "all-gather has no shapes to work with",
        )
        self.assertEqual(len(preds), 1)
        self.assertIsNotNone(preds[0])
        self.assertTrue(torch.equal(preds[0], torch.zeros(1, 2)))

    def test_two_branches_still_split_across_the_ranks(self):
        from sglang.multimodal_gen.runtime.distributed.cfg_parallel_utils import (
            dispatch_branches,
        )

        self.assertEqual(dispatch_branches(1, 2), [[0], []])
        self.assertEqual(dispatch_branches(2, 2), [[0], [1]])


if __name__ == "__main__":
    unittest.main()
