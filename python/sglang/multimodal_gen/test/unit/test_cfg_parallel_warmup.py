"""Unit tests for the --enable-cfg-parallel warmup fix and guard.

Covers three code paths introduced alongside this file:
- Scheduler.prepare_server_warmup_reqs synthesizes warmup Reqs that
  actually enable classifier-free guidance when cfg-parallel is on.
- InputValidationStage.forward rejects non-CFG requests when the server
  has cfg-parallel on.
- Server-based warmup can opt into model-default negative prompts so warmup
  populates the negative text embedding cache.

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
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageVAEEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.server_warmup import (
    DEFAULT_LIGHTWEIGHT_IMAGE_RESOLUTION,
    DEFAULT_PLACEHOLDER_PROMPT,
    build_warmup_reqs,
    should_include_warmup_image,
)


def _make_bare_scheduler(enable_cfg_parallel: bool) -> Scheduler:
    """
    Build a minimal Scheduler without calling __init__ (which requires
    distributed init, ZMQ sockets, pipeline load, etc.). Populates only
    the attributes prepare_server_warmup_reqs reads/writes for a
    text-only task so _prepare_shared_warmup_image_path is skipped.
    """
    scheduler = object.__new__(Scheduler)

    server_args = MagicMock()
    server_args.warmup = True
    server_args.warmup_steps = 1
    server_args.warmup_resolutions = ["512x512"]
    server_args.enable_cfg_parallel = enable_cfg_parallel

    # Text-only task — accepts_image_input() False skips the image-path
    # branch entirely, so we don't need to mock
    # _prepare_shared_warmup_image_path.
    task_type = MagicMock()
    task_type.requires_image_input.return_value = False
    task_type.accepts_image_input.return_value = False
    task_type.data_type.return_value = ModelTaskType.T2I.data_type()
    server_args.pipeline_config.task_type = task_type

    scheduler.server_args = server_args
    scheduler.warmed_up = False
    scheduler.waiting_queue = deque()
    return scheduler


def _make_input_validation_stage() -> InputValidationStage:
    return InputValidationStage()


def _make_validation_server_args(enable_cfg_parallel: bool) -> MagicMock:
    sa = MagicMock()
    sa.enable_cfg_parallel = enable_cfg_parallel
    sa.pipeline_config.task_type = ModelTaskType.T2I
    return sa


class TestWarmupReqCfgParallel(unittest.TestCase):
    """Commit 1 regression: prepare_server_warmup_reqs."""

    def test_warmup_req_cfg_parallel_sets_do_cfg(self):
        scheduler = _make_bare_scheduler(enable_cfg_parallel=True)
        scheduler.prepare_server_warmup_reqs()

        self.assertEqual(len(scheduler.waiting_queue), 1)
        _, req, _ = scheduler.waiting_queue[0]
        self.assertIs(req.do_classifier_free_guidance, True)
        self.assertEqual(req.negative_prompt, DEFAULT_PLACEHOLDER_PROMPT)

    def test_warmup_req_no_cfg_parallel_unchanged(self):
        # Regression guard: the cfg-parallel=on fix must not bleed into
        # the cfg-parallel=off path. Key invariant is do_cfg stays False
        # AND the synthesized Req is not using the cfg-parallel-specific
        # "warmup" placeholder for negative_prompt (which would indicate
        # the fix's kwargs leaked into this branch).
        scheduler = _make_bare_scheduler(enable_cfg_parallel=False)
        scheduler.prepare_server_warmup_reqs()

        self.assertEqual(len(scheduler.waiting_queue), 1)
        _, req, _ = scheduler.waiting_queue[0]
        self.assertIs(req.do_classifier_free_guidance, False)
        self.assertNotEqual(req.negative_prompt, DEFAULT_PLACEHOLDER_PROMPT)

    def test_server_based_warmup_uses_model_default_negative_prompt(self):
        server_args = MagicMock()
        server_args.warmup_steps = 1
        server_args.enable_cfg_parallel = False

        task_type = MagicMock()
        task_type.requires_image_input.return_value = False
        task_type.accepts_image_input.return_value = False
        task_type.is_image_gen.return_value = True
        task_type.data_type.return_value = ModelTaskType.T2I.data_type()
        server_args.pipeline_config.task_type = task_type

        sampling_defaults = SamplingParams(
            negative_prompt="model default negative",
            guidance_scale=4.0,
            num_inference_steps=20,
        )
        with patch(
            "sglang.multimodal_gen.runtime.server_warmup.get_model_sampling_defaults",
            return_value=sampling_defaults,
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=None,
                use_model_sampling_defaults=True,
                return_warmup_result=True,
                server_based_warmup=True,
            )

        self.assertEqual(len(reqs), 1)
        req = reqs[0]
        self.assertTrue(req.is_warmup)
        self.assertEqual(req.negative_prompt, "model default negative")
        self.assertIs(req.do_classifier_free_guidance, True)
        self.assertTrue(req.extra["return_warmup_result"])
        self.assertTrue(req.extra["server_based_warmup"])

    def test_server_based_warmup_uses_model_default_resolution(self):
        server_args = MagicMock()
        server_args.warmup_steps = 1
        server_args.enable_cfg_parallel = False

        task_type = MagicMock()
        task_type.requires_image_input.return_value = False
        task_type.accepts_image_input.return_value = False
        task_type.is_image_gen.return_value = True
        task_type.data_type.return_value = ModelTaskType.T2I.data_type()
        server_args.pipeline_config.task_type = task_type

        sampling_defaults = SamplingParams(width=640, height=640)
        with patch(
            "sglang.multimodal_gen.runtime.server_warmup.get_model_sampling_defaults",
            return_value=sampling_defaults,
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=None,
                use_model_sampling_defaults=True,
                server_based_warmup=True,
            )

        req = reqs[0]
        self.assertEqual(req.width, 640)
        self.assertEqual(req.height, 640)

    def test_server_based_warmup_keeps_lightweight_image_fallback(self):
        server_args = MagicMock()
        server_args.warmup_steps = 1
        server_args.enable_cfg_parallel = False

        task_type = MagicMock()
        task_type.requires_image_input.return_value = False
        task_type.accepts_image_input.return_value = False
        task_type.is_image_gen.return_value = True
        task_type.data_type.return_value = ModelTaskType.T2I.data_type()
        server_args.pipeline_config.task_type = task_type

        with patch(
            "sglang.multimodal_gen.runtime.server_warmup.get_model_sampling_defaults",
            return_value=SamplingParams(),
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=None,
                use_model_sampling_defaults=True,
                server_based_warmup=True,
            )

        req = reqs[0]
        self.assertEqual(
            (req.width, req.height),
            DEFAULT_LIGHTWEIGHT_IMAGE_RESOLUTION,
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
        }

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
                task_type.accepts_image_input(),
                task_type.name,
            )

    def test_server_based_warmup_keeps_ti2i_image_input(self):
        server_args = MagicMock()
        server_args.warmup_steps = 1
        server_args.enable_cfg_parallel = False
        server_args.pipeline_config.task_type = ModelTaskType.TI2I

        with patch(
            "sglang.multimodal_gen.runtime.server_warmup.get_model_sampling_defaults",
            return_value=SamplingParams(width=512, height=512),
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=None,
                warmup_input_path="/tmp/warmup.png",
                use_model_sampling_defaults=True,
                server_based_warmup=True,
            )

        self.assertEqual(reqs[0].image_path, ["/tmp/warmup.png"])

    def test_server_based_warmup_keeps_required_image_input(self):
        server_args = MagicMock()
        server_args.warmup_steps = 1
        server_args.enable_cfg_parallel = False
        server_args.pipeline_config.task_type = ModelTaskType.I2I

        with patch(
            "sglang.multimodal_gen.runtime.server_warmup.get_model_sampling_defaults",
            return_value=SamplingParams(width=512, height=512),
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=None,
                warmup_input_path="/tmp/warmup.png",
                use_model_sampling_defaults=True,
                server_based_warmup=True,
            )

        self.assertEqual(reqs[0].image_path, ["/tmp/warmup.png"])

    def test_server_based_warmup_keeps_ti2v_image_input(self):
        server_args = MagicMock()
        server_args.warmup_steps = 1
        server_args.enable_cfg_parallel = False
        server_args.pipeline_config.task_type = ModelTaskType.TI2V

        with patch(
            "sglang.multimodal_gen.runtime.server_warmup.get_model_sampling_defaults",
            return_value=SamplingParams(width=512, height=512),
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=None,
                warmup_input_path="/tmp/warmup.png",
                use_model_sampling_defaults=True,
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


class TestInputValidationCfgParallelGuard(unittest.TestCase):
    """Commit 2: per-request cfg-parallel check.

    Both tests patch _generate_seeds (the first statement of
    InputValidationStage.forward, input_validation.py:274) to sidestep
    its device-lookup / generator-creation code which pulls in torch
    CUDA bindings — keeps the suite strictly CPU-only. We still need
    num_inference_steps on the Req because the stage's
    "num_inference_steps <= 0" check at L305-308 raises TypeError on
    None before the new commit-2 check is reached.
    """

    def test_input_validation_rejects_cfg_parallel_without_cfg(self):
        # negative_prompt="" (non-None) ensures the existing
        # negative_prompt-is-None check at input_validation.py:295-298
        # does NOT fire first — this isolates the new commit-2 check.
        # width/height/num_outputs_per_prompt pre-set so the stage's
        # default-dimension block at L352-361 doesn't mutate the Req
        # in a way that obscures the assertion target.
        req = Req(
            prompt="test",
            negative_prompt="",
            guidance_scale=1.0,
            true_cfg_scale=None,
            num_inference_steps=4,
            num_outputs_per_prompt=1,
            width=512,
            height=512,
        )
        self.assertIs(
            req.do_classifier_free_guidance,
            False,
            "Sanity: test setup must leave do_cfg=False so the "
            "commit-2 check is the one that fires, not an upstream check.",
        )

        stage = _make_input_validation_stage()
        server_args = _make_validation_server_args(enable_cfg_parallel=True)

        with patch.object(InputValidationStage, "_generate_seeds"):
            with self.assertRaises(ValueError) as ctx:
                stage.forward(req, server_args)

        msg = str(ctx.exception).lower()
        self.assertIn("cfg-parallel", msg)
        for field in (
            "do_classifier_free_guidance",
            "guidance_scale",
            "true_cfg_scale",
            "negative_prompt",
        ):
            self.assertIn(field, str(ctx.exception))

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


if __name__ == "__main__":
    unittest.main()
