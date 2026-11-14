"""Batched CFG must preserve branch order, combination, and fallback behavior."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.zimage import ZImagePipelineConfig
from sglang.multimodal_gen.configs.sample.zimage import ZImageSamplingParams
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGBranch
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zimage_denoising import (
    ZImageDenoisingStage,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class StageHarness(ZImageDenoisingStage):
    def __init__(self):
        self._cache_dit_enabled = False
        self.forward_batches = []

    def _predict_noise(self, *, latent_model_input, encoder_hidden_states, **kwargs):
        self.forward_batches.append(latent_model_input.shape[0])
        offsets = torch.stack([caption[0, 0] for caption in encoder_hidden_states])
        return latent_model_input + offsets.reshape(-1, 1, 1, 1, 1)


class TestZImageBatchedCFG(CustomTestCase):
    def setUp(self):
        self.stage = StageHarness()
        topology = patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zimage_denoising.get_sp_world_size",
            return_value=1,
        )
        topology.start()
        self.addCleanup(topology.stop)
        self.config = ZImagePipelineConfig()
        self.batch = Req(sampling_params=ZImageSamplingParams(cfg_normalization=False))
        self.batch.prompt_embeds = [torch.zeros(32, 4)]
        self.batch.negative_prompt_embeds = [torch.zeros(32, 4)]
        self.args = SimpleNamespace(
            enable_cfg_parallel=False,
            enable_torch_compile=False,
            enable_breakable_cuda_graph=False,
            pipeline_config=self.config,
        )
        self.positive = self._branch(True, 2)
        self.negative = self._branch(False, -1)
        self.policy = self.config.cfg_policy.build(
            self.batch, {}, self.positive.kwargs, self.negative.kwargs
        )
        self.kwargs = dict(
            current_model=None,
            latent_model_input=torch.ones(1, 1, 1, 2, 2),
            timestep=torch.tensor([500.0]),
            batch=self.batch,
            timestep_index=0,
            attn_metadata=None,
            target_dtype=torch.float32,
            current_guidance_scale=4.0,
            cfg_policy=self.policy,
            cfg_gate_state=None,
            server_args=self.args,
            guidance=None,
            latents=torch.ones(1, 1, 1, 2, 2),
        )

    def _branch(self, positive, value):
        return CFGBranch(
            "conditional" if positive else "unconditional",
            positive,
            {
                "encoder_hidden_states": [torch.full((32, 4), float(value))],
                "caption_valid_lens": torch.tensor([11 if positive else 8]),
                "freqs_cis": tuple(
                    (torch.zeros(32, 4), torch.ones(32, 4)) for _ in range(2)
                ),
            },
        )

    def test_batched_prediction_matches_serial_arithmetic_and_reuses_tables(self):
        result = self.stage._predict_noise_with_cfg(**self.kwargs)
        torch.testing.assert_close(result, torch.full_like(result, 15), rtol=0, atol=0)
        self.assertEqual(self.stage.forward_batches, [2])
        conditioning = self.batch.extra["zimage_batched_cfg_conditioning"][1]
        self.assertEqual(conditioning["caption_valid_lens"].tolist(), [11, 8])
        self.assertIs(
            conditioning["freqs_cis"], self.policy.branches[0].kwargs["freqs_cis"]
        )
        self.stage._predict_noise_with_cfg(**self.kwargs)
        self.assertIs(
            self.batch.extra["zimage_batched_cfg_conditioning"][1], conditioning
        )
        self.kwargs["cfg_policy"] = self.config.cfg_policy.build(
            self.batch, {}, self._branch(True, 3).kwargs, self.negative.kwargs
        )
        changed = self.stage._predict_noise_with_cfg(**self.kwargs)
        torch.testing.assert_close(
            changed, torch.full_like(changed, 20), rtol=0, atol=0
        )
        self.assertIsNot(
            self.batch.extra["zimage_batched_cfg_conditioning"][1], conditioning
        )

    def test_exact_quality_keeps_serial_predictions(self):
        self.batch.sampling_params.quality = "exact"
        result = self.stage._predict_noise_with_cfg(**self.kwargs)
        torch.testing.assert_close(result, torch.full_like(result, 15), rtol=0, atol=0)
        self.assertEqual(self.stage.forward_batches, [1, 1])
        self.assertNotIn("zimage_batched_cfg_conditioning", self.batch.extra)

    def test_request_conditioning_cleanup_on_success_and_failure(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
            DenoisingStage,
        )

        for error in (None, RuntimeError("generation failed")):
            self.stage._predict_noise_with_cfg(**self.kwargs)
            with patch.object(
                DenoisingStage, "forward", return_value=self.batch, side_effect=error
            ):
                if error is None:
                    self.assertIs(self.stage.forward(self.batch, self.args), self.batch)
                else:
                    with self.assertRaisesRegex(RuntimeError, "generation failed"):
                        self.stage.forward(self.batch, self.args)
            self.assertNotIn("zimage_batched_cfg_conditioning", self.batch.extra)

    def test_sp_conditioning_stacks_branch_rope(self):
        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zimage_denoising.get_sp_world_size",
            return_value=8,
        ):
            self.stage._predict_noise_with_cfg(**self.kwargs)
        conditioning = self.batch.extra["zimage_batched_cfg_conditioning"][1]
        self.assertEqual(conditioning["freqs_cis"][0][0].shape, (2, 32, 4))

    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args",
        return_value=None,
    )
    def test_shared_factory_preserves_default_and_custom_stage_routing(
        self, config_boundary
    ):
        from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
            ComposedPipelineBase,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
            DenoisingStage,
        )

        class FactoryHarness:
            def __init__(self):
                self.modules = {
                    "transformer": object(),
                    "scheduler": object(),
                    "vae": object(),
                }
                self.factory = None

            def get_module(self, name, default=None):
                return self.modules.get(name, default)

            def add_stage_factory(self, role, factory, name):
                self.factory = factory
                return self

        class FakeStage:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def forward(self, batch, args):
                return batch

        for selected in (None, FakeStage):
            pipeline = FactoryHarness()
            kwargs = {} if selected is None else {"standard_stage_cls": selected}
            if selected is None:
                with patch.object(DenoisingStage, "__init__", return_value=None):
                    ComposedPipelineBase.add_progressive_denoising_stage(
                        pipeline, FakeStage, **kwargs
                    )
                    router = pipeline.factory()
                self.assertIsInstance(router.standard_stage, DenoisingStage)
            else:
                ComposedPipelineBase.add_progressive_denoising_stage(
                    pipeline, FakeStage, **kwargs
                )
                router = pipeline.factory()
                self.assertIsInstance(router.standard_stage, FakeStage)
                self.assertNotIn("transformer_2", router.standard_stage.kwargs)
                self.assertIs(
                    router.standard_stage.kwargs["vae"], pipeline.modules["vae"]
                )
                self.assertIs(
                    router.forward(
                        SimpleNamespace(progressive_mode="fullres"), None
                    ).__class__,
                    SimpleNamespace,
                )
            self.assertIsNone(router._progressive_stage)
            progressive = router._get_progressive_stage()
            self.assertIsInstance(progressive, FakeStage)
            self.assertIs(
                progressive.kwargs["transformer"], pipeline.modules["transformer"]
            )

    def test_unsupported_modes_keep_serial_predictions(self):
        for flag in ("enable_torch_compile", "enable_breakable_cuda_graph"):
            setattr(self.args, flag, True)
            self.stage.forward_batches.clear()
            result = self.stage._predict_noise_with_cfg(**self.kwargs)
            torch.testing.assert_close(
                result, torch.full_like(result, 15), rtol=0, atol=0
            )
            self.assertEqual(self.stage.forward_batches, [1, 1])
            setattr(self.args, flag, False)
        self.policy.branches[1].kwargs["freqs_cis"] = tuple(
            (torch.zeros(64, 4), torch.ones(64, 4)) for _ in range(2)
        )
        self.stage.forward_batches.clear()
        result = self.stage._predict_noise_with_cfg(**self.kwargs)
        torch.testing.assert_close(result, torch.full_like(result, 15), rtol=0, atol=0)
        self.assertEqual(self.stage.forward_batches, [1, 1])


if __name__ == "__main__":
    unittest.main()
