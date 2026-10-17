import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.sample.spectrum import SpectrumParams
from sglang.multimodal_gen.runtime.cache.spectrum import SpectrumMixin
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.test.test_utils import CustomTestCase


class _PipelineConfig:
    def get_classifier_free_guidance_scale(self, batch, current_guidance_scale):
        return current_guidance_scale

    def slice_noise_pred(self, noise_pred, latents):
        return noise_pred

    def postprocess_cfg_noise(self, batch, noise_pred, noise_pred_cond):
        return noise_pred


class TestCFGGating(CustomTestCase):
    def _make_server_args(self, enable_cfg_parallel=False):
        return SimpleNamespace(
            enable_cfg_parallel=enable_cfg_parallel,
            pipeline_config=_PipelineConfig(),
        )

    def _make_batch(self, cfg_gate_step=None):
        return SimpleNamespace(
            cfg_normalization=0,
            guidance_rescale=0,
            do_classifier_free_guidance=True,
            is_cfg_negative=False,
            sampling_params=SimpleNamespace(cfg_gate_step=cfg_gate_step),
        )

    def _make_gate_state(self, gate_step=1, model_id=None, delta=None):
        return {
            "fraction": 0.5,
            "requested": True,
            "active": True,
            "gate_step": gate_step,
            "delta": delta,
            "model_id": model_id,
            "fresh_uncond": 0,
            "reused": 0,
            "invalidations": 0,
        }

    def test_reuses_unconditional_residual_after_gate_step(self):
        stage = DenoisingStage.__new__(DenoisingStage)
        batch = self._make_batch()
        server_args = self._make_server_args()
        policy = CFGPolicy().build(batch, {}, {}, {})
        calls = []

        def fake_predict_noise(**kwargs):
            calls.append("neg" if batch.is_cfg_negative else "pos")
            timestep = kwargs["timestep"]
            timestep_value = float(timestep.item())
            offset = 0.25 if batch.is_cfg_negative else 1.25
            return torch.tensor([timestep_value + offset])

        stage._predict_noise = fake_predict_noise
        model = torch.nn.Identity()
        latents = torch.zeros(1)
        state = self._make_gate_state(gate_step=1)

        first = stage._predict_noise_with_cfg(
            current_model=model,
            latent_model_input=latents,
            timestep=torch.tensor(0),
            batch=batch,
            timestep_index=0,
            attn_metadata=None,
            target_dtype=torch.float32,
            current_guidance_scale=4.0,
            cfg_policy=policy,
            cfg_gate_state=state,
            server_args=server_args,
            guidance=None,
            latents=latents,
        )
        second = stage._predict_noise_with_cfg(
            current_model=model,
            latent_model_input=latents,
            timestep=torch.tensor(1),
            batch=batch,
            timestep_index=1,
            attn_metadata=None,
            target_dtype=torch.float32,
            current_guidance_scale=4.0,
            cfg_policy=policy,
            cfg_gate_state=state,
            server_args=server_args,
            guidance=None,
            latents=latents,
        )

        self.assertTrue(torch.equal(first, torch.tensor([4.25])))
        self.assertTrue(torch.equal(second, torch.tensor([5.25])))
        self.assertEqual(calls, ["pos", "neg", "pos"])
        self.assertEqual(state["fresh_uncond"], 1)
        self.assertEqual(state["reused"], 1)
        self.assertEqual(state["invalidations"], 0)

    def test_spectrum_forecasts_cannot_be_saved_as_gating_residuals(self):
        """A scheduled Spectrum skip at the freeze step must run both branches."""

        class TinyModel(SpectrumMixin):
            def __init__(self):
                self.prefix = "flux"
                self._init_spectrum_state()

        stage = DenoisingStage.__new__(DenoisingStage)
        model = TinyModel()
        batch = self._make_batch()
        batch.enable_spectrum = True
        batch.num_inference_steps = 4
        batch.debug = False
        batch.spectrum_params = SpectrumParams(
            warmup_steps=1, window_size=2, flex_window=0, m=1
        )
        server_args = self._make_server_args()
        policy = CFGPolicy().build(batch, {}, {}, {})
        state = self._make_gate_state(gate_step=2)
        calls = []

        def predict(**kwargs):
            ctx = get_forward_context()
            real = model.begin_spectrum_step()
            calls.append((ctx.current_timestep, batch.is_cfg_negative, real))
            value = (
                1 - ctx.current_timestep
                if batch.is_cfg_negative
                else 5 + ctx.current_timestep
            )
            features = torch.tensor([float(value)])
            if real:
                model.spectrum_record_features(features)
                return features
            return model.spectrum_predict_features(features)

        stage._predict_noise = predict
        with patch(
            "sglang.multimodal_gen.runtime.server_args.get_global_server_args",
            return_value=server_args,
        ):
            for step in range(3):
                stage._predict_noise_with_cfg(
                    current_model=model,
                    latent_model_input=torch.zeros(1),
                    timestep=torch.tensor(step),
                    batch=batch,
                    timestep_index=step,
                    attn_metadata=None,
                    target_dtype=torch.float32,
                    current_guidance_scale=4.0,
                    cfg_policy=policy,
                    cfg_gate_state=state,
                    server_args=server_args,
                    guidance=None,
                    latents=torch.zeros(1),
                )
        self.assertEqual(
            calls[:4],
            [(0, False, True), (0, True, True), (1, False, True), (1, True, True)],
        )
        torch.testing.assert_close(state["delta"][0], torch.tensor([6.0]))
        self.assertEqual(state["reused"], 1)
        self.assertEqual(len(calls), 5)
        for predictor in (
            model.spectrum_forecaster,
            model.spectrum_forecaster_negative,
        ):
            self.assertEqual(predictor.cheb._count, 2)
            torch.testing.assert_close(
                predictor.cheb.t_buf[:2], torch.tensor([0.0, 1.0])
            )

    def test_model_switch_invalidates_cached_delta(self):
        stage = DenoisingStage.__new__(DenoisingStage)
        batch = self._make_batch()
        server_args = self._make_server_args()
        policy = CFGPolicy().build(batch, {}, {}, {})
        calls = []

        def fake_predict_noise(**kwargs):
            calls.append("neg" if batch.is_cfg_negative else "pos")
            value = 3.0 if batch.is_cfg_negative else 10.0
            return torch.tensor([value])

        stage._predict_noise = fake_predict_noise
        old_model = torch.nn.Identity()
        new_model = torch.nn.Identity()
        latents = torch.zeros(1)
        state = self._make_gate_state(
            gate_step=0,
            model_id=id(old_model),
            delta=(torch.tensor([2.0]),),
        )

        output = stage._predict_noise_with_cfg(
            current_model=new_model,
            latent_model_input=latents,
            timestep=torch.tensor(2),
            batch=batch,
            timestep_index=2,
            attn_metadata=None,
            target_dtype=torch.float32,
            current_guidance_scale=2.0,
            cfg_policy=policy,
            cfg_gate_state=state,
            server_args=server_args,
            guidance=None,
            latents=latents,
        )

        self.assertTrue(torch.equal(output, torch.tensor([17.0])))
        self.assertEqual(calls, ["pos", "neg"])
        self.assertEqual(state["model_id"], id(new_model))
        self.assertEqual(state["fresh_uncond"], 1)
        self.assertEqual(state["reused"], 0)
        self.assertEqual(state["invalidations"], 1)

    def test_cfg_parallel_disables_gate_state(self):
        stage = DenoisingStage.__new__(DenoisingStage)
        ctx = SimpleNamespace(timesteps=torch.arange(10), extra={}, is_warmup=True)
        batch = self._make_batch()
        server_args = self._make_server_args(enable_cfg_parallel=True)

        with patch.dict(os.environ, {"SGLANG_DIFFUSION_CFG_GATE_STEP": "0.5"}):
            stage._init_cfg_gate_state(ctx, batch, server_args)

        self.assertTrue(ctx.extra["cfg_gate_state"]["requested"])
        self.assertFalse(ctx.extra["cfg_gate_state"]["active"])
        self.assertEqual(ctx.extra["cfg_gate_state"]["gate_step"], 11)

    def test_flux_gate_quality_bound_preserves_explicit_disable_and_parallel(self):
        from sglang.multimodal_gen.configs.pipeline_configs.flux import (
            Flux2PipelineConfig,
            FluxPipelineConfig,
        )

        stage = DenoisingStage.__new__(DenoisingStage)
        for steps in (20, 50):
            for parallel in (False, True):
                for fraction in (0.3, 1.0):
                    with self.subTest(
                        steps=steps, parallel=parallel, fraction=fraction
                    ):
                        ctx = SimpleNamespace(
                            timesteps=torch.arange(steps), extra={}, is_warmup=True
                        )
                        args = self._make_server_args(parallel)
                        args.pipeline_config = FluxPipelineConfig()
                        stage._init_cfg_gate_state(
                            ctx, self._make_batch(fraction), args
                        )
                        state = ctx.extra["cfg_gate_state"]
                        active = fraction < 1 and not parallel
                        self.assertEqual(state["active"], active)
                        self.assertEqual(
                            state["gate_step"], steps - 1 if active else steps + 1
                        )
        self.assertIsNone(Flux2PipelineConfig().cfg_gate_max_reused_steps)

    def test_wan_gate_reuses_at_most_five_steps(self):
        from sglang.multimodal_gen.configs.pipeline_configs.wan import (
            Wan2_2_T2V_A14B_Config,
            WanI2V480PConfig,
            WanT2V480PConfig,
        )

        stage = DenoisingStage.__new__(DenoisingStage)
        for steps in (20, 50):
            for parallel in (False, True):
                # 0.3 is capped; 0.95 already reuses fewer than five steps.
                for fraction in (0.3, 0.95, 1.0):
                    with self.subTest(
                        steps=steps, parallel=parallel, fraction=fraction
                    ):
                        ctx = SimpleNamespace(
                            timesteps=torch.arange(steps), extra={}, is_warmup=True
                        )
                        args = self._make_server_args(parallel)
                        args.pipeline_config = WanT2V480PConfig()
                        stage._init_cfg_gate_state(
                            ctx, self._make_batch(fraction), args
                        )
                        state = ctx.extra["cfg_gate_state"]
                        active = fraction < 1 and not parallel
                        self.assertEqual(state["active"], active)
                        expected = (
                            max(int(steps * fraction), steps - 5)
                            if active
                            else steps + 1
                        )
                        self.assertEqual(state["gate_step"], expected)
        for config in (WanI2V480PConfig(), Wan2_2_T2V_A14B_Config()):
            self.assertEqual(config.cfg_gate_max_reused_steps, 5)

    def test_rejects_invalid_gate_fraction(self):
        stage = DenoisingStage.__new__(DenoisingStage)
        ctx = SimpleNamespace(timesteps=torch.arange(10), extra={}, is_warmup=True)
        batch = self._make_batch()
        server_args = self._make_server_args()

        with patch.dict(os.environ, {"SGLANG_DIFFUSION_CFG_GATE_STEP": "1.5"}):
            with self.assertRaises(ValueError):
                stage._init_cfg_gate_state(ctx, batch, server_args)

    def test_request_fraction_overrides_env_default(self):
        stage = DenoisingStage.__new__(DenoisingStage)
        ctx = SimpleNamespace(timesteps=torch.arange(10), extra={}, is_warmup=True)
        batch = self._make_batch(cfg_gate_step=0.5)
        server_args = self._make_server_args()

        # env default is 1.0 (off); the request opts in.
        stage._init_cfg_gate_state(ctx, batch, server_args)

        state = ctx.extra["cfg_gate_state"]
        self.assertTrue(state["requested"])
        self.assertTrue(state["active"])
        self.assertEqual(state["gate_step"], 5)

    def test_request_fraction_disables_env_default(self):
        stage = DenoisingStage.__new__(DenoisingStage)
        ctx = SimpleNamespace(timesteps=torch.arange(10), extra={}, is_warmup=True)
        batch = self._make_batch(cfg_gate_step=1.0)
        server_args = self._make_server_args()

        with patch.dict(os.environ, {"SGLANG_DIFFUSION_CFG_GATE_STEP": "0.5"}):
            stage._init_cfg_gate_state(ctx, batch, server_args)

        state = ctx.extra["cfg_gate_state"]
        self.assertFalse(state["requested"])
        self.assertFalse(state["active"])

    def test_rejects_invalid_request_fraction(self):
        stage = DenoisingStage.__new__(DenoisingStage)
        ctx = SimpleNamespace(timesteps=torch.arange(10), extra={}, is_warmup=True)
        batch = self._make_batch(cfg_gate_step=1.5)
        server_args = self._make_server_args()

        with self.assertRaises(ValueError):
            stage._init_cfg_gate_state(ctx, batch, server_args)


if __name__ == "__main__":
    unittest.main()
