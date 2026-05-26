import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)


class _PipelineConfig:
    def get_classifier_free_guidance_scale(self, batch, current_guidance_scale):
        return current_guidance_scale

    def slice_noise_pred(self, noise_pred, latents):
        return noise_pred

    def postprocess_cfg_noise(self, batch, noise_pred, noise_pred_cond):
        return noise_pred


class TestCFGGating(unittest.TestCase):
    def _make_server_args(self, enable_cfg_parallel=False):
        return SimpleNamespace(
            enable_cfg_parallel=enable_cfg_parallel,
            pipeline_config=_PipelineConfig(),
        )

    def _make_batch(self):
        return SimpleNamespace(
            cfg_normalization=0,
            guidance_rescale=0,
            do_classifier_free_guidance=True,
            is_cfg_negative=False,
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

    def test_rejects_invalid_gate_fraction(self):
        stage = DenoisingStage.__new__(DenoisingStage)
        ctx = SimpleNamespace(timesteps=torch.arange(10), extra={}, is_warmup=True)
        batch = self._make_batch()
        server_args = self._make_server_args()

        with patch.dict(os.environ, {"SGLANG_DIFFUSION_CFG_GATE_STEP": "1.5"}):
            with self.assertRaises(ValueError):
                stage._init_cfg_gate_state(ctx, batch, server_args)


if __name__ == "__main__":
    unittest.main()
