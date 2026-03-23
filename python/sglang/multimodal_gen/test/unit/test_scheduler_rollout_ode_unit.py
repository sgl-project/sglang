import math
import types
import unittest

import torch

import sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin as rl_mixin_module
from sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin import SchedulerRLMixin


class _DummyScheduler(SchedulerRLMixin):
    def __init__(self):
        self.sigmas = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0], dtype=torch.float32)


class TestSchedulerRolloutOdeUnit(unittest.TestCase):
    def setUp(self):
        self._orig_get_sp_world_size = rl_mixin_module.get_sp_world_size
        rl_mixin_module.get_sp_world_size = lambda: 1

    def tearDown(self):
        rl_mixin_module.get_sp_world_size = self._orig_get_sp_world_size

    def _build_batch(self, *, debug_mode: bool) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            rollout_log_prob_no_const=True,
            rollout_noise_level=0.5,
            rollout_sde_type="ode",
            rollout_debug_mode=debug_mode,
            latents=torch.zeros(2, 4, 8, 8, dtype=torch.float32),
            _rollout_session_data=None,
        )

    def test_ode_step_does_not_call_variance_noise_sampler(self):
        scheduler = _DummyScheduler()
        batch = self._build_batch(debug_mode=False)
        scheduler.prepare_rollout(batch)

        def _raise_if_called(*args, **kwargs):
            raise AssertionError("ODE path should not call _rollout_variance_noise")

        scheduler._rollout_variance_noise = _raise_if_called  # type: ignore[method-assign]

        sample = torch.randn(2, 4, 8, 8, dtype=torch.float32)
        model_output = torch.randn_like(sample)
        current_sigma = torch.tensor(0.6, dtype=torch.float32)
        next_sigma = torch.tensor(0.4, dtype=torch.float32)

        prev_sample, log_prob_local_sum, local_elem_count = scheduler.flow_sde_sampling(
            batch,
            model_output=model_output,
            sample=sample,
            current_sigma=current_sigma,
            next_sigma=next_sigma,
            generator=torch.Generator(device=sample.device).manual_seed(1),
        )

        expected_prev = sample + (next_sigma - current_sigma) * model_output
        self.assertTrue(torch.allclose(prev_sample, expected_prev, atol=1e-6, rtol=0.0))
        self.assertTrue(torch.allclose(log_prob_local_sum, torch.zeros_like(log_prob_local_sum)))
        self.assertEqual(tuple(log_prob_local_sum.shape), (sample.shape[0],))
        self.assertEqual(tuple(local_elem_count.shape), (sample.shape[0],))
        self.assertTrue(torch.all(local_elem_count == float(sample[0].numel())))

    def test_ode_debug_tensors_have_shape_safe_noise_std(self):
        scheduler = _DummyScheduler()
        batch = self._build_batch(debug_mode=True)
        scheduler.prepare_rollout(batch)

        sample = torch.randn(2, 4, 8, 8, dtype=torch.float32)
        model_output = torch.randn_like(sample)
        current_sigma = torch.tensor(0.6, dtype=torch.float32)
        next_sigma = torch.tensor(0.4, dtype=torch.float32)

        scheduler.flow_sde_sampling(
            batch,
            model_output=model_output,
            sample=sample,
            current_sigma=current_sigma,
            next_sigma=next_sigma,
            generator=torch.Generator(device=sample.device).manual_seed(2),
        )

        (
            variance_noises,
            prev_sample_means,
            noise_std_devs,
            model_outputs,
        ) = scheduler.consume_local_rollout_debug_tensors(batch)

        # [B, T, ...] with one step in this test.
        self.assertEqual(tuple(variance_noises.shape), (2, 1, 4, 8, 8))
        self.assertEqual(tuple(prev_sample_means.shape), (2, 1, 4, 8, 8))
        self.assertEqual(tuple(model_outputs.shape), (2, 1, 4, 8, 8))
        self.assertEqual(tuple(noise_std_devs.shape), (2, 1, 1))
        self.assertTrue(torch.allclose(noise_std_devs, torch.zeros_like(noise_std_devs)))
        self.assertTrue(torch.allclose(variance_noises, torch.zeros_like(variance_noises)))


def _flowgrpo_reference_one_step(
    *,
    model_output: torch.Tensor,
    sample: torch.Tensor,
    variance_noise: torch.Tensor,
    current_sigma: torch.Tensor,
    next_sigma: torch.Tensor,
    noise_level: float,
    sde_type: str,
    sigma_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference one-step update from FlowGRPO formulas."""
    dt = next_sigma - current_sigma

    if sde_type == "sde":
        std_dev_t = torch.sqrt(
            current_sigma
            / (
                1
                - torch.where(
                    torch.isclose(current_sigma, current_sigma.new_tensor(1.0)),
                    current_sigma.new_tensor(sigma_max),
                    current_sigma,
                )
            )
        ) * noise_level
        noise_std_dev = std_dev_t * torch.sqrt(-1 * dt)
        prev_sample_mean = (
            sample * (1 + std_dev_t**2 / (2 * current_sigma) * dt)
            + model_output
            * (1 + std_dev_t**2 * (1 - current_sigma) / (2 * current_sigma))
            * dt
        )
    elif sde_type == "cps":
        std_dev_t = next_sigma * math.sin(noise_level * math.pi / 2)
        noise_std_dev = std_dev_t
        pred_original_sample = sample - current_sigma * model_output
        noise_estimate = sample + model_output * (1 - current_sigma)
        prev_sample_mean = pred_original_sample * (1 - next_sigma) + noise_estimate * torch.sqrt(
            next_sigma**2 - std_dev_t**2
        )
    else:
        raise ValueError(f"Unsupported sde_type: {sde_type}")

    prev_sample = prev_sample_mean + noise_std_dev * variance_noise
    return prev_sample, prev_sample_mean, noise_std_dev


class TestSchedulerFlowGRPOStepAlignmentUnit(unittest.TestCase):
    def setUp(self):
        self._orig_get_sp_world_size = rl_mixin_module.get_sp_world_size
        rl_mixin_module.get_sp_world_size = lambda: 1

    def tearDown(self):
        rl_mixin_module.get_sp_world_size = self._orig_get_sp_world_size

    def _build_batch(self, *, sde_type: str, shape: tuple[int, ...]) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            rollout_log_prob_no_const=True,
            rollout_noise_level=0.5,
            rollout_sde_type=sde_type,
            rollout_debug_mode=True,
            latents=torch.empty(shape, dtype=torch.float32),
            _rollout_session_data=None,
        )

    def test_single_step_matches_flowgrpo_reference(self):
        scheduler = _DummyScheduler()
        current_sigma = torch.tensor(0.5, dtype=torch.float32)
        next_sigma = torch.tensor(0.3, dtype=torch.float32)
        shape = (1, 16, 1, 32, 32)
        atol = 1e-3
        pipeline_config = types.SimpleNamespace(
            shard_latents_for_sp=lambda _batch, latents: (latents, False)
        )

        for sde_type in ("sde", "cps"):
            for seed in (0, 1, 2, 3):
                batch = self._build_batch(sde_type=sde_type, shape=shape)
                scheduler.release_rollout_resources(batch)
                scheduler.prepare_rollout(batch=batch, pipeline_config=pipeline_config)

                g = torch.Generator(device="cpu").manual_seed(seed)
                model_output = torch.randn(shape, generator=g, dtype=torch.float32)
                sample = torch.randn(shape, generator=g, dtype=torch.float32)
                variance_noise = torch.randn(shape, generator=g, dtype=torch.float32)
                scheduler._rollout_variance_noise = (  # type: ignore[method-assign]
                    lambda _batch, *_args, **_kwargs: variance_noise
                )

                prev_sgl, _, _ = scheduler.flow_sde_sampling(
                    batch,
                    model_output=model_output,
                    sample=sample,
                    current_sigma=current_sigma,
                    next_sigma=next_sigma,
                    generator=g,
                )
                (
                    _variance_noises,
                    prev_sample_means,
                    noise_std_devs,
                    _model_outputs,
                ) = scheduler.consume_local_rollout_debug_tensors(batch)

                prev_ref, prev_mean_ref, noise_std_ref = _flowgrpo_reference_one_step(
                    model_output=model_output,
                    sample=sample,
                    variance_noise=variance_noise,
                    current_sigma=current_sigma,
                    next_sigma=next_sigma,
                    noise_level=0.5,
                    sde_type=sde_type,
                    sigma_max=scheduler.sigmas[1].item(),
                )

                max_abs_prev = float((prev_sgl - prev_ref).abs().max().item())
                max_abs_mean = float(
                    (prev_sample_means[:, 0] - prev_mean_ref).abs().max().item()
                )
                max_abs_std = float(
                    (noise_std_devs[:, 0, 0] - noise_std_ref.reshape(-1)).abs().max().item()
                )

                self.assertLessEqual(
                    max_abs_prev,
                    atol,
                    msg=f"{sde_type} seed={seed} prev_sample max_abs={max_abs_prev:.6f}",
                )
                self.assertLessEqual(
                    max_abs_mean,
                    atol,
                    msg=f"{sde_type} seed={seed} prev_sample_mean max_abs={max_abs_mean:.6f}",
                )
                self.assertLessEqual(
                    max_abs_std,
                    atol,
                    msg=f"{sde_type} seed={seed} noise_std max_abs={max_abs_std:.6f}",
                )


if __name__ == "__main__":
    unittest.main()
