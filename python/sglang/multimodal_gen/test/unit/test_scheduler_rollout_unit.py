import math
import types
import unittest

import torch

import sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin as rl_mixin_module
from sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin import (
    SchedulerRLMixin,
)


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

        prev_sample = scheduler.flow_sde_sampling(
            batch,
            model_output=model_output,
            sample=sample,
            current_sigma=current_sigma,
            next_sigma=next_sigma,
            generator=torch.Generator(device=sample.device).manual_seed(1),
        )
        log_prob_local_sum, local_elem_count = (
            scheduler.consume_local_rollout_log_probs(batch)
        )
        log_prob_local_sum = log_prob_local_sum.squeeze(-1)
        local_elem_count = local_elem_count.squeeze(-1)

        expected_prev = sample + (next_sigma - current_sigma) * model_output
        self.assertTrue(torch.allclose(prev_sample, expected_prev, atol=1e-6, rtol=0.0))
        self.assertTrue(
            torch.allclose(log_prob_local_sum, torch.zeros_like(log_prob_local_sum))
        )
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
        self.assertTrue(
            torch.allclose(noise_std_devs, torch.zeros_like(noise_std_devs))
        )
        self.assertTrue(
            torch.allclose(variance_noises, torch.zeros_like(variance_noises))
        )


def _flowgrpo_sde_step_with_logprob(
    *,
    model_output: torch.Tensor,
    sample: torch.Tensor,
    variance_noise: torch.Tensor,
    sigma: torch.Tensor,
    sigma_prev: torch.Tensor,
    sigma_max: float,
    noise_level: float,
    sde_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Verbatim from FlowGRPO sd3_sde_with_logprob.py ``sde_step_with_logprob``.

    Returns (prev_sample, log_prob, prev_sample_mean, noise_std_dev).
    ``sigma`` / ``sigma_prev`` follow FlowGRPO convention (current / next).
    """
    model_output = model_output.float()
    sample = sample.float()

    dt = sigma_prev - sigma

    if sde_type == "sde":
        std_dev_t = (
            torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))
            * noise_level
        )
        prev_sample_mean = (
            sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
            + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        )
        noise_std_dev = std_dev_t * torch.sqrt(-1 * dt)
        prev_sample = prev_sample_mean + noise_std_dev * variance_noise

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2)
            / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
            - torch.log(std_dev_t * torch.sqrt(-1 * dt))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )

    elif sde_type == "cps":
        std_dev_t = sigma_prev * math.sin(noise_level * math.pi / 2)
        noise_std_dev = std_dev_t
        pred_original_sample = sample - sigma * model_output
        noise_estimate = sample + model_output * (1 - sigma)
        prev_sample_mean = pred_original_sample * (
            1 - sigma_prev
        ) + noise_estimate * torch.sqrt(sigma_prev**2 - std_dev_t**2)
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)

    else:
        raise ValueError(f"Unsupported sde_type: {sde_type}")

    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return prev_sample, log_prob, prev_sample_mean, noise_std_dev


# FlowGRPO convention: SDE uses full Gaussian log-prob, CPS uses no_const.
_FLOWGRPO_LOG_PROB_NO_CONST = {"sde": False, "cps": True}


class TestSchedulerFlowGRPOStepAlignmentUnit(unittest.TestCase):
    def setUp(self):
        self._orig_get_sp_world_size = rl_mixin_module.get_sp_world_size
        rl_mixin_module.get_sp_world_size = lambda: 1

    def tearDown(self):
        rl_mixin_module.get_sp_world_size = self._orig_get_sp_world_size

    def _build_batch(
        self, *, sde_type: str, shape: tuple[int, ...]
    ) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            rollout_log_prob_no_const=_FLOWGRPO_LOG_PROB_NO_CONST[sde_type],
            rollout_noise_level=0.5,
            rollout_sde_type=sde_type,
            rollout_debug_mode=True,
            latents=torch.empty(shape, dtype=torch.float32),
            _rollout_session_data=None,
        )

    def test_single_step_matches_flowgrpo_reference(self):
        """Verify prev_sample, prev_sample_mean, noise_std_dev, and log_prob
        all match FlowGRPO's ``sde_step_with_logprob`` for SDE and CPS."""
        scheduler = _DummyScheduler()
        current_sigma = torch.tensor(0.5, dtype=torch.float32)
        next_sigma = torch.tensor(0.3, dtype=torch.float32)
        shape = (1, 16, 1, 32, 32)
        atol = 1e-6
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

                prev_sgl = scheduler.flow_sde_sampling(
                    batch,
                    model_output=model_output,
                    sample=sample,
                    current_sigma=current_sigma,
                    next_sigma=next_sigma,
                    generator=g,
                )
                log_prob_sum, elem_count = scheduler.consume_local_rollout_log_probs(
                    batch
                )
                log_prob_sum = log_prob_sum.squeeze(-1)
                elem_count = elem_count.squeeze(-1)
                (
                    _variance_noises,
                    prev_sample_means,
                    noise_std_devs,
                    _model_outputs,
                ) = scheduler.consume_local_rollout_debug_tensors(batch)

                prev_ref, log_prob_ref, prev_mean_ref, noise_std_ref = (
                    _flowgrpo_sde_step_with_logprob(
                        model_output=model_output,
                        sample=sample,
                        variance_noise=variance_noise,
                        sigma=current_sigma,
                        sigma_prev=next_sigma,
                        sigma_max=scheduler.sigmas[1].item(),
                        noise_level=0.5,
                        sde_type=sde_type,
                    )
                )

                log_prob_mean = log_prob_sum / elem_count

                errs = {
                    "prev_sample": float((prev_sgl - prev_ref).abs().max().item()),
                    "prev_sample_mean": float(
                        (prev_sample_means[:, 0] - prev_mean_ref).abs().max().item()
                    ),
                    "noise_std": float(
                        (noise_std_devs[:, 0, 0] - noise_std_ref.reshape(-1))
                        .abs()
                        .max()
                        .item()
                    ),
                    "log_prob": float(
                        (log_prob_mean - log_prob_ref).abs().max().item()
                    ),
                }

                for name, err in errs.items():
                    self.assertLessEqual(
                        err,
                        atol,
                        msg=f"{sde_type} seed={seed} {name} max_abs={err:.9f}",
                    )


if __name__ == "__main__":
    unittest.main()
