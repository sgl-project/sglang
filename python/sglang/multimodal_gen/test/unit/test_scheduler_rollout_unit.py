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

    def test_ode_bit_exact_with_non_rollout_path(self):
        """ODE rollout must produce the exact same prev_sample as the
        non-rollout deterministic branch in
        ``scheduling_flow_match_euler_discrete.step`` (``prev_sample =
        sample + dt * model_output``). Uses bf16 model_output because the
        wrapped-scalar promotion difference that a spurious
        ``model_output.float()`` in the ODE branch would introduce is most
        visible at bf16 precision."""
        scheduler = _DummyScheduler()
        batch = self._build_batch(debug_mode=False)
        scheduler.prepare_rollout(batch)

        sample = torch.randn(2, 4, 8, 8, dtype=torch.float32)
        model_output = torch.randn_like(sample).to(torch.bfloat16)
        current_sigma = torch.tensor(0.6, dtype=torch.float32)
        next_sigma = torch.tensor(0.4, dtype=torch.float32)
        dt = next_sigma - current_sigma

        rollout_prev = scheduler.flow_sde_sampling(
            batch,
            model_output=model_output,
            sample=sample,
            current_sigma=current_sigma,
            next_sigma=next_sigma,
            generator=torch.Generator(device=sample.device).manual_seed(0),
        )
        # Exact expression used by the non-rollout branch at
        # scheduling_flow_match_euler_discrete.py `prev_sample = sample +
        # dt * model_output` (after the shared ``sample.to(fp32)`` cast).
        non_rollout_prev = sample + dt * model_output

        pre_cast_max_abs_diff = (rollout_prev - non_rollout_prev).abs().max().item()
        post_cast_max_abs_diff = (
            (
                rollout_prev.to(model_output.dtype)
                - non_rollout_prev.to(model_output.dtype)
            )
            .abs()
            .max()
            .item()
        )
        print(
            f"\n[ODE rollout vs non-rollout, bf16 model_output] "
            f"max |diff| pre-cast={pre_cast_max_abs_diff}, "
            f"post-cast={post_cast_max_abs_diff}"
        )

        self.assertEqual(rollout_prev.dtype, non_rollout_prev.dtype)
        self.assertEqual(pre_cast_max_abs_diff, 0.0)
        self.assertEqual(post_cast_max_abs_diff, 0.0)

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

                def _mock_rollout_variance_noise(_batch, *_args, **_kwargs):
                    # flow_sde_sampling reads the full pre-shard noise from
                    # rollout_session_data.noise_buffer to compute log_prob, so
                    # the mock must populate it alongside returning the
                    # (single-GPU trivially-sharded) noise.
                    scheduler._get_rollout_session_data(  # type: ignore[attr-defined]
                        _batch
                    ).noise_buffer = variance_noise
                    return variance_noise

                scheduler._rollout_variance_noise = (  # type: ignore[method-assign]
                    _mock_rollout_variance_noise
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

    def test_sde_cps_force_fp32_with_bf16_model_output(self):
        """Regression for PyTorch's wrapped-scalar promotion trap: a 0-dim
        fp32 ``noise_std_dev`` multiplied by an N-dim bf16 tensor silently
        demotes to bf16, which would corrupt log-prob precision. SDE/CPS
        branches therefore cast ``model_output.float()`` at entry. Passing
        bf16 ``model_output`` must still yield an fp32 noise buffer and
        an fp32 log-prob sum."""
        scheduler = _DummyScheduler()
        current_sigma = torch.tensor(0.5, dtype=torch.float32)
        next_sigma = torch.tensor(0.3, dtype=torch.float32)
        shape = (1, 16, 1, 32, 32)
        pipeline_config = types.SimpleNamespace(
            shard_latents_for_sp=lambda batch, latents: (latents, False)
        )

        for sde_type in ("sde", "cps"):
            batch = self._build_batch(sde_type=sde_type, shape=shape)
            scheduler.release_rollout_resources(batch)
            scheduler.prepare_rollout(batch=batch, pipeline_config=pipeline_config)

            g = torch.Generator(device="cpu").manual_seed(0)
            model_output = torch.randn(shape, generator=g, dtype=torch.float32).to(
                torch.bfloat16
            )
            sample = torch.randn(shape, generator=g, dtype=torch.float32)

            # Use the real _rollout_variance_noise (no mock) so its dtype
            # propagates from the (original) model_output.dtype into the
            # noise buffer. If flow_sde_sampling fails to cast to fp32 at
            # entry, the buffer is bf16 → log_prob becomes bf16.
            scheduler.flow_sde_sampling(
                batch,
                model_output=model_output,
                sample=sample,
                current_sigma=current_sigma,
                next_sigma=next_sigma,
                generator=g,
            )
            log_prob_sum, _count = scheduler.consume_local_rollout_log_probs(batch)
            self.assertEqual(
                log_prob_sum.dtype,
                torch.float32,
                msg=f"{sde_type}: log_prob_sum must be fp32 with bf16 model_output",
            )
            noise_buffer = scheduler._get_rollout_session_data(batch).noise_buffer
            self.assertEqual(
                noise_buffer.dtype,
                torch.float32,
                msg=f"{sde_type}: noise_buffer must be fp32 with bf16 model_output",
            )

    def test_timestep_filters_gate_sde_and_trajectory(self):
        """Per-step index filters: rollout_sde_step_indices gates variance-noise
        injection (excluded steps = ODE transition + zero log-prob); independently,
        rollout_return_step_indices gates the dit_trajectory append. Both features
        are exercised here because they share the same step_index predicate."""
        from sglang.multimodal_gen.runtime.post_training.rollout_denoising_mixin import (
            RolloutDenoisingMixin,
        )

        # --- Part 1: rollout_sde_step_indices gates SDE noise injection ---
        scheduler = _DummyScheduler()
        shape = (1, 4, 8, 8)
        pipeline_config = types.SimpleNamespace(
            shard_latents_for_sp=lambda _batch, latents: (latents, False)
        )
        batch = types.SimpleNamespace(
            rollout_log_prob_no_const=False,
            rollout_noise_level=0.5,
            rollout_sde_type="sde",
            rollout_debug_mode=False,
            rollout_sde_step_indices=[1],  # only step 1 is stochastic
            latents=torch.empty(shape, dtype=torch.float32),
            _rollout_session_data=None,
        )
        scheduler.prepare_rollout(batch=batch, pipeline_config=pipeline_config)

        g = torch.Generator(device="cpu").manual_seed(0)
        sample = torch.randn(shape, generator=g, dtype=torch.float32)
        model_output = torch.randn(shape, generator=g, dtype=torch.float32)
        current_sigma = torch.tensor(0.6, dtype=torch.float32)
        next_sigma = torch.tensor(0.4, dtype=torch.float32)

        variance_noise_ref = torch.randn(shape, generator=g, dtype=torch.float32)
        variance_noise_call_count = {"n": 0}

        def _mock_variance_noise(_batch, *_args, **_kwargs):
            variance_noise_call_count["n"] += 1
            scheduler._get_rollout_session_data(_batch).noise_buffer = (
                variance_noise_ref
            )
            return variance_noise_ref

        scheduler._rollout_variance_noise = (  # type: ignore[method-assign]
            _mock_variance_noise
        )

        # Step 0: not in filter → deterministic ODE transition, no noise draw.
        batch._rollout_loop_step_index = 0
        prev_0 = scheduler.flow_sde_sampling(
            batch,
            model_output=model_output,
            sample=sample,
            current_sigma=current_sigma,
            next_sigma=next_sigma,
            generator=g,
        )
        self.assertEqual(variance_noise_call_count["n"], 0)
        expected_ode = sample + (next_sigma - current_sigma) * model_output
        self.assertTrue(torch.allclose(prev_0, expected_ode, atol=1e-6))

        # Step 1: in filter → real SDE, noise drawn, prev differs from ODE form.
        batch._rollout_loop_step_index = 1
        prev_1 = scheduler.flow_sde_sampling(
            batch,
            model_output=model_output,
            sample=sample,
            current_sigma=current_sigma,
            next_sigma=next_sigma,
            generator=g,
        )
        self.assertEqual(variance_noise_call_count["n"], 1)
        self.assertFalse(torch.allclose(prev_1, expected_ode, atol=1e-3))

        log_prob_sum, elem_count = scheduler.consume_local_rollout_log_probs(batch)
        self.assertEqual(tuple(log_prob_sum.shape), (shape[0], 2))
        # Filtered step contributes zero log-prob; real SDE step does not.
        self.assertTrue(
            torch.allclose(log_prob_sum[:, 0], torch.zeros_like(log_prob_sum[:, 0]))
        )
        self.assertFalse(
            torch.allclose(log_prob_sum[:, 1], torch.zeros_like(log_prob_sum[:, 1]))
        )
        # elem_count dimension must be preserved for both steps so downstream
        # consume_local_rollout_log_probs stacking stays consistent.
        self.assertTrue(torch.all(elem_count > 0))

        # --- Part 2: rollout_return_step_indices gates dit trajectory append ---
        class _DummyDit(RolloutDenoisingMixin):
            pass

        dit = _DummyDit()
        lat = torch.zeros(1, 4, 8, 8)
        ts = torch.tensor(0.5)

        # Filter [0, 2] over steps 0,1,2 → steps 0 and 2 appended, step 1 skipped.
        traj_filtered = types.SimpleNamespace(
            rollout=True,
            rollout_return_dit_trajectory=True,
            rollout_return_step_indices=[0, 2],
            _rollout_denoising_env_state={"step_latents": [], "step_timesteps": []},
        )
        for i in range(3):
            dit._maybe_append_dit_trajectory_step(
                batch=traj_filtered,
                latents=lat,
                timestep_value=ts,
                step_index=i,
            )
        self.assertEqual(
            len(traj_filtered._rollout_denoising_env_state["step_latents"]), 2
        )
        self.assertEqual(
            len(traj_filtered._rollout_denoising_env_state["step_timesteps"]), 2
        )

        # None (default) → all steps appended (back-compat).
        traj_all = types.SimpleNamespace(
            rollout=True,
            rollout_return_dit_trajectory=True,
            rollout_return_step_indices=None,
            _rollout_denoising_env_state={"step_latents": [], "step_timesteps": []},
        )
        for i in range(3):
            dit._maybe_append_dit_trajectory_step(
                batch=traj_all,
                latents=lat,
                timestep_value=ts,
                step_index=i,
            )
        self.assertEqual(len(traj_all._rollout_denoising_env_state["step_latents"]), 3)

        # Filter excludes step_index=T (the final/(T+1)-th latent appended by
        # _postprocess_rollout_outputs). Simulate T=3 loop steps + final append.
        traj_exclude_final = types.SimpleNamespace(
            rollout=True,
            rollout_return_dit_trajectory=True,
            rollout_return_step_indices=[0, 1, 2],  # excludes T=3
            _rollout_denoising_env_state={"step_latents": [], "step_timesteps": []},
        )
        for i in range(3):
            dit._maybe_append_dit_trajectory_step(
                batch=traj_exclude_final,
                latents=lat,
                timestep_value=ts,
                step_index=i,
            )
        # Mimic the final append routed through the same filter.
        dit._maybe_append_dit_trajectory_step(
            batch=traj_exclude_final,
            latents=lat,
            timestep_value=torch.zeros(()),
            step_index=3,
        )
        self.assertEqual(
            len(traj_exclude_final._rollout_denoising_env_state["step_latents"]), 3
        )
        self.assertEqual(
            len(traj_exclude_final._rollout_denoising_env_state["step_timesteps"]), 3
        )

        # Filter includes only step_index=T → only the final latent survives.
        traj_only_final = types.SimpleNamespace(
            rollout=True,
            rollout_return_dit_trajectory=True,
            rollout_return_step_indices=[3],
            _rollout_denoising_env_state={"step_latents": [], "step_timesteps": []},
        )
        for i in range(3):
            dit._maybe_append_dit_trajectory_step(
                batch=traj_only_final,
                latents=lat,
                timestep_value=ts,
                step_index=i,
            )
        dit._maybe_append_dit_trajectory_step(
            batch=traj_only_final,
            latents=lat,
            timestep_value=torch.zeros(()),
            step_index=3,
        )
        self.assertEqual(
            len(traj_only_final._rollout_denoising_env_state["step_latents"]), 1
        )
        self.assertEqual(
            len(traj_only_final._rollout_denoising_env_state["step_timesteps"]), 1
        )


if __name__ == "__main__":
    unittest.main()
