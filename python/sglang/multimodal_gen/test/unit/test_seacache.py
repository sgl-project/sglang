# SPDX-License-Identifier: Apache-2.0
"""Unit tests for runtime/cache/seacache."""

import contextlib
import dataclasses
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.seacache import SeaCacheParams
from sglang.multimodal_gen.runtime.cache.seacache import (
    SeaCache,
    ab_from_sigma,
    apply_sea_filter,
    sea_filter_response,
)
from sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin import (
    SchedulerDisaggMixin,
    extract_transfer_fields,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.codec import pack_tensors
from sglang.multimodal_gen.runtime.pipelines_core import Req

_GRID = (8, 8)
_CHANNELS = 4
_TOKENS = _GRID[0] * _GRID[1]


def _linspace_sigmas(num_steps: int) -> torch.Tensor:
    """Descending sigmas with the terminal zero a flow-matching scheduler appends."""
    return torch.cat([torch.linspace(1.0, 1.0 / num_steps, num_steps), torch.zeros(1)])


def _fake_batch(**overrides):
    batch = SimpleNamespace(
        enable_seacache=True,
        seacache_params=SeaCacheParams(),
        is_warmup=False,
        progressive_mode="fullres",
        did_sp_shard_latents=False,
        is_cfg_negative=False,
        debug=False,
        num_inference_steps=6,
        scheduler=SimpleNamespace(sigmas=_linspace_sigmas(6)),
    )
    for key, value in overrides.items():
        setattr(batch, key, value)
    return batch


@contextlib.contextmanager
def _forward_context(*, batch, step):
    context = SimpleNamespace(current_timestep=step, forward_batch=batch)
    server_args = SimpleNamespace(enable_breakable_cuda_graph=False)
    with (
        patch(
            "sglang.multimodal_gen.runtime.managers.forward_context.get_forward_context",
            return_value=context,
        ),
        patch(
            "sglang.multimodal_gen.runtime.server_args.get_global_server_args",
            return_value=server_args,
        ),
    ):
        yield


def _run_trajectory(cache, *, batch, features):
    """Drive one full denoising trajectory; return (decisions, accumulator history)."""
    decisions = []
    accumulators = []
    for step, feature in enumerate(features):
        with _forward_context(batch=batch, step=step):
            ran = cache.should_run_blocks(modulated_inp=feature, grid_hw=_GRID)
            if ran:
                cache.record_residual(
                    hidden_states=feature + 1.0, original_hidden_states=feature
                )
        decisions.append(ran)
        branch = cache._branches[batch.is_cfg_negative]
        accumulators.append(branch.accumulated_rel_l1_distance)
    return decisions, accumulators


def _constant_features(num_steps: int) -> list[torch.Tensor]:
    torch.manual_seed(0)
    feature = torch.randn(1, _TOKENS, _CHANNELS)
    return [feature.clone() for _ in range(num_steps)]


class TestSeaFilter(unittest.TestCase):
    def test_normalization_pins_the_filter_gain(self) -> None:
        """Mean mode gives unit mean gain, peak mode unit maximum gain."""
        for sigma in (0.9, 0.5, 0.1):
            a, b = ab_from_sigma(sigma)
            mean_response = sea_filter_response(
                shape=(1, 32, 32, 1),
                dims=(-2, -3),
                a=a,
                b=b,
                power_exp=2.0,
                norm_mode="mean",
                device=torch.device("cpu"),
            )
            self.assertAlmostEqual(mean_response.mean().item(), 1.0, places=5)
            peak_response = sea_filter_response(
                shape=(1, 32, 32, 1),
                dims=(-2, -3),
                a=a,
                b=b,
                power_exp=2.0,
                norm_mode="peak",
                device=torch.device("cpu"),
            )
            self.assertAlmostEqual(peak_response.amax().item(), 1.0, places=5)

    def test_passband_widens_as_the_signal_coefficient_grows(self) -> None:
        """Spectral evolution: later steps must retain more high-frequency energy.

        This is the property the method rests on, and the only guard against an
        inverted filter (swapped a/b, or the complementary 1-SEA response).
        """
        size = 32
        freq = torch.fft.fftfreq(size).abs()
        radius = (freq.reshape(-1, 1) ** 2 + freq.reshape(1, -1) ** 2).sqrt()
        high = radius > 0.25

        gains = []
        for sigma in (0.95, 0.8, 0.6, 0.4, 0.2, 0.05):
            a, b = ab_from_sigma(sigma)
            response = sea_filter_response(
                shape=(1, size, size, 1),
                dims=(-2, -3),
                a=a,
                b=b,
                power_exp=2.0,
                norm_mode="mean",
                device=torch.device("cpu"),
            ).squeeze()
            gains.append((response[high].mean() / response.mean()).item())

        self.assertEqual(gains, sorted(gains))
        self.assertLess(gains[0], gains[-1])

    def test_terminal_sigma_stays_finite(self) -> None:
        """FLUX's first sigma is exactly 1.0, which without clamping gives a=0."""
        a, b = ab_from_sigma(1.0)
        self.assertGreater(a, 0.0)
        filtered = apply_sea_filter(torch.randn(1, 16, 16, 4), a=a, b=b)
        self.assertTrue(torch.isfinite(filtered).all())


class TestSeaCacheSchedule(unittest.TestCase):
    def test_zero_threshold_never_skips(self) -> None:
        """thresh=0 must reduce to the uncached trajectory; it is the A/B control."""
        cache = SeaCache(prefix="flux")
        batch = _fake_batch(seacache_params=SeaCacheParams(thresh=0.0))
        decisions, _ = _run_trajectory(
            cache, batch=batch, features=_constant_features(6)
        )
        self.assertEqual(decisions, [True] * 6)

    def test_large_threshold_skips_every_interior_step(self) -> None:
        """First and last step always refresh; everything between is skippable."""
        cache = SeaCache(prefix="flux")
        batch = _fake_batch(seacache_params=SeaCacheParams(thresh=1e9))
        decisions, _ = _run_trajectory(
            cache, batch=batch, features=_constant_features(6)
        )
        self.assertEqual(decisions, [True, False, False, False, False, True])

    def test_accumulator_grows_on_skip_and_clears_on_refresh(self) -> None:
        """The accumulated distance is what makes consecutive skips progressively
        harder; resetting it on a skip would uncap the skip run."""
        num_steps = 20
        cache = SeaCache(prefix="flux")
        batch = _fake_batch(
            seacache_params=SeaCacheParams(thresh=0.3),
            num_inference_steps=num_steps,
            scheduler=SimpleNamespace(sigmas=_linspace_sigmas(num_steps)),
        )
        decisions, accumulators = _run_trajectory(
            cache, batch=batch, features=_constant_features(num_steps)
        )
        interior = decisions[1:-1]
        self.assertIn(False, interior, "no skip: threshold too low for this schedule")
        self.assertIn(
            True, interior, "no refresh: threshold too high for this schedule"
        )
        for step in range(1, len(decisions)):
            if decisions[step]:
                self.assertEqual(accumulators[step], 0.0)
            else:
                self.assertGreater(accumulators[step], accumulators[step - 1])

    def test_retrieve_adds_the_cached_residual(self) -> None:
        cache = SeaCache(prefix="flux")
        batch = _fake_batch(seacache_params=SeaCacheParams(thresh=1e9))
        features = _constant_features(6)
        with _forward_context(batch=batch, step=0):
            cache.should_run_blocks(modulated_inp=features[0], grid_hw=_GRID)
            cache.record_residual(
                hidden_states=features[0] + 3.0, original_hidden_states=features[0]
            )
        with _forward_context(batch=batch, step=1):
            self.assertFalse(
                cache.should_run_blocks(modulated_inp=features[1], grid_hw=_GRID)
            )
            retrieved = cache.retrieve(hidden_states=features[1])
        torch.testing.assert_close(retrieved, features[1] + 3.0)

    def test_second_trajectory_starts_clean(self) -> None:
        """State must not leak across generations on a reused module."""
        cache = SeaCache(prefix="flux")
        batch = _fake_batch(seacache_params=SeaCacheParams(thresh=1e9))
        first, _ = _run_trajectory(cache, batch=batch, features=_constant_features(6))
        second, _ = _run_trajectory(cache, batch=batch, features=_constant_features(6))
        self.assertEqual(first, second)

    def test_cfg_branches_accumulate_independently(self) -> None:
        """A shared accumulator would let the negative branch trigger positive
        refreshes, which is how counter-parity implementations get this wrong."""
        cache = SeaCache(prefix="flux")
        params = SeaCacheParams(thresh=1e9)
        positive = _fake_batch(seacache_params=params, is_cfg_negative=False)
        negative = _fake_batch(seacache_params=params, is_cfg_negative=True)
        features = _constant_features(6)

        for step in range(6):
            for batch in (positive, negative):
                with _forward_context(batch=batch, step=step):
                    ran = cache.should_run_blocks(
                        modulated_inp=features[step], grid_hw=_GRID
                    )
                    if ran:
                        cache.record_residual(
                            hidden_states=features[step] + 1.0,
                            original_hidden_states=features[step],
                        )

        for is_negative in (False, True):
            branch = cache._branches[is_negative]
            self.assertEqual(branch.real_steps, 2, f"negative={is_negative}")
            self.assertEqual(branch.skipped_steps, 4, f"negative={is_negative}")

    def test_single_branch_model_shares_one_accumulator(self) -> None:
        """Models outside _CFG_SUPPORTED_PREFIXES must not touch negative state."""
        cache = SeaCache(prefix="qwenimage")
        self.assertFalse(cache.supports_cfg_cache)
        batch = _fake_batch(
            seacache_params=SeaCacheParams(thresh=1e9), is_cfg_negative=True
        )
        _run_trajectory(cache, batch=batch, features=_constant_features(6))
        self.assertEqual(cache._branches[True].real_steps, 0)
        self.assertGreater(cache._branches[False].real_steps, 0)


class TestSeaCacheBailOuts(unittest.TestCase):
    def test_warmup_request_leaves_state_untouched(self) -> None:
        """Warmup and CUDA-graph capture issue several forwards under one timestep
        and carry a truncated step count, so they must not advance the schedule."""
        cache = SeaCache(prefix="flux")
        batch = _fake_batch(is_warmup=True, num_inference_steps=1)
        decisions, _ = _run_trajectory(
            cache, batch=batch, features=_constant_features(4)
        )
        self.assertEqual(decisions, [True] * 4)
        branch = cache._branches[False]
        self.assertIsNone(branch.previous_modulated_input)
        self.assertIsNone(branch.previous_residual)
        self.assertEqual(branch.real_steps, 0)

    def test_disabled_request_runs_every_step(self) -> None:
        cache = SeaCache(prefix="flux")
        for batch in (
            _fake_batch(enable_seacache=False),
            _fake_batch(seacache_params=None),
            _fake_batch(progressive_mode="dct"),
        ):
            decisions, _ = _run_trajectory(
                cache, batch=batch, features=_constant_features(4)
            )
            self.assertEqual(decisions, [True] * 4)

    def test_sequence_parallel_shard_is_rejected(self) -> None:
        """Each rank holds a row slice, so the 2-D filter would silently see a
        sub-image instead of the latent grid."""
        cache = SeaCache(prefix="flux")
        batch = _fake_batch(did_sp_shard_latents=True)
        with _forward_context(batch=batch, step=0):
            with self.assertRaisesRegex(RuntimeError, "sequence parallelism"):
                cache.should_run_blocks(
                    modulated_inp=_constant_features(1)[0], grid_hw=_GRID
                )

    def test_breakable_cuda_graph_disables_the_gate(self) -> None:
        """A replayed graph freezes whichever branch was taken at capture time."""
        cache = SeaCache(prefix="flux")
        batch = _fake_batch()
        context = SimpleNamespace(current_timestep=0, forward_batch=batch)
        server_args = SimpleNamespace(enable_breakable_cuda_graph=True)
        with (
            patch(
                "sglang.multimodal_gen.runtime.managers.forward_context.get_forward_context",
                return_value=context,
            ),
            patch(
                "sglang.multimodal_gen.runtime.server_args.get_global_server_args",
                return_value=server_args,
            ),
        ):
            self.assertTrue(
                cache.should_run_blocks(
                    modulated_inp=_constant_features(1)[0], grid_hw=_GRID
                )
            )


class TestSeaCacheParamsValidation(unittest.TestCase):
    def test_invalid_controls_are_rejected(self) -> None:
        for kwargs in (
            {"thresh": -0.1},
            {"thresh": float("nan")},
            {"thresh": float("inf")},
            {"norm_mode": "lowpass"},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    SeaCacheParams(**kwargs)

    def test_cache_fields_stay_in_batch_signature(self) -> None:
        """batch_sig_exclude would let requests with different cache settings share
        a dynamic batch, which changes generated output."""
        fields = {field.name: field for field in dataclasses.fields(SamplingParams)}
        for name in ("enable_seacache", "seacache_params"):
            with self.subTest(field=name):
                self.assertFalse(fields[name].metadata.get("batch_sig_exclude"))


class TestSeaCacheDisaggTransfer(unittest.TestCase):
    """Dropped params at the disagg hop silently disable SeaCache."""

    @staticmethod
    def _scheduler():
        # Resolves FluxSamplingParams via the registry, no model files.
        return SimpleNamespace(
            server_args=SimpleNamespace(
                pipeline_class_name="ComfyUIFluxPipeline",
                model_path="/nonexistent",
                backend="auto",
                model_id=None,
            )
        )

    def test_non_default_threshold_round_trips(self) -> None:
        req = Req(
            request_id="seacache-disagg",
            prompt="x",
            sampling_params=SamplingParams(
                enable_seacache=True,
                seacache_params=SeaCacheParams(thresh=0.42, norm_mode="peak"),
            ),
        )

        _, scalar_fields = extract_transfer_fields(req)
        # pack_tensors is the RDMA metadata frame's json path
        metadata_bytes, _ = pack_tensors({}, scalar_fields)
        decoded = json.loads(metadata_bytes.decode("utf-8"))["scalar_fields"]
        rebuilt = SchedulerDisaggMixin._build_disagg_req(
            self._scheduler(), dict(decoded), {}
        )

        self.assertIsInstance(rebuilt.seacache_params, SeaCacheParams)
        self.assertEqual(rebuilt.seacache_params.thresh, 0.42)
        self.assertEqual(rebuilt.seacache_params.norm_mode, "peak")

    def test_model_subclass_defaults_are_reconstructed(self) -> None:
        """num_inference_steps equals the subclass default, so it is never
        transferred."""
        rebuilt = SchedulerDisaggMixin._build_disagg_req(
            self._scheduler(), {"request_id": "t", "prompt": "x"}, {}
        )

        self.assertEqual(rebuilt.num_inference_steps, 50)


if __name__ == "__main__":
    unittest.main()
