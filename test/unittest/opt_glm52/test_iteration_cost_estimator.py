"""Unit tests for IterationCostEstimator and cost-aware chunked prefill.

Tests validate:
- Disabled state preserves baseline behavior
- TPOT is NOT computed as iteration_ms / decode_bs
- Pure decode baseline is tracked separately from mixed
- Controller throttles based on relative slowdown, not absolute target
- Mixed-batch observations update the controller
- Starvation prevention eventually force-admits prefill
- Pure decode being slow does NOT collapse prefill
- Bounded rate-of-change prevents oscillation
- Wait counter only increments when prefill is actually throttled
"""

import pytest
from sglang.srt.managers.scheduler_components.iteration_cost_estimator import (
    IterationCostEstimator,
)


class TestIterationCostEstimator:
    """Basic enable/disable tests."""

    def test_disabled_by_default(self):
        est = IterationCostEstimator()
        assert not est.enabled
        assert est.choose_prefill_chunk_size(2048, False, 8192) == 2048

    def test_enable(self):
        est = IterationCostEstimator()
        est.enable()
        assert est.enabled

    def test_no_decode_returns_full_chunk(self):
        est = IterationCostEstimator()
        est.enable()
        assert est.choose_prefill_chunk_size(2048, False, 8192) == 2048


class TestTPOTSemantics:
    """Verify that TPOT is NOT iteration_ms / decode_bs."""

    def test_tpot_is_iteration_latency_not_divided(self):
        """100ms decode iteration, bs=1 -> ~100ms TPOT (not divided)."""
        est = IterationCostEstimator(ema_alpha=0.5)
        est.enable()
        for _ in range(20):
            est.update_observation(batch_type="decode", iteration_ms=100.0)
        assert abs(est.ema_decode_ms - 100.0) < 1.0

    def test_tpot_does_not_shrink_with_batch_size(self):
        """100ms decode iteration, bs=16 -> still ~100ms TPOT (not 6.25ms).

        The estimator does not receive decode_bs at all — it tracks
        raw iteration latency. Per-request TPOT ~= iteration latency
        because each decode request produces one token per iteration.
        """
        est = IterationCostEstimator(ema_alpha=0.5)
        est.enable()
        for _ in range(20):
            est.update_observation(batch_type="decode", iteration_ms=100.0)
        assert abs(est.ema_decode_ms - 100.0) < 1.0
        assert est.ema_decode_ms > 50.0  # NOT 100/16 = 6.25


class TestMixedBatchObservation:
    """Verify mixed batches are correctly classified and observed."""

    def test_mixed_observation_updates_mixed_ema(self):
        est = IterationCostEstimator(ema_alpha=0.5)
        est.enable()
        for _ in range(15):
            est.update_observation(batch_type="decode", iteration_ms=20.0)
        est.update_observation(batch_type="mixed", iteration_ms=100.0)
        assert est.ema_mixed_ms > 0
        assert abs(est.ema_decode_ms - 20.0) < 1.0

    def test_decode_ema_not_updated_by_mixed(self):
        est = IterationCostEstimator(ema_alpha=0.3)
        est.enable()
        for _ in range(20):
            est.update_observation(batch_type="decode", iteration_ms=30.0)
        decode_before = est.ema_decode_ms
        for _ in range(10):
            est.update_observation(batch_type="mixed", iteration_ms=200.0)
        assert est.ema_decode_ms == decode_before

    def test_idle_not_treated_as_decode(self):
        """Idle iterations (0ms) should be skipped."""
        est = IterationCostEstimator(ema_alpha=0.3)
        est.enable()
        est.update_observation(batch_type="decode", iteration_ms=0.0)
        assert est.decode_obs_count == 0


class TestRelativeController:
    """Verify the controller uses relative slowdown, not absolute target."""

    def test_pure_decode_slow_does_not_collapse_prefill(self):
        """If pure decode is 200ms (already slow), prefill should not be
        permanently reduced to minimum."""
        est = IterationCostEstimator(
            ema_alpha=0.5,
            max_slowdown_ratio=1.5,
            min_chunk_ratio=0.25,
            warmup_iters=5,
        )
        est.enable()
        for _ in range(10):
            est.update_observation(batch_type="decode", iteration_ms=200.0)
        # No mixed observations -> no known slowdown -> full chunk
        chunk = est.choose_prefill_chunk_size(2048, True, 8192)
        assert chunk == 2048

    def test_throttles_when_mixed_exceeds_slowdown_ratio(self):
        """Mixed is 3x decode -> should throttle."""
        est = IterationCostEstimator(
            ema_alpha=0.5,
            max_slowdown_ratio=1.5,
            min_chunk_ratio=0.25,
            warmup_iters=5,
        )
        est.enable()
        for _ in range(10):
            est.update_observation(batch_type="decode", iteration_ms=30.0)
        for _ in range(10):
            est.update_observation(batch_type="mixed", iteration_ms=90.0)
        chunk = 2048
        for _ in range(50):
            chunk = est.choose_prefill_chunk_size(2048, True, 8192)
        assert chunk < 2048

    def test_relaxes_when_mixed_within_ratio(self):
        """Mixed is 1.2x decode -> should not throttle."""
        est = IterationCostEstimator(
            ema_alpha=0.5,
            max_slowdown_ratio=1.5,
            warmup_iters=5,
        )
        est.enable()
        for _ in range(10):
            est.update_observation(batch_type="decode", iteration_ms=30.0)
        for _ in range(10):
            est.update_observation(batch_type="mixed", iteration_ms=36.0)
        chunk = est.choose_prefill_chunk_size(2048, True, 8192)
        assert chunk == 2048

    def test_absolute_limit_triggers_regardless_of_ratio(self):
        """Even if ratio is OK, absolute cap triggers throttle."""
        est = IterationCostEstimator(
            ema_alpha=0.5,
            max_slowdown_ratio=10.0,
            absolute_latency_limit_ms=100.0,
            warmup_iters=5,
        )
        est.enable()
        for _ in range(10):
            est.update_observation(batch_type="decode", iteration_ms=50.0)
        for _ in range(10):
            est.update_observation(batch_type="mixed", iteration_ms=150.0)
        chunk = 2048
        for _ in range(50):
            chunk = est.choose_prefill_chunk_size(2048, True, 8192)
        assert chunk < 2048


class TestStarvationPrevention:
    """Verify prefill eventually makes progress."""

    def test_starvation_force_admits(self):
        """After max_prefill_wait_iters of throttling, force-admit full chunk.

        With max_prefill_wait_iters=5, the first 5 calls are throttled
        (wait_count goes 1→2→3→4→5).  The 6th call sees wait_count>=5
        and force-admits the full chunk.
        """
        est = IterationCostEstimator(
            ema_alpha=0.5,
            max_slowdown_ratio=1.1,
            min_chunk_ratio=0.1,
            max_prefill_wait_iters=5,
            warmup_iters=3,
        )
        est.enable()
        for _ in range(5):
            est.update_observation(batch_type="decode", iteration_ms=20.0)
        for _ in range(5):
            est.update_observation(batch_type="mixed", iteration_ms=200.0)
        # Should throttle for 5 iterations
        for _ in range(5):
            chunk = est.choose_prefill_chunk_size(2048, True, 8192)
            assert chunk < 2048, f"Expected throttling, got chunk={chunk}"
        # 6th call triggers starvation prevention -> full chunk
        chunk = est.choose_prefill_chunk_size(2048, True, 8192)
        assert chunk == 2048

    def test_wait_counter_resets_on_full_chunk(self):
        """When controller returns full chunk (no throttling), wait counter resets."""
        est = IterationCostEstimator(
            ema_alpha=0.5,
            max_slowdown_ratio=1.5,
            warmup_iters=5,
            max_prefill_wait_iters=10,
        )
        est.enable()
        for _ in range(10):
            est.update_observation(batch_type="decode", iteration_ms=30.0)
        # No mixed observations -> no throttling -> full chunk
        est._prefill_wait_count = 5  # simulate some wait
        chunk = est.choose_prefill_chunk_size(2048, True, 8192)
        assert chunk == 2048
        assert est._prefill_wait_count == 0

    def test_continuous_decode_pressure(self):
        """Prefill should eventually be admitted even under continuous
        decode pressure."""
        est = IterationCostEstimator(
            max_slowdown_ratio=1.01,
            max_prefill_wait_iters=10,
            warmup_iters=3,
        )
        est.enable()
        for _ in range(5):
            est.update_observation(batch_type="decode", iteration_ms=50.0)
        for _ in range(5):
            est.update_observation(batch_type="mixed", iteration_ms=500.0)

        admitted = False
        for i in range(30):
            chunk = est.choose_prefill_chunk_size(2048, True, 8192)
            if chunk == 2048:
                admitted = True
                break
        assert admitted, "Prefill should eventually be force-admitted"

    def test_no_decode_work_resets_wait_counter(self):
        """When there is no decode work, wait counter resets."""
        est = IterationCostEstimator(max_prefill_wait_iters=5)
        est.enable()
        est._prefill_wait_count = 10
        chunk = est.choose_prefill_chunk_size(2048, False, 8192)
        assert chunk == 2048
        assert est._prefill_wait_count == 0


class TestBoundedRateOfChange:
    """Verify chunk ratio changes gradually, not in jumps."""

    def test_no_oscillation(self):
        est = IterationCostEstimator(
            ema_alpha=0.5,
            max_slowdown_ratio=1.5,
            min_chunk_ratio=0.25,
            warmup_iters=5,
        )
        est.enable()
        for _ in range(10):
            est.update_observation(batch_type="decode", iteration_ms=30.0)
        for _ in range(10):
            est.update_observation(batch_type="mixed", iteration_ms=90.0)

        ratios = []
        for _ in range(20):
            est.choose_prefill_chunk_size(2048, True, 8192)
            ratios.append(est.current_chunk_ratio)

        for i in range(1, len(ratios)):
            assert abs(ratios[i] - ratios[i - 1]) <= 0.11


class TestWarmup:
    """Verify warmup behavior."""

    def test_warmup_returns_full_chunk(self):
        est = IterationCostEstimator(warmup_iters=20)
        est.enable()
        for _ in range(5):
            est.update_observation(batch_type="decode", iteration_ms=100.0)
        assert est.decode_obs_count < 20
        chunk = est.choose_prefill_chunk_size(2048, True, 8192)
        assert chunk == 2048

    def test_warmup_completes(self):
        est = IterationCostEstimator(warmup_iters=10)
        est.enable()
        for _ in range(10):
            est.update_observation(batch_type="decode", iteration_ms=30.0)
        assert est.decode_obs_count >= 10


class TestDisabledFallback:
    """Verify disabled state is identical to baseline."""

    def test_disabled_returns_base(self):
        est = IterationCostEstimator()
        for decode_bs in [0, 1, 4, 16]:
            for base in [512, 2048, 8192]:
                assert est.choose_prefill_chunk_size(base, True, 16384) == base
                assert est.choose_prefill_chunk_size(base, False, 16384) == base


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
