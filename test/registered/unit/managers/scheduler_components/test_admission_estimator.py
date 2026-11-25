import random
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.admission_estimator import (
    AdmissionEstimator,
    StepRegression,
    kv_reads,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

CHUNK = 8192


def _fill(est: AdmissionEstimator, n: int, c0=20000.0, c1=4000.0, c2=500.0):
    rng = random.Random(0)
    for _ in range(n):
        bs = rng.randint(1, 4)
        prefix = [rng.randint(0, 4 * CHUNK) for _ in range(bs)]
        extend = [rng.randint(1, CHUNK // bs) for _ in range(bs)]
        u = sum(extend)
        r = sum(kv_reads(p, e, CHUNK) for p, e in zip(prefix, extend))
        gpu_us = int(c0 + c1 * u / 1000 + c2 * r / 1000)
        est.push_step(gpu_us, gpu_us, False, prefix, extend)


class TestStepRegression(unittest.TestCase):
    def test_recovers_coefficients(self):
        est = AdmissionEstimator(CHUNK, min_samples=32)
        _fill(est, 400)
        c0, c1, c2 = est.regression.coefficients()
        self.assertAlmostEqual(c0, 20000.0, delta=200.0)
        self.assertAlmostEqual(c1, 4000.0, delta=40.0)
        self.assertAlmostEqual(c2, 500.0, delta=20.0)

    def test_slopes_stay_non_negative(self):
        reg = StepRegression()
        # Every step at the budget: c0 and c1 are collinear.
        for i in range(100):
            reg.push(100000 + (i % 7) * 500, CHUNK, 0.0, large=True)
        reg.push(30000, 100, 0.0)
        reg.push(31000, 200, 0.0)
        c0, c1, c2 = reg.coefficients()
        self.assertGreaterEqual(c1, 0.0)
        self.assertEqual(c2, 0.0)

    def test_drain_counts_passes_not_requests(self):
        reg = StepRegression()
        for u in (1000, 2000, 4000, 8000) * 10:
            reg.push(10000 + u, u, 0.0, large=2 * u >= CHUNK)
        # 16 requests of 2k tokens pack into 4 full passes.
        drain = reg.simulate_queue_s([(2000, 2000)] * 15, (2000, 2000), CHUNK)
        self.assertAlmostEqual(drain, 4 * (10000 + CHUNK) / 1e6, delta=1e-3)

    def test_drain_respects_max_requests_per_pass(self):
        reg = StepRegression()
        for u in (1000, 2000, 4000, 8000) * 10:
            reg.push(10000 + u, u, 0.0, large=2 * u >= CHUNK)
        # One request per pass: 16 passes instead of 4.
        drain = reg.simulate_queue_s([(2000, 2000)] * 15, (2000, 2000), CHUNK, 1)
        self.assertAlmostEqual(drain, 16 * (10000 + 2000) / 1e6, delta=1e-3)


class TestAdmissionEstimator(unittest.TestCase):
    def test_long_steps_clipped_not_dropped(self):
        est = AdmissionEstimator(CHUNK, min_samples=32)
        # Before a fit, a step over 2 s is skipped.
        est.push_step(2_500_000, 2_500_000, False, [0], [CHUNK])
        self.assertEqual(len(est.regression.large), 0)
        for i in range(64):
            u = CHUNK if i % 2 else CHUNK // 2
            us = 100_000 + 1_400_000 * u // CHUNK
            est.push_step(us, us, False, [0], [u])
        est.regression.coefficients()
        pred = est.regression.predict_step_us(CHUNK, kv_reads(0, CHUNK, CHUNK))
        # A slow step within 3x the prediction is kept as is.
        est.push_step(2_500_000, 2_500_000, False, [0], [CHUNK])
        self.assertEqual(est.regression.large[-1][0], 2_500_000)
        # A stall is clipped to 3x the prediction, not dropped.
        est.push_step(60_000_000, 60_000_000, False, [0], [CHUNK])
        self.assertAlmostEqual(est.regression.large[-1][0], 3 * pred, delta=1.0)

    def test_no_prediction_before_min_samples(self):
        est = AdmissionEstimator(CHUNK, min_samples=32)
        _fill(est, 31)
        self.assertIsNone(est.estimate_s([], (1000, 1000)))
        _fill(est, 1)
        self.assertIsNotNone(est.estimate_s([], (1000, 1000)))

    def test_prediction_grows_with_queue(self):
        est = AdmissionEstimator(CHUNK, min_samples=32)
        _fill(est, 200)
        short = est.estimate_s([(CHUNK, CHUNK)] * 2, (1000, 1000))
        long = est.estimate_s([(CHUNK, CHUNK)] * 20, (1000, 1000))
        self.assertGreater(long, 5 * short)

    def test_survival_and_gain_reprice(self):
        est = AdmissionEstimator(CHUNK, min_samples=32)
        self.assertEqual(est.price([(100, 40)]), [(100, 40)])
        for _ in range(100):
            # Credited 6k of 10k, 3k survive; nothing gained.
            est.push_match(10000, 6000, 3000)
        self.assertAlmostEqual(est.survival(), 0.5)
        self.assertEqual(est.gain(), 0.0)
        # u + (1 - survival) * matched
        self.assertEqual(est.price([(1000, 400)]), [(1000, 700)])


if __name__ == "__main__":
    unittest.main()
