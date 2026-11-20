import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.bench_drafter import bench_drafter_temperature
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftParams,
    resolve_draft_params,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _sampling_info(temperatures, top_ks):
    top_ks = torch.tensor(top_ks)
    return SimpleNamespace(
        temperatures=torch.tensor(temperatures, dtype=torch.float32).view(-1, 1),
        top_ks=top_ks,
        is_all_greedy=bool((top_ks <= 1).all()),
    )


GREEDY = _sampling_info([0.0, 0.0, 0.0], [1, 1, 1])
MIXED = _sampling_info([0.0, 0.7, 1.0], [1, 20, 1 << 30])


class _BenchTemperature(CustomTestCase):
    """Each test sets the env through override(); the parsed value is cached."""

    def setUp(self):
        bench_drafter_temperature.cache_clear()

    def tearDown(self):
        bench_drafter_temperature.cache_clear()

    def _resolve(self, value, sampling_info, **kwargs):
        if value is None:
            return resolve_draft_params(
                sampling_info=sampling_info, bs=3, device="cpu", **kwargs
            )
        with envs.SGLANG_BENCH_DRAFTER_TEMPERATURE.override(value):
            bench_drafter_temperature.cache_clear()
            return resolve_draft_params(
                sampling_info=sampling_info, bs=3, device="cpu", **kwargs
            )


class TestResolveDraftParams(_BenchTemperature):
    def test_unset_keeps_request_temperatures(self):
        greedy = self._resolve(None, GREEDY)
        self.assertTrue(torch.equal(greedy.step_temperatures, torch.zeros(3)))
        mixed = self._resolve(None, MIXED)
        self.assertTrue(
            torch.allclose(mixed.step_temperatures, torch.tensor([0.0, 0.7, 1.0]))
        )

    def test_zero_is_off(self):
        params = self._resolve(0.0, GREEDY)
        self.assertTrue(torch.equal(params.step_temperatures, torch.zeros(3)))

    def test_override_changes_only_step_temperatures(self):
        baseline = self._resolve(None, GREEDY)
        params = self._resolve(1000.0, GREEDY)
        self.assertTrue(torch.equal(params.step_temperatures, torch.full((3,), 1000.0)))
        # What the target verify reads is untouched: still greedy.
        self.assertTrue(torch.equal(params.temperatures, baseline.temperatures))
        self.assertTrue(torch.equal(params.greedy_mask, baseline.greedy_mask))
        self.assertTrue(bool(params.greedy_mask.all()))

    def test_override_writes_only_the_batch_rows_of_static_buffers(self):
        out = DraftParams(
            temperatures=torch.full((5,), -1.0),
            greedy_mask=torch.zeros(5, dtype=torch.bool),
            step_temperatures=torch.full((5,), -1.0),
        )
        self._resolve(1000.0, GREEDY, out=out)
        self.assertTrue(
            torch.equal(out.step_temperatures, torch.tensor([1000.0] * 3 + [-1.0] * 2))
        )

    def test_override_without_sampling_info(self):
        params = self._resolve(1000.0, None)
        self.assertTrue(torch.equal(params.step_temperatures, torch.full((3,), 1000.0)))


class TestParse(_BenchTemperature):
    def test_unset_is_zero(self):
        self.assertEqual(bench_drafter_temperature(), 0.0)

    def test_valid_value(self):
        with envs.SGLANG_BENCH_DRAFTER_TEMPERATURE.override("1000"):
            bench_drafter_temperature.cache_clear()
            self.assertEqual(bench_drafter_temperature(), 1000.0)

    def test_out_of_range_values_raise(self):
        for raw in ("-1", "inf", "nan", "1e-50", "1e50"):
            with self.subTest(raw=raw):
                with envs.SGLANG_BENCH_DRAFTER_TEMPERATURE.override(raw):
                    bench_drafter_temperature.cache_clear()
                    with self.assertRaises(ValueError):
                        bench_drafter_temperature()

    def test_malformed_value_warns_and_stays_off(self):
        # EnvField falls back to the default (off) with a warning, as for any env.
        with envs.SGLANG_BENCH_DRAFTER_TEMPERATURE.override("1e3x"):
            bench_drafter_temperature.cache_clear()
            with self.assertWarnsRegex(UserWarning, "SGLANG_BENCH_DRAFTER_TEMPERATURE"):
                self.assertEqual(bench_drafter_temperature(), 0.0)


def _worker(**overrides):
    fields = dict(
        _simulate_acc_len=3.8,
        _greedy_draft=False,
        _verify_planner=SimpleNamespace(mode_value="static"),
        _acceptance_estimator=None,
        _target_tp_rank=1,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestWorkerGuard(_BenchTemperature):
    def _check(self, worker, value="1000"):
        with envs.SGLANG_BENCH_DRAFTER_TEMPERATURE.override(value):
            bench_drafter_temperature.cache_clear()
            DSparkWorkerV2._check_bench_drafter_temperature(worker)

    def test_sampling_request_rejected_before_simulated_greedy_can_mask_it(self):
        worker = _worker(_hosts_draft=True, _simulate_acc_greedy=True)
        batch = SimpleNamespace(sampling_info=MIXED)
        with envs.SGLANG_BENCH_DRAFTER_TEMPERATURE.override("1000"):
            with self.assertRaisesRegex(ValueError, "requires greedy requests"):
                DSparkWorkerV2.forward_batch_generation(worker, batch)

    def test_off_accepts_anything(self):
        self._check(_worker(_simulate_acc_len=-1, _greedy_draft=True), "0")

    def test_benchmark_config_passes(self):
        worker = _worker()
        self._check(worker)
        self.assertFalse(worker._greedy_draft)

    def test_overrides_greedy_draft(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                worker = _worker(_greedy_draft=True, _target_tp_rank=rank)
                self._check(worker)
                self.assertFalse(worker._greedy_draft)

    def test_off_keeps_greedy_draft(self):
        worker = _worker(_greedy_draft=True)
        self._check(worker, "0")
        self.assertTrue(worker._greedy_draft)

    def test_env_preconditions_are_checked_with_the_server_args(self):
        # Simulated acceptance and static verify are refused before weights load
        # (test_bench_drafter_temperature.py), so the worker check accepts them.
        self._check(_worker(_simulate_acc_len=-1))


if __name__ == "__main__":
    unittest.main()
