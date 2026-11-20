"""SGLANG_BENCH_DRAFTER_TEMPERATURE: the shared helper, the DFLASH-family draft heads and
the startup gates. DSpark's draft params are covered in
test_dspark_bench_drafter_temperature.py."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative import dflash_worker_v2 as dfw
from sglang.srt.speculative.bench_drafter import (
    EAGER_STEP_OFFSET,
    bench_drafter_temperature,
    check_algorithm,
    gumbel_draft_scores,
    override_draft_sampling,
)
from sglang.srt.speculative.lilicorr_utils import LiLiCorrDraftSampler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _sampling_info(temperatures, top_ks):
    top_ks = torch.tensor(top_ks)
    return SimpleNamespace(
        temperatures=torch.tensor(temperatures, dtype=torch.float32).view(-1, 1),
        top_ks=top_ks,
        is_all_greedy=bool((top_ks <= 1).all()),
    )


GREEDY = _sampling_info([0.0, 0.0, 0.0], [1, 1, 1])
MIXED = _sampling_info([0.0, 0.7, 1.0], [1, 20, 1 << 30])


@contextmanager
def _bench(value):
    with envs.SGLANG_BENCH_DRAFTER_TEMPERATURE.override(value):
        bench_drafter_temperature.cache_clear()
        try:
            yield
        finally:
            bench_drafter_temperature.cache_clear()


class _Case(CustomTestCase):
    def setUp(self):
        bench_drafter_temperature.cache_clear()

    def tearDown(self):
        bench_drafter_temperature.cache_clear()


class TestGumbelDraftScores(_Case):
    def _scores(self, logits, step, col_start=0, row_start=0):
        return gumbel_draft_scores(
            logits,
            col_start=col_start,
            row_start=row_start,
            step=torch.tensor(step, dtype=torch.int64),
        )

    def test_off_is_identity(self):
        logits = torch.randn(3, 7)
        self.assertIs(self._scores(logits, 0), logits)

    def test_does_not_modify_float32_logits(self):
        logits = torch.randn(3, 7)
        before = logits.clone()
        with _bench(1000):
            self._scores(logits, 0)
        self.assertTrue(torch.equal(logits, before))

    def test_deterministic_per_step_and_finite(self):
        logits = torch.randn(4, 50).to(torch.bfloat16)
        with _bench(1000):
            a, b = self._scores(logits, 3), self._scores(logits, 3)
            c = self._scores(logits, 4)
        self.assertEqual(a.dtype, torch.float32)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))
        self.assertTrue(bool(torch.isfinite(a).all()))

    def test_eager_and_graph_streams_do_not_share_noise(self):
        logits = torch.zeros(4, 64)
        with _bench(1000):
            graph = self._scores(logits, 7)
            eager = self._scores(logits, 7 + EAGER_STEP_OFFSET)
        self.assertFalse(torch.equal(graph.argmax(-1), eager.argmax(-1)))

    def test_vocab_shards_match_the_full_vocabulary(self):
        # Keyed by global token id: a TP shard's scores are exactly its slice of the
        # full-vocab scores, so per-shard max + all-gather max is an exact sample.
        logits = torch.randn(5, 96)
        with _bench(1000):
            full = self._scores(logits, 11)
            shards = [
                self._scores(logits[:, i : i + 32], 11, col_start=i)
                for i in (0, 32, 64)
            ]
        self.assertTrue(torch.equal(torch.cat(shards, dim=1), full))

    def test_high_temperature_is_near_uniform(self):
        vocab, steps = 16, 4000
        logits = torch.linspace(-8.0, 8.0, vocab).repeat(1, 1)
        with _bench(1000):
            picks = torch.stack(
                [self._scores(logits, s).argmax(-1) for s in range(steps)]
            ).view(-1)
        counts = torch.bincount(picks, minlength=vocab).float()
        expected = steps / vocab
        self.assertEqual(int((counts > 0).sum()), vocab)
        self.assertLess(float((counts - expected).abs().max()), 0.35 * expected)

    def test_low_temperature_follows_the_logits(self):
        logits = torch.tensor([[0.0, 5.0, 1.0, -2.0]])
        with _bench(0.01):
            picks = {int(self._scores(logits, s).argmax()) for s in range(200)}
        self.assertEqual(picks, {1})


class TestOverride(_Case):
    def test_override_fills_only_when_enabled(self):
        temps, mask = torch.ones(3), torch.ones(3, dtype=torch.bool)
        override_draft_sampling(temps, mask)
        self.assertTrue(bool(mask.all()))
        with _bench(1000):
            override_draft_sampling(temps, mask)
        self.assertTrue(torch.equal(temps, torch.full((3,), 1000.0)))
        self.assertFalse(bool(mask.any()))

    def test_algorithm_gate(self):
        check_algorithm("EAGLE")
        with _bench(1000):
            check_algorithm("DSPARK")
            check_algorithm("DFLASH")
            for algorithm in (
                "EAGLE",
                "EAGLE3",
                "STANDALONE",
                "FROZEN_KV_MTP",
                "NGRAM",
            ):
                with self.subTest(algorithm=algorithm):
                    with self.assertRaisesRegex(ValueError, "supports"):
                        check_algorithm(algorithm)


VOCAB, HIDDEN, BLOCK = 40, 8, 4


def _plain_sampler(weight):
    return dfw._DflashDraftSampler(
        weight=weight, block_size=BLOCK, num_org=VOCAB, org_vocab_start=0, max_bs=2
    )


class TestPlainDflashGraphSampler(_Case):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        self.weight = torch.randn(VOCAB, HIDDEN)
        self.hidden = torch.randn(2 * BLOCK, HIDDEN)
        rows = self.hidden.view(2, BLOCK, HIDDEN)[:, 1:, :].reshape(-1, HIDDEN)
        self.argmax = (rows @ self.weight.T).argmax(-1)

    def test_off_takes_the_argmax(self):
        sampler = _plain_sampler(self.weight)
        sampler(self.hidden)
        self.assertTrue(torch.equal(sampler.out[: self.argmax.numel()], self.argmax))

    def test_on_samples_and_advances_its_step(self):
        with _bench(1000):
            sampler = _plain_sampler(self.weight)
            seen = []
            for _ in range(20):
                sampler(self.hidden)
                seen.append(sampler.out[: self.argmax.numel()].clone())
        self.assertEqual(int(sampler.noise_step), 20)
        draws = torch.stack(seen)
        self.assertGreater(int(torch.unique(draws).numel()), VOCAB // 2)
        self.assertFalse(bool((draws == self.argmax).all()))


def _eager_worker():
    worker = SimpleNamespace(_bench_eager_noise_step=0)
    worker._bench_draft_scores = lambda logits, **kw: (
        dfw.DFlashWorkerV2._bench_draft_scores(worker, logits, **kw)
    )
    worker._sample_from_vocab_parallel_head = lambda **kw: (
        dfw.DFlashWorkerV2._sample_from_vocab_parallel_head(worker, **kw)
    )
    return worker


class TestPlainDflashEagerHead(_Case):
    def _sample(self, worker, hidden, weight):
        return dfw.DFlashWorkerV2._greedy_sample_from_vocab_parallel_head(
            worker, hidden_states=hidden, lm_head=SimpleNamespace(weight=weight)
        )

    def test_off_is_the_argmax_and_on_samples(self):
        torch.manual_seed(0)
        weight, hidden = torch.randn(VOCAB, HIDDEN), torch.randn(6, HIDDEN)
        argmax = (hidden @ weight.T).argmax(-1)
        worker = _eager_worker()
        self.assertTrue(torch.equal(self._sample(worker, hidden, weight), argmax))
        with _bench(1000):
            draws = torch.stack(
                [self._sample(worker, hidden, weight) for _ in range(20)]
            )
        self.assertEqual(worker._bench_eager_noise_step, 21)
        self.assertGreater(int(torch.unique(draws).numel()), VOCAB // 2)


class TestSelectorAndLiLiCorrStaging(_Case):
    def _selector(self, sampling_enabled=True):
        draft_model = SimpleNamespace(candidate_selector=SimpleNamespace(top_k=4))
        return dfw._SelectorDraftSampler(
            draft_model=draft_model,
            block_size=BLOCK,
            max_bs=4,
            device="cpu",
            sampling_enabled=sampling_enabled,
        )

    def test_selector_off_keeps_greedy_rows(self):
        sampler = self._selector()
        sampler.stage_sampling_params(bs=3, sampling_info=GREEDY)
        self.assertTrue(bool(sampler.greedy_mask[:3].all()))

    def test_selector_on_samples_every_row(self):
        sampler = self._selector()
        with _bench(1000):
            sampler.stage_sampling_params(bs=3, sampling_info=GREEDY)
            self.assertTrue(
                torch.equal(sampler.temperatures[:3], torch.full((3,), 1000.0))
            )
            self.assertFalse(bool(sampler.greedy_mask[:3].any()))
            with self.assertRaisesRegex(ValueError, "requires greedy requests"):
                sampler.stage_sampling_params(bs=3, sampling_info=MIXED)

    def test_megakernel_selector_stages_the_benchmark_temperature(self):
        head = Mock()
        sampler = dfw._MegakernelSelectorDraftSampler(
            head=head,
            out=None,
            candidate_out=None,
            q_out=torch.empty(1),
            sampling_enabled=True,
        )
        with _bench(1000):
            sampler.stage_sampling_params(bs=3, sampling_info=GREEDY)
        staged = head.stage_draft_head_sampling.call_args.kwargs
        self.assertTrue(torch.equal(staged["temperatures"], torch.full((3,), 1000.0)))
        self.assertFalse(bool(staged["greedy_mask"].any()))

    def test_lilicorr_on_samples_every_row(self):
        sampler = SimpleNamespace(
            sampling_enabled=True,
            temperatures=torch.ones(4),
            greedy_mask=torch.ones(4, dtype=torch.bool),
        )
        with _bench(1000):
            LiLiCorrDraftSampler.stage_sampling_params(
                sampler, bs=3, sampling_info=GREEDY
            )
        self.assertTrue(torch.equal(sampler.temperatures[:3], torch.full((3,), 1000.0)))
        self.assertFalse(bool(sampler.greedy_mask[:3].any()))
        self.assertTrue(bool(sampler.greedy_mask[3]))


def _dflash_worker(**overrides):
    fields = dict(
        _is_domino=False,
        selector=None,
        _selector_sampling_enabled=True,
        lilicorr=None,
        _lilicorr_sampling_enabled=False,
        _verify_planner=None,
        _acceptance_estimator=None,
        _target_tp_rank=1,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestDflashWorkerGuard(_Case):
    def _check(self, worker, value="1000", acc_len=3.0, mode="real-draft-token"):
        with (
            patch.object(dfw, "SIMULATE_ACC_LEN", acc_len),
            patch.object(dfw, "SIMULATE_ACC_TOKEN_MODE", mode),
            _bench(value),
        ):
            dfw.DFlashWorkerV2._check_bench_drafter_temperature(worker)

    def test_off_accepts_anything(self):
        self._check(
            _dflash_worker(_is_domino=True), value="0", acc_len=-1, mode="fixed"
        )

    def test_benchmark_config_passes(self):
        self._check(_dflash_worker())
        # The default 'fixed' token mode is overridden (warning), not refused.
        self._check(_dflash_worker(_target_tp_rank=0), mode="fixed")
        self._check(_dflash_worker(selector=object()))
        self._check(_dflash_worker(lilicorr=object(), _lilicorr_sampling_enabled=True))

    def test_rejects_unsupported_configs(self):
        cases = {
            "SGLANG_SIMULATE_ACC_LEN": dict(acc_len=-1),
            "Domino": dict(worker=_dflash_worker(_is_domino=True)),
            "selector sampling": dict(
                worker=_dflash_worker(
                    selector=object(), _selector_sampling_enabled=False
                )
            ),
            "SGLANG_ENABLE_LILICORR_SAMPLING": dict(
                worker=_dflash_worker(lilicorr=object())
            ),
            "SGLANG_RAGGED_VERIFY_MODE": dict(
                worker=_dflash_worker(_verify_planner=object())
            ),
        }
        for reason, kw in cases.items():
            with self.subTest(reason=reason):
                worker = kw.pop("worker", _dflash_worker())
                with self.assertRaisesRegex(ValueError, reason):
                    self._check(worker, **kw)


if __name__ == "__main__":
    unittest.main()
