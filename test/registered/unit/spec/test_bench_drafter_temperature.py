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
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


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

    def test_noise_endpoints_are_finite_and_respect_masked_tokens(self):
        # Test both extremes of the 32-bit RNG. Rounding the upper uniform
        # endpoint to 1 used to let a negligible (or masked) logit win.
        words = torch.tensor([[0xFFFFFFFF, 0, 0]])
        with (
            _bench(1),
            patch(
                "sglang.kernels.ops.speculative.dspark.markov_sample._philox",
                return_value=(words, None, None, None),
            ),
        ):
            scores = self._scores(torch.tensor([[0.0, -1000.0, -torch.inf]]), 0)
        self.assertTrue(torch.isfinite(scores[:, :2]).all())
        self.assertEqual(scores[0, 2].item(), -torch.inf)
        self.assertEqual(scores.argmax().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_graph_replay_matches_eager_vocab_shards(self):
        cpu_logits = torch.arange(3 * 503).view(3, 503).to(torch.bfloat16)
        logits = cpu_logits.cuda()
        # Exercise both words of the counter and a non-aligned vocab shard.
        initial = (1 << 32) + 7
        step = torch.tensor(initial, device="cuda", dtype=torch.int64)

        def scores(x, s, col=0):
            return gumbel_draft_scores(x, step=s, col_start=col, row_start=13)

        with _bench(1000):
            scores(logits, step)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                full = scores(logits, step)
                step.add_(1)

            step.fill_(initial)
            for s in range(initial, initial + 3):
                graph.replay()
                sharded = torch.cat(
                    [scores(logits[:, :117], s), scores(logits[:, 117:], s, 117)],
                    dim=1,
                )
                self.assertTrue(torch.equal(full, sharded))
                torch.testing.assert_close(full.cpu(), scores(cpu_logits, s))
                self.assertEqual(step.item(), s + 1)

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
        vocab, steps, rows = 16, 16, 256
        logits = torch.linspace(-8.0, 8.0, vocab).expand(rows, -1)
        with _bench(1000):
            picks = torch.stack(
                [self._scores(logits, s).argmax(-1) for s in range(steps)]
            ).view(-1)
        counts = torch.bincount(picks, minlength=vocab).float()
        expected = steps * rows / vocab
        self.assertEqual(int((counts > 0).sum()), vocab)
        self.assertLess(float((counts - expected).abs().max()), 0.35 * expected)

    def test_smallest_temperature_keeps_finite_logits_ordered(self):
        logits = torch.tensor([[15.0, 25.0]])
        with _bench(torch.finfo(torch.float32).tiny):
            scores = self._scores(logits, 0)
        self.assertTrue(torch.isfinite(scores).all())
        self.assertEqual(scores.argmax().item(), 1)

    def test_low_temperature_follows_the_logits(self):
        logits = torch.tensor([[0.0, 5.0, 1.0, -2.0]]).expand(200, -1)
        with _bench(0.01):
            picks = self._scores(logits, 0).argmax(-1)
        self.assertTrue(torch.all(picks == 1))


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


def _eager_worker(token_id_vocab_size=VOCAB):
    worker = object.__new__(dfw.DFlashWorkerV2)
    worker._bench_eager_noise_step = 0
    worker.model_runner = SimpleNamespace(
        model_config=SimpleNamespace(token_id_vocab_size=token_id_vocab_size)
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
        self.assertEqual(worker._bench_eager_noise_step, 20)
        self.assertGreater(int(torch.unique(draws).numel()), VOCAB // 2)

    def test_rank_with_only_padding_uses_eager_head_for_added_vocab(self):
        head = SimpleNamespace(
            weight=torch.ones(4, 3),
            num_added_embeddings=1,
            shard_indices=SimpleNamespace(num_added_elements=0),
        )
        worker = SimpleNamespace(
            _target_tp_rank=1,
            _target_worker=SimpleNamespace(
                model_runner=SimpleNamespace(model=SimpleNamespace(lm_head=head))
            ),
            block_size=BLOCK,
            selector=None,
            lilicorr=None,
            _is_domino=False,
        )
        with patch.object(
            dfw, "get_parallel", return_value=SimpleNamespace(tp_group=Mock())
        ):
            self.assertIsNone(dfw.DFlashWorkerV2._maybe_build_draft_sampler(worker))

    def test_chunked_added_vocab_matches_unpadded_head(self):
        valid = 9
        worker = _eager_worker(valid)
        worker._draft_greedy_local_cap = 0
        hidden = torch.ones(11, HIDDEN)
        weight = torch.zeros((16, HIDDEN))
        weight[4:8].fill_(1e5)  # padding between base and added embeddings
        weight[13:].fill_(1e5)  # padding after the added embeddings
        head = SimpleNamespace(
            weight=weight,
            shard_indices=SimpleNamespace(
                num_org_elements=4,
                num_org_elements_padded=8,
                num_added_elements=5,
                org_vocab_start_index=0,
                added_vocab_start_index=4,
            ),
        )
        with (
            _bench(1000),
            patch.object(
                dfw,
                "get_parallel",
                return_value=SimpleNamespace(tp_group=SimpleNamespace(world_size=1)),
            ),
        ):
            actual = worker._greedy_sample_from_vocab_parallel_head(
                hidden_states=hidden, lm_head=head, chunk_size=2
            )
            quant_head = SimpleNamespace(
                org_vocab_size=4,
                shard_indices=head.shard_indices,
                quant_method=SimpleNamespace(apply=lambda _, x, bias: x @ weight.T),
            )
            quantized = _eager_worker(valid)._greedy_sample_from_vocab_parallel_head(
                hidden_states=hidden, lm_head=quant_head, chunk_size=2
            )
            expected = self._sample(
                _eager_worker(valid), hidden, torch.cat((weight[:4], weight[8:13]))
            )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(quantized, expected)
        self.assertTrue(bool((actual >= 4).any()))

    def test_benchmark_skips_padded_token_ids(self):
        # Some checkpoints include padding within their declared LM-head vocabulary.
        valid = 7
        weight = torch.zeros(VOCAB, HIDDEN)
        weight[valid:].fill_(1e5)
        hidden = torch.ones(BLOCK, HIDDEN)
        worker = _eager_worker(valid)
        with _bench(1000):
            sampler = dfw._DflashDraftSampler(
                weight=weight,
                block_size=BLOCK,
                num_org=VOCAB,
                org_vocab_start=0,
                max_bs=1,
                token_id_vocab_size=valid,
            )
            sampler(hidden)
            self.assertTrue(bool((sampler.out < valid).all()))
            self.assertTrue(bool((self._sample(worker, hidden, weight) < valid).all()))


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

    def test_megakernel_selector_stages_the_benchmark_temperature(self):
        head = Mock(draft_head_requests=(1, 4))
        with _bench(1000):
            sampler = dfw._MegakernelSelectorDraftSampler(
                head=head,
                out=None,
                candidate_out=None,
                q_out=torch.empty(1),
                sampling_enabled=True,
            )
            worker = SimpleNamespace(_draft_sampler=sampler)
            for bs in (3, 3, 5):  # repeat the shape, then exceed the head capacity
                dfw.DFlashWorkerV2._stage_draft_sampler(
                    worker, bs=bs, sampling_info=GREEDY
                )
        self.assertEqual(head.stage_draft_head_sampling.call_count, 2)
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


class TestDflashTokenMode(_Case):
    def test_benchmark_forces_real_draft_tokens(self):
        with patch.object(dfw, "SIMULATE_ACC_TOKEN_MODE", "fixed"):
            self.assertEqual(dfw._sim_acc_token_mode(), "fixed")
            with _bench(1000):
                self.assertEqual(dfw._sim_acc_token_mode(), "real-draft-token")


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

    def test_rejects_non_finite_acceptance_length(self):
        for length in (float("nan"), float("inf")):
            with self.subTest(length=length):
                with self.assertRaisesRegex(ValueError, "SGLANG_SIMULATE_ACC_LEN"):
                    self._check(_dflash_worker(), acc_len=length)

    def test_plain_dflash_requires_greedy_requests(self):
        worker = _dflash_worker()
        with _bench(1000):
            for sampling_info in (None, GREEDY):
                dfw.DFlashWorkerV2._validate_phase1_sampling_support(
                    worker, SimpleNamespace(sampling_info=sampling_info)
                )
            with self.assertRaisesRegex(ValueError, "requires greedy requests"):
                dfw.DFlashWorkerV2._validate_phase1_sampling_support(
                    worker, SimpleNamespace(sampling_info=MIXED)
                )

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
