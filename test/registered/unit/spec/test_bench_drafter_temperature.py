"""SGLANG_BENCH_ACC_DRAFT_TEMPERATURE: under simulated acceptance every DSpark and
DFLASH-family draft row samples, a sampled request at its own temperature and a greedy
one at the benchmark temperature, and the drafts are what gets committed."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative import bench_drafter as bd
from sglang.srt.speculative import dflash_worker_v2 as dfw
from sglang.srt.speculative.dspark_components import dspark_worker_v2 as dsw
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftParams,
    resolve_draft_params,
)
from sglang.srt.speculative.lilicorr_utils import LiLiCorrDraftSampler
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import simulated_acc_len_generator
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
def _bench(value, acc_len=None):
    with envs.SGLANG_BENCH_ACC_DRAFT_TEMPERATURE.override(value):
        if acc_len is None:
            bd.bench_drafter_temperature.cache_clear()
            try:
                yield
            finally:
                bd.bench_drafter_temperature.cache_clear()
            return
        with envs.SGLANG_SIMULATE_ACC_LEN.override(acc_len):
            bd.bench_drafter_temperature.cache_clear()
            try:
                yield
            finally:
                bd.bench_drafter_temperature.cache_clear()


class _Case(CustomTestCase):
    def setUp(self):
        bd.bench_drafter_temperature.cache_clear()

    def tearDown(self):
        bd.bench_drafter_temperature.cache_clear()


class TestResolve(_Case):
    def test_resolution(self):
        for value, acc_len, expected in (
            (None, "-1", 0.0),  # no simulated acceptance
            (None, "1.0", 0.0),  # nothing committed past the bonus token
            (None, "3.8", 1000.0),
            ("0", "3.8", 0.0),  # opted out: greedy drafts
            ("250", "3.8", 250.0),
        ):
            with self.subTest(value=value, acc_len=acc_len), _bench(value, acc_len):
                self.assertEqual(bd.bench_drafter_temperature(), expected)

    def test_explicit_greedy_dspark_drafts_win_over_the_default(self):
        with envs.SGLANG_DSPARK_USE_GREEDY_DRAFT.override("1"):
            with _bench(None, "3.8"):
                self.assertEqual(bd.bench_drafter_temperature(), 0.0)
            with _bench("1000", "3.8"):
                self.assertEqual(bd.bench_drafter_temperature(), 1000.0)

    def test_out_of_range_values_raise(self):
        # Below the request clamp the draft kernels' value / temperature overflows.
        for raw in ("-1", "1e-6", "1.2e-38", "1e39", "inf", "nan"):
            with self.subTest(raw=raw), _bench(raw):
                with self.assertRaises(ValueError):
                    bd.bench_drafter_temperature()

    def test_malformed_value_warns_and_falls_back_to_the_default(self):
        # EnvField's fallback for any env var: the default, with a warning.
        for acc_len, default in (("-1", 0.0), ("3.8", 1000.0)):
            with self.subTest(acc_len=acc_len), _bench("1e3x", acc_len):
                with self.assertWarnsRegex(UserWarning, bd.ENV_NAME):
                    self.assertEqual(bd.bench_drafter_temperature(), default)


class TestTokenMode(_Case):
    def test_the_drafts_are_committed_unless_the_mode_was_set_explicitly(self):
        with _bench(None, "-1"):
            self.assertEqual(bd.bench_token_mode("fixed"), "fixed")
        with _bench(None, "3.8"):
            self.assertEqual(bd.bench_token_mode("fixed"), "real-draft-token")
            with envs.SGLANG_SIMULATE_ACC_TOKEN_MODE.override("fixed"):
                self.assertEqual(bd.bench_token_mode("fixed"), "fixed")
        with (
            _bench("1000", "3.8"),
            envs.SGLANG_SIMULATE_ACC_TOKEN_MODE.override("fixed"),
        ):
            self.assertEqual(bd.bench_token_mode("fixed"), "real-draft-token")

    def test_the_accept_block_commits_the_drafts(self):
        worker = SimpleNamespace(
            _selector_sample=None,
            _tp_sync=SimpleNamespace(sync=lambda site, tensor: None),
            _use_triton_accept_bonus=False,
            _simulated_acc_len_rng=simulated_acc_len_generator(17),
            block_size=4,
        )

        def commit(value):
            with _bench(value):
                num_correct_drafts, _, _, out_tokens, _, _ = (
                    dfw.DFlashWorkerV2._accept_block(
                        worker,
                        candidates=torch.tensor([[9, 1, 2, 3]]),
                        next_token_logits=torch.zeros(1, 4, 8),
                        sampling_info=SimpleNamespace(is_all_greedy=True),
                        draft_input=object(),
                        prefix_lens=torch.tensor([3]),
                        bs=1,
                    )
                )
            return int(num_correct_drafts[0]), out_tokens[0].tolist()

        with (
            patch.object(dfw, "SIMULATE_ACC_LEN", 4.0),
            patch.object(dfw, "SIMULATE_ACC_TOKEN_MODE", "fixed"),
        ):
            accepted, tokens = commit("1000")
            self.assertGreater(accepted, 0)
            # The drafts, then the target's (argmax) bonus: no fixed token id.
            self.assertEqual(tokens[:accepted], [1, 2, 3][:accepted])
            self.assertEqual(tokens[accepted], 0)
            _, tokens = commit("0")
            self.assertEqual(set(tokens), {100})


class TestValidateBenchArgs(_Case):
    @contextmanager
    def _benchmark(self, value="1000", acc_len="3.8"):
        with _bench(value, acc_len):
            yield

    def test_off_accepts_any_server(self):
        for algorithm in (None, "EAGLE", "DFLASH"):
            with self.subTest(algorithm=algorithm), self._benchmark("0", "-1"):
                bd.validate_bench_args(algorithm)

    def test_benchmark_config_passes(self):
        for algorithm in bd.SUPPORTED_ALGORITHMS:
            with self.subTest(algorithm=algorithm), self._benchmark():
                bd.validate_bench_args(algorithm)

    def test_default_leaves_other_algorithms_alone(self):
        for algorithm in (None, "EAGLE", "NGRAM"):
            with self.subTest(algorithm=algorithm), self._benchmark(None):
                bd.validate_bench_args(algorithm)

    def test_each_launch_reads_the_variables_afresh(self):
        self.assertEqual(bd.bench_drafter_temperature(), 0.0)  # cached while unset
        with envs.SGLANG_BENCH_ACC_DRAFT_TEMPERATURE.override("1000"):
            with self.assertRaisesRegex(ValueError, "forced acceptance"):
                bd.validate_bench_args("DFLASH")

    def test_refuses_unsupported_configs(self):
        cases = (
            ("-1 ", "must be 0", "DSPARK", {"value": "-1"}),
            ("other algorithm", "applies to --speculative", "EAGLE", {}),
            ("no simulation", "SGLANG_SIMULATE_ACC_LEN", "DSPARK", {"acc_len": "-1"}),
            ("infinite", "SGLANG_SIMULATE_ACC_LEN", "DFLASH", {"acc_len": "inf"}),
            ("length 1", "SGLANG_SIMULATE_ACC_LEN=1", "DSPARK", {"acc_len": "1.0"}),
        )
        for name, reason, algorithm, overrides in cases:
            with self.subTest(name), self._benchmark(**overrides):
                with self.assertRaisesRegex(ValueError, reason):
                    bd.validate_bench_args(algorithm)

    def test_refuses_an_out_of_tree_platform_worker_naming_the_opt_out(self):
        platform_worker = type("PlatformDSparkWorker", (), {})
        with (
            self._benchmark(None),
            patch.object(
                SpeculativeAlgorithm,
                "get_platform_worker_cls",
                return_value=platform_worker,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "PlatformDSparkWorker.*=0 to keep"):
                bd.validate_bench_args("DSPARK")


def _noise(key=7, output_multiplier=1.0, softcap=None, vocab_limit=1 << 20):
    return bd.BenchHeadNoise(
        key=key,
        output_multiplier=output_multiplier,
        softcap=softcap,
        vocab_limit=vocab_limit,
    )


def _sample(
    logits, temperature=1000.0, noise=None, offset=0, vocab_start=0, gather=None
):
    return bd.gumbel_argmax(
        logits,
        noise=noise or _noise(),
        temperatures=torch.full((logits.shape[0],), temperature),
        offset=torch.tensor([offset], dtype=torch.int64),
        step=bd.GRAPH_STEP,
        vocab_start=vocab_start,
        gather=gather,
    )


class TestNoise(_Case):
    def test_dp_groups_get_distinct_keys(self):
        keys = []
        for rank in range(4):
            with get_parallel().override(attn_dp_rank=rank):
                keys.append(bd.bench_noise_key(12345))
        self.assertEqual(keys[0], 12345)
        self.assertEqual(len(set(keys)), 4)
        self.assertTrue(all(0 <= key < 2**63 for key in keys))

    def test_head_noise_folds_the_target_transform(self):
        self.assertIsNone(bd.bench_head_noise(random_seed=1, vocab_limit=100))
        with _bench(1000), get_parallel().override(attn_dp_rank=0):
            scaled = bd.bench_head_noise(
                random_seed=1,
                vocab_limit=100,
                output_multiplier=0.25,
                final_logit_softcapping=20.0,
            )
            other_seed = bd.bench_head_noise(random_seed=2, vocab_limit=100)
        self.assertEqual(
            (scaled.output_multiplier, scaled.softcap, scaled.vocab_limit),
            (0.25, 80.0, 100),
        )
        self.assertNotEqual(scaled.key, other_seed.key)

    def test_head_rows_sample_greedy_requests_at_the_benchmark_temperature(self):
        noise = _noise(output_multiplier=0.5)
        with _bench(1000):
            mixed = bd.head_row_temperatures(
                sampling_info=MIXED, bs=3, rows_per_request=2, noise=noise, device="cpu"
            )
            unknown = bd.head_row_temperatures(
                sampling_info=None, bs=2, rows_per_request=2, noise=noise, device="cpu"
            )
        # Request-major rows, folded onto the head's raw logits (T / m).
        self.assertTrue(
            torch.allclose(mixed, torch.tensor([2000.0, 2000.0, 1.4, 1.4, 2.0, 2.0]))
        )
        self.assertTrue(torch.equal(unknown, torch.full((4,), 2000.0)))


def _candidates(logits, *, offset, vocab_start):
    """One rank's per-block Gumbel-max candidates, as its TP gather would send them."""
    captured = []

    def capture(partial):
        captured.append(partial)
        return partial

    _sample(logits, offset=offset, vocab_start=vocab_start, gather=capture)
    return captured[0]


class TestGumbelArgmax(_Case):
    def test_vocab_shards_draw_exactly_the_full_vocabulary_sample(self):
        torch.manual_seed(0)
        logits = torch.randn(5, 2048)
        shard = 1024
        for offset in range(4):
            full = _sample(logits, offset=offset)
            parts = [
                _candidates(logits[:, i : i + shard], offset=offset, vocab_start=i)
                for i in (0, shard)
            ]
            # Rank 0 of a TP=2 draw: its shard, then every rank's gathered candidates.
            sharded = _sample(
                logits[:, :shard],
                offset=offset,
                gather=lambda partial: torch.cat(parts),
            )
            self.assertTrue(torch.equal(full, sharded))

    def test_high_temperature_is_near_uniform(self):
        vocab, draws = 16, 4000
        logits = torch.linspace(-8.0, 8.0, vocab).repeat(1, 1)
        picks = torch.cat([_sample(logits, offset=s) for s in range(draws)])
        counts = torch.bincount(picks, minlength=vocab).float()
        expected = draws / vocab
        self.assertEqual(int((counts > 0).sum()), vocab)
        self.assertLess(float((counts - expected).abs().max()), 0.3 * expected)

    def test_low_temperature_follows_the_logits(self):
        logits = torch.tensor([[0.0, 5.0, 1.0, -2.0]])
        picks = {int(_sample(logits, temperature=0.01, offset=s)) for s in range(100)}
        self.assertEqual(picks, {1})

    def test_padding_ids_are_never_drawn(self):
        logits = torch.zeros(64, 32)
        logits[:, 20:] = 50.0  # padding rows, which the target masks too
        picks = torch.cat(
            [_sample(logits, noise=_noise(vocab_limit=20), offset=s) for s in range(20)]
        )
        self.assertLess(int(picks.max()), 20)

    def test_dp_keys_and_offsets_change_the_draw(self):
        logits = torch.zeros(32, 256)
        base = _sample(logits, noise=_noise(key=1))
        self.assertFalse(torch.equal(base, _sample(logits, noise=_noise(key=2))))
        self.assertFalse(
            torch.equal(base, _sample(logits, noise=_noise(key=1), offset=1))
        )
        self.assertTrue(torch.equal(base, _sample(logits, noise=_noise(key=1))))

    def test_folded_transform_matches_sampling_the_transformed_logits(self):
        torch.manual_seed(0)
        raw = torch.randn(6, 512) * 30
        multiplier, softcap, temperature = 0.25, 20.0, 0.7
        transformed = torch.tanh(raw * multiplier / softcap) * softcap
        folded = _noise(softcap=softcap / multiplier)
        for offset in range(3):
            self.assertTrue(
                torch.equal(
                    _sample(raw, temperature / multiplier, folded, offset=offset),
                    _sample(transformed, temperature, offset=offset),
                )
            )


class TestDSparkDraftParams(_Case):
    def _resolve(self, value, sampling_info, acc_len=None, **kwargs):
        with _bench(value, acc_len):
            return resolve_draft_params(
                sampling_info=sampling_info, bs=3, device="cpu", **kwargs
            )

    def test_off_keeps_the_request_temperatures(self):
        for value in (None, "0"):
            with self.subTest(value=value):
                mixed = self._resolve(value, MIXED)
                self.assertTrue(
                    torch.allclose(
                        mixed.step_temperatures, torch.tensor([0.0, 0.7, 1.0])
                    )
                )

    def test_on_changes_only_the_step_temperatures_of_greedy_rows(self):
        baseline = self._resolve(None, MIXED)
        params = self._resolve(1000.0, MIXED)
        self.assertTrue(
            torch.allclose(params.step_temperatures, torch.tensor([1000.0, 0.7, 1.0]))
        )
        # What the target verify reads is untouched: greedy rows verify greedily.
        self.assertTrue(torch.equal(params.temperatures, baseline.temperatures))
        self.assertTrue(torch.equal(params.greedy_mask, baseline.greedy_mask))

    def test_on_writes_only_the_batch_rows_of_static_buffers(self):
        out = DraftParams(
            temperatures=torch.full((5,), -1.0),
            greedy_mask=torch.zeros(5, dtype=torch.bool),
            step_temperatures=torch.full((5,), -1.0),
        )
        self._resolve(1000.0, GREEDY, out=out)
        self.assertTrue(
            torch.equal(out.step_temperatures, torch.tensor([1000.0] * 3 + [-1.0] * 2))
        )

    def test_simulated_acceptance_defaults_to_1000(self):
        params = self._resolve(None, GREEDY, acc_len="3.8")
        self.assertTrue(torch.equal(params.step_temperatures, torch.full((3,), 1000.0)))


class TestDSparkWorker(_Case):
    def _apply(self, value, greedy_draft=True):
        worker = SimpleNamespace(
            _simulate_acc_len=3.8, _greedy_draft=greedy_draft, _target_tp_rank=1
        )
        with envs.SGLANG_DSPARK_USE_GREEDY_DRAFT.override(str(int(greedy_draft))):
            with _bench(value, "3.8"):
                dsw.DSparkWorkerV2._apply_bench_drafter_temperature(worker)
        return worker._greedy_draft

    def test_an_explicit_temperature_overrides_greedy_drafts(self):
        self.assertFalse(self._apply("1000"))

    def test_explicit_greedy_drafts_win_over_the_default(self):
        self.assertTrue(self._apply(None))
        self.assertTrue(self._apply("0"))


VOCAB, HIDDEN, BLOCK = 40, 8, 4


class TestPlainDflashGraphSampler(_Case):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        self.weight = torch.randn(VOCAB, HIDDEN)
        self.hidden = torch.randn(2 * BLOCK, HIDDEN)
        rows = self.hidden.view(2, BLOCK, HIDDEN)[:, 1:, :].reshape(-1, HIDDEN)
        self.argmax = (rows @ self.weight.T).argmax(-1)

    def _sampler(self, noise=None):
        return dfw._DflashDraftSampler(
            weight=self.weight,
            block_size=BLOCK,
            num_org=VOCAB,
            org_vocab_start=0,
            max_bs=2,
            bench_noise=noise,
            bench_padded_width=VOCAB,
        )

    def test_off_takes_the_argmax(self):
        sampler = self._sampler()
        sampler.stage_sampling_params(bs=2, sampling_info=GREEDY)
        sampler(self.hidden)
        self.assertTrue(torch.equal(sampler.out[: self.argmax.numel()], self.argmax))

    def test_staging_writes_per_row_temperatures(self):
        sampler = self._sampler(_noise())
        with _bench(1000):
            sampler.stage_sampling_params(
                bs=2, sampling_info=_sampling_info([0.0, 0.7], [1, 20])
            )
        rows_per_request = BLOCK - 1
        expected = torch.tensor([1000.0] * rows_per_request + [0.7] * rows_per_request)
        self.assertTrue(torch.allclose(sampler.bench_temperatures[:6], expected))

    def test_on_samples_and_advances_its_offset(self):
        sampler = self._sampler(_noise())
        with _bench(1000):
            sampler.stage_sampling_params(bs=2, sampling_info=GREEDY)
        draws = []
        for _ in range(20):
            sampler(self.hidden)
            draws.append(sampler.out[: self.argmax.numel()].clone())
        self.assertEqual(int(sampler.bench_offset), 20)
        draws = torch.stack(draws)
        self.assertGreater(int(torch.unique(draws).numel()), VOCAB // 2)
        self.assertFalse(bool((draws == self.argmax).all()))


class TestPlainDflashEagerHead(_Case):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        self.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
        self.hidden = torch.randn(6, HIDDEN)

    def _sample(self, temperatures, offset, chunk_size=256):
        return bd.sample_head(
            hidden_states=self.hidden,
            lm_head=self.lm_head,
            noise=_noise(),
            temperatures=temperatures,
            offset=offset,
            tp_group=SimpleNamespace(world_size=1),
            chunk_size=chunk_size,
        )

    def test_samples_at_the_row_temperatures(self):
        offset = torch.zeros(1, dtype=torch.int64)
        hot = torch.stack(
            [self._sample(torch.full((6,), 1000.0), offset) for _ in range(20)]
        )
        self.assertGreater(int(torch.unique(hot).numel()), VOCAB // 2)
        argmax = (self.hidden @ self.lm_head.weight.T).argmax(-1)
        cold = self._sample(torch.full((6,), 1e-5), offset)
        self.assertTrue(torch.equal(cold, argmax))

    def test_every_chunk_draws_at_its_own_offset(self):
        offset = torch.zeros(1, dtype=torch.int64)
        self._sample(torch.full((6,), 1000.0), offset, chunk_size=1)
        # One row per chunk: the offset advanced once per chunk, not once per call.
        self.assertEqual(int(offset), self.hidden.shape[0])


class TestSelectorAndLiLiCorrStaging(_Case):
    def _selector(self):
        draft_model = SimpleNamespace(candidate_selector=SimpleNamespace(top_k=4))
        return dfw._SelectorDraftSampler(
            draft_model=draft_model,
            block_size=BLOCK,
            max_bs=4,
            device="cpu",
            sampling_enabled=True,
        )

    def test_selector_off_keeps_greedy_rows(self):
        sampler = self._selector()
        sampler.stage_sampling_params(bs=3, sampling_info=GREEDY)
        self.assertTrue(bool(sampler.greedy_mask[:3].all()))

    def test_selector_on_samples_greedy_rows_and_keeps_sampled_ones(self):
        sampler = self._selector()
        with _bench(1000):
            sampler.stage_sampling_params(bs=3, sampling_info=MIXED)
        self.assertTrue(
            torch.allclose(sampler.temperatures[:3], torch.tensor([1000.0, 0.7, 1.0]))
        )
        self.assertFalse(bool(sampler.greedy_mask[:3].any()))

    def test_lilicorr_on_samples_greedy_rows(self):
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


class TestCheckDflashDrafts(_Case):
    def _check(self, value="1000", **overrides):
        kwargs = dict(
            is_domino=False,
            selector_sampling=None,
            lilicorr_sampling=None,
            lm_head=torch.nn.Linear(HIDDEN, 4032, bias=False),
            is_npu=False,
            runtime=None,
            log=False,
            configured_token_mode="fixed",
        )
        kwargs.update(overrides)
        with _bench(value, "3.8"):
            bd.check_dflash_drafts(**kwargs)

    def test_off_accepts_anything(self):
        self._check("0", is_domino=True, is_npu=True)

    def test_benchmark_configs_pass(self):
        self._check()
        self._check(lm_head=None, selector_sampling=True)
        self._check(lm_head=None, lilicorr_sampling=True)

    def test_rejects_drafts_it_cannot_sample(self):
        # A TP32 shard of a 128,832-row padded vocab starts at 4,026 on rank 1.
        cases = {
            "Domino": dict(is_domino=True),
            "selector sampling": dict(lm_head=None, selector_sampling=False),
            "SGLANG_ENABLE_LILICORR_SAMPLING": dict(
                lm_head=None, lilicorr_sampling=False
            ),
            "on NPU": dict(is_npu=True),
            "multiples of 4 rows, got 4026": dict(
                lm_head=torch.nn.Linear(HIDDEN, 4026, bias=False)
            ),
        }
        for reason, overrides in cases.items():
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(ValueError, reason):
                    self._check(**overrides)

    def test_a_refused_default_names_the_opt_out(self):
        with self.assertRaisesRegex(ValueError, "=0 to keep greedy drafts"):
            self._check(None, is_domino=True)

    def test_a_megakernel_runtime_keeps_running(self):
        runtime = type("FakeDFlashRuntime", (), {})()
        with self.assertLogs(bd.logger, "WARNING") as logs:
            self._check(runtime=runtime, log=True)
        self.assertTrue(any("FakeDFlashRuntime" in line for line in logs.output))
        self.assertFalse(any("BENCHMARK:" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
