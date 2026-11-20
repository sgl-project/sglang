"""SGLANG_BENCH_DRAFTER_TEMPERATURE: the shared helper, the server-args gate, the DSpark
draft params, the DFLASH-family draft heads and the worker hooks."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
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
        bd.bench_drafter_temperature.cache_clear()
        try:
            yield
        finally:
            bd.bench_drafter_temperature.cache_clear()


@contextmanager
def _dp_rank(rank):
    with patch(
        "sglang.srt.runtime_context.get_parallel",
        return_value=SimpleNamespace(attn_dp_rank=rank),
    ):
        yield


class _Case(CustomTestCase):
    def setUp(self):
        bd.bench_drafter_temperature.cache_clear()

    def tearDown(self):
        bd.bench_drafter_temperature.cache_clear()


class TestParse(_Case):
    def test_unset_is_zero_without_simulated_acceptance(self):
        self.assertEqual(bd.bench_drafter_temperature(), 0.0)

    def test_simulated_acceptance_defaults_to_1000(self):
        for value, acc_len, expected in (
            (None, "3.8", 1000.0),
            (None, "1.0", 0.0),  # nothing committed past the bonus token
            ("0", "3.8", 0.0),  # opted out: greedy drafts
            ("250", "3.8", 250.0),
        ):
            with self.subTest(value=value, acc_len=acc_len):
                with envs.SGLANG_SIMULATE_ACC_LEN.override(acc_len), _bench(value):
                    self.assertEqual(bd.bench_drafter_temperature(), expected)

    def test_valid_values(self):
        for raw, value in (("1000", 1000.0), ("1e-5", 1e-5), ("0", 0.0)):
            with self.subTest(raw=raw), _bench(raw):
                self.assertEqual(bd.bench_drafter_temperature(), value)

    def test_out_of_range_values_raise(self):
        # Below the request clamp the draft kernels' value / temperature overflows.
        for raw in ("-1", "1e-6", "1.2e-38", "1e39", "inf", "nan"):
            with self.subTest(raw=raw), _bench(raw):
                with self.assertRaises(ValueError):
                    bd.bench_drafter_temperature()

    def test_malformed_value_warns_and_falls_back_to_the_default(self):
        # EnvField's fallback for any env var: the default, with a warning.
        for acc_len, default in (("-1", 0.0), ("3.8", 1000.0)):
            with self.subTest(acc_len=acc_len):
                with envs.SGLANG_SIMULATE_ACC_LEN.override(acc_len), _bench("1e3x"):
                    with self.assertWarnsRegex(UserWarning, bd.ENV_NAME):
                        self.assertEqual(bd.bench_drafter_temperature(), default)


class TestValidateServerArgs(_Case):
    @contextmanager
    def _benchmark(self, value="1000", acc_len="3.8", mode="static"):
        with (
            _bench(value),
            envs.SGLANG_SIMULATE_ACC_LEN.override(acc_len),
            envs.SGLANG_RAGGED_VERIFY_MODE.override(mode),
        ):
            yield

    def test_off_accepts_any_server(self):
        for algorithm in (None, "EAGLE", "DFLASH"):
            with self.subTest(algorithm=algorithm), self._benchmark("0", "-1"):
                bd.validate_server_args(algorithm)

    def test_invalid_value_fails_on_every_server(self):
        with _bench("-1"):
            with self.assertRaises(ValueError):
                bd.validate_server_args(None)

    def test_benchmark_config_passes(self):
        for algorithm in bd.SUPPORTED_ALGORITHMS:
            with self.subTest(algorithm=algorithm), self._benchmark():
                bd.validate_server_args(algorithm)

    def test_default_leaves_other_algorithms_alone(self):
        for algorithm in (None, "EAGLE", "NGRAM"):
            with self.subTest(algorithm=algorithm):
                with self._benchmark(None, mode="compact"):
                    bd.validate_server_args(algorithm)

    def test_a_refused_default_names_the_opt_out(self):
        platform_worker = type("PlatformDSparkWorker", (), {})
        with (
            self._benchmark(None),
            patch.object(
                SpeculativeAlgorithm,
                "get_platform_worker_cls",
                return_value=platform_worker,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "=0 to keep greedy drafts"):
                bd.validate_server_args("DSPARK")

    def test_the_verify_schedule_is_left_to_the_workers(self):
        # They refuse SGLANG_SIMULATE_ACC_LEN > 1 without a verify-all schedule.
        for mode in ("static", "compact"):
            with self.subTest(mode=mode), self._benchmark(None, mode=mode):
                bd.validate_server_args("DSPARK")

    def test_each_launch_reads_the_variables_afresh(self):
        self.assertEqual(bd.bench_drafter_temperature(), 0.0)  # cached while unset
        with envs.SGLANG_BENCH_DRAFTER_TEMPERATURE.override("1000"):
            with self.assertRaisesRegex(ValueError, "forced acceptance"):
                bd.validate_server_args("DFLASH")

    def test_refuses_unsupported_configs(self):
        cases = (
            ("applies to --speculative-algorithm", None, {}),
            ("applies to --speculative-algorithm", "EAGLE", {}),
            ("applies to --speculative-algorithm", "NGRAM", {}),
            ("SGLANG_SIMULATE_ACC_LEN", "DSPARK", {"acc_len": "-1"}),
            ("SGLANG_SIMULATE_ACC_LEN", "DFLASH", {"acc_len": "inf"}),
            ("SGLANG_SIMULATE_ACC_LEN=1", "DSPARK", {"acc_len": "1.0"}),
        )
        for reason, algorithm, overrides in cases:
            with self.subTest(reason=reason, algorithm=algorithm):
                with self._benchmark(**overrides):
                    with self.assertRaisesRegex(ValueError, reason):
                        bd.validate_server_args(algorithm)

    def test_refuses_an_out_of_tree_platform_worker(self):
        platform_worker = type("PlatformDFlashWorker", (), {})
        with (
            self._benchmark(),
            patch.object(
                SpeculativeAlgorithm,
                "get_platform_worker_cls",
                return_value=platform_worker,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "PlatformDFlashWorker"):
                bd.validate_server_args("DFLASH")


class TestNoiseKey(_Case):
    def test_dp_groups_get_distinct_keys(self):
        keys = []
        for rank in range(4):
            with _dp_rank(rank):
                keys.append(bd.bench_noise_key(12345))
        self.assertEqual(keys[0], 12345)
        self.assertEqual(len(set(keys)), 4)
        self.assertTrue(all(0 <= key < 2**63 for key in keys))

    def test_head_noise_folds_the_target_transform(self):
        self.assertIsNone(bd.bench_head_noise(random_seed=1, vocab_limit=100))
        with _bench(1000), _dp_rank(0):
            plain = bd.bench_head_noise(random_seed=1, vocab_limit=100)
            scaled = bd.bench_head_noise(
                random_seed=1,
                vocab_limit=100,
                output_multiplier=0.25,
                final_logit_softcapping=20.0,
            )
            other_seed = bd.bench_head_noise(random_seed=2, vocab_limit=100)
        self.assertEqual(
            (plain.temperature, plain.softcap, plain.vocab_limit), (1000.0, None, 100)
        )
        self.assertEqual((scaled.temperature, scaled.softcap), (4000.0, 80.0))
        self.assertNotEqual(plain.key, other_seed.key)


def _noise(temperature=1000.0, key=7, softcap=None, vocab_limit=1 << 20):
    return bd.BenchHeadNoise(
        key=key, temperature=temperature, softcap=softcap, vocab_limit=vocab_limit
    )


def _sample(logits, noise=None, offset=0, vocab_start=0, gather=None):
    noise = noise or _noise()
    return bd.gumbel_argmax(
        logits,
        noise=noise,
        temperatures=torch.full((logits.shape[0],), noise.temperature),
        offset=torch.tensor([offset], dtype=torch.int64),
        step=bd.GRAPH_STEP,
        vocab_start=vocab_start,
        gather=gather,
    )


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
        picks = {int(_sample(logits, _noise(0.01), offset=s)) for s in range(100)}
        self.assertEqual(picks, {1})

    def test_padding_ids_are_never_drawn(self):
        logits = torch.zeros(64, 32)
        logits[:, 20:] = 50.0  # padding rows, which the target masks too
        picks = torch.cat(
            [_sample(logits, _noise(vocab_limit=20), offset=s) for s in range(20)]
        )
        self.assertLess(int(picks.max()), 20)

    def test_dp_keys_and_offsets_change_the_draw(self):
        logits = torch.zeros(32, 256)
        base = _sample(logits, _noise(key=1))
        self.assertFalse(torch.equal(base, _sample(logits, _noise(key=2))))
        self.assertFalse(torch.equal(base, _sample(logits, _noise(key=1), offset=1)))
        self.assertTrue(torch.equal(base, _sample(logits, _noise(key=1))))

    def test_folded_transform_matches_sampling_the_transformed_logits(self):
        torch.manual_seed(0)
        raw = torch.randn(6, 512) * 30
        multiplier, softcap, temperature = 0.25, 20.0, 0.7
        transformed = torch.tanh(raw * multiplier / softcap) * softcap
        folded = _noise(temperature / multiplier, softcap=softcap / multiplier)
        for offset in range(3):
            self.assertTrue(
                torch.equal(
                    _sample(raw, folded, offset=offset),
                    _sample(transformed, _noise(temperature), offset=offset),
                )
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_graph_replay_matches_the_torch_reference(self):
        torch.manual_seed(0)
        cpu_logits = torch.randn(7, 3000).to(torch.bfloat16)
        logits = cpu_logits.cuda()
        noise = _noise(1000.0, key=99, vocab_limit=2999)
        temperatures = torch.full((7,), noise.temperature, device="cuda")
        offset = torch.zeros(1, dtype=torch.int64, device="cuda")
        out = torch.empty(7, dtype=torch.int64, device="cuda")

        def draw():
            out.copy_(
                bd.gumbel_argmax(
                    logits,
                    noise=noise,
                    temperatures=temperatures,
                    offset=offset,
                    step=bd.GRAPH_STEP,
                    vocab_start=0,
                )
            )
            offset.add_(1)

        draw()  # compile the kernels before capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            draw()
        offset.zero_()
        for step in range(3):
            graph.replay()
            expected = _sample(cpu_logits.float(), noise, offset=step)
            self.assertTrue(torch.equal(out.cpu(), expected))
        self.assertEqual(int(offset.item()), 3)


class TestDSparkDraftParams(_Case):
    def _resolve(self, value, sampling_info, **kwargs):
        with _bench(value):
            return resolve_draft_params(
                sampling_info=sampling_info, bs=3, device="cpu", **kwargs
            )

    def test_off_keeps_the_request_temperatures(self):
        for value in (None, "0"):
            with self.subTest(value=value):
                greedy = self._resolve(value, GREEDY)
                self.assertTrue(torch.equal(greedy.step_temperatures, torch.zeros(3)))
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
        no_info = self._resolve(1000.0, None)
        self.assertTrue(
            torch.equal(no_info.step_temperatures, torch.full((3,), 1000.0))
        )

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
        for acc_len, expected in (("3.8", 1000.0), ("1.0", 0.0)):
            with self.subTest(acc_len=acc_len):
                with envs.SGLANG_SIMULATE_ACC_LEN.override(acc_len):
                    params = self._resolve(None, GREEDY)
                self.assertTrue(
                    torch.equal(params.step_temperatures, torch.full((3,), expected))
                )


class TestDSparkWorker(_Case):
    def _apply(self, value="1000", **fields):
        worker = SimpleNamespace(
            **{"_simulate_acc_len": 3.8, "_greedy_draft": True, "_target_tp_rank": 1}
            | fields
        )
        with _bench(value):
            dsw.DSparkWorkerV2._apply_bench_drafter_temperature(worker)
        return worker

    def test_overrides_greedy_draft(self):
        self.assertFalse(self._apply()._greedy_draft)
        with self.assertLogs(dsw.logger, "WARNING") as logs:
            self.assertFalse(self._apply(_target_tp_rank=0)._greedy_draft)
        self.assertIn("overrides SGLANG_DSPARK_USE_GREEDY_DRAFT", logs.output[0])
        self.assertIn("BENCHMARK: DSpark drafts sample at temperature", logs.output[1])

    def test_off_keeps_greedy_draft(self):
        self.assertTrue(self._apply("0")._greedy_draft)


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
        sampler(self.hidden)
        self.assertTrue(torch.equal(sampler.out[: self.argmax.numel()], self.argmax))

    def test_on_samples_and_advances_its_offset(self):
        sampler = self._sampler(_noise())
        draws = []
        for _ in range(20):
            sampler(self.hidden)
            draws.append(sampler.out[: self.argmax.numel()].clone())
        self.assertEqual(int(sampler.bench_offset), 20)
        draws = torch.stack(draws)
        self.assertGreater(int(torch.unique(draws).numel()), VOCAB // 2)
        self.assertFalse(bool((draws == self.argmax).all()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_graph_replays_draw_fresh_noise(self):
        weight, hidden = self.weight.cuda(), self.hidden.cuda()
        sampler = dfw._DflashDraftSampler(
            weight=weight,
            block_size=BLOCK,
            num_org=VOCAB,
            org_vocab_start=0,
            max_bs=2,
            bench_noise=_noise(),
            bench_padded_width=VOCAB,
        )
        sampler(hidden)  # compile the kernels before capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sampler(hidden)
        sampler.bench_offset.zero_()
        rows = hidden.view(2, BLOCK, HIDDEN)[:, 1:, :].reshape(-1, HIDDEN)
        logits = (rows @ weight.T).cpu()
        n = rows.shape[0]
        for step in range(2):
            graph.replay()
            expected = _sample(logits, _noise(), offset=step)
            self.assertTrue(torch.equal(sampler.out[:n].cpu(), expected))
        self.assertEqual(int(sampler.bench_offset), 2)


def _eager_worker(noise):
    worker = SimpleNamespace(
        _bench_head_noise=noise,
        _bench_eager_offset=torch.zeros(1, dtype=torch.int64),
    )
    worker._bench_sample_from_head = lambda **kw: (
        dfw.DFlashWorkerV2._bench_sample_from_head(worker, **kw)
    )
    return worker


class TestPlainDflashEagerHead(_Case):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        self.weight = torch.randn(VOCAB, HIDDEN)
        self.hidden = torch.randn(6, HIDDEN)

    def _sample(self, worker, chunk_size=256):
        tp = SimpleNamespace(tp_group=SimpleNamespace(world_size=1))
        with patch.object(dfw, "get_parallel", return_value=tp):
            return dfw.DFlashWorkerV2._greedy_sample_from_vocab_parallel_head(
                worker,
                hidden_states=self.hidden,
                lm_head=SimpleNamespace(weight=self.weight),
                chunk_size=chunk_size,
            )

    def test_off_is_the_argmax_and_needs_no_benchmark_state(self):
        argmax = (self.hidden @ self.weight.T).argmax(-1)
        self.assertTrue(torch.equal(self._sample(SimpleNamespace()), argmax))

    def test_on_samples(self):
        with _bench(1000):
            worker = _eager_worker(_noise())
            draws = torch.stack([self._sample(worker) for _ in range(20)])
        self.assertEqual(int(worker._bench_eager_offset), 20)
        self.assertGreater(int(torch.unique(draws).numel()), VOCAB // 2)

    def test_every_chunk_draws_at_its_own_offset(self):
        with _bench(1000):
            worker = _eager_worker(_noise())
            self._sample(worker, chunk_size=1)
        # One row per chunk: the offset advanced once per chunk, not once per call.
        self.assertEqual(int(worker._bench_eager_offset), self.hidden.shape[0])


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

    def test_selector_on_keeps_sampled_rows(self):
        sampler = self._selector()
        with _bench(1000):
            sampler.stage_sampling_params(bs=3, sampling_info=MIXED)
        self.assertTrue(
            torch.allclose(sampler.temperatures[:3], torch.tensor([1000.0, 0.7, 1.0]))
        )
        self.assertFalse(bool(sampler.greedy_mask[:3].any()))

    def test_selector_on_samples_every_row(self):
        sampler = self._selector()
        with _bench(1000):
            sampler.stage_sampling_params(bs=3, sampling_info=GREEDY)
        self.assertTrue(torch.equal(sampler.temperatures[:3], torch.full((3,), 1000.0)))
        self.assertFalse(bool(sampler.greedy_mask[:3].any()))

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


class TestDflashTokenMode(_Case):
    def test_benchmark_forces_real_draft_tokens(self):
        with patch.object(dfw, "SIMULATE_ACC_TOKEN_MODE", "fixed"):
            self.assertEqual(dfw._sim_acc_token_mode(), "fixed")
            with _bench(1000):
                self.assertEqual(dfw._sim_acc_token_mode(), "real-draft-token")
            # The default under simulated acceptance commits the drafts as well.
            with envs.SGLANG_SIMULATE_ACC_LEN.override("3.8"), _bench(None):
                self.assertEqual(dfw._sim_acc_token_mode(), "real-draft-token")
            with envs.SGLANG_SIMULATE_ACC_LEN.override("3.8"), _bench("0"):
                self.assertEqual(dfw._sim_acc_token_mode(), "fixed")

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
                accept_len, _, _, out_tokens, _, _ = dfw.DFlashWorkerV2._accept_block(
                    worker,
                    candidates=torch.tensor([[9, 1, 2, 3]]),
                    next_token_logits=torch.zeros(1, 4, 8),
                    sampling_info=SimpleNamespace(is_all_greedy=True),
                    draft_input=object(),
                    prefix_lens=torch.tensor([3]),
                    bs=1,
                )
            return int(accept_len[0]), out_tokens[0].tolist()

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


def _target_worker(rows_per_rank=4032):
    shard = SimpleNamespace(num_org_elements_padded=rows_per_rank)
    lm_head = SimpleNamespace(shard_indices=shard)
    return SimpleNamespace(
        model_runner=SimpleNamespace(model=SimpleNamespace(lm_head=lm_head))
    )


def _dflash_worker(**overrides):
    fields = dict(
        _is_domino=False,
        selector=None,
        _selector_sampling_enabled=True,
        lilicorr=None,
        _lilicorr_sampling_enabled=False,
        _runtime=None,
        _target_worker=_target_worker(),
        _target_tp_rank=1,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestDflashWorkerGuard(_Case):
    def _check(self, worker, value="1000"):
        with _bench(value):
            dfw.DFlashWorkerV2._check_bench_drafter_temperature(worker)

    def test_off_accepts_anything(self):
        self._check(_dflash_worker(_is_domino=True), value="0")

    def test_benchmark_config_passes(self):
        self._check(_dflash_worker())
        self._check(_dflash_worker(selector=object()))
        self._check(_dflash_worker(lilicorr=object(), _lilicorr_sampling_enabled=True))
        with patch.object(dfw, "SIMULATE_ACC_TOKEN_MODE", "fixed"):
            self._check(_dflash_worker(_target_tp_rank=0))

    def test_a_refused_default_names_the_opt_out(self):
        with envs.SGLANG_SIMULATE_ACC_LEN.override("3.8"), _bench(None):
            with self.assertRaisesRegex(ValueError, "=0 to keep greedy drafts"):
                dfw.DFlashWorkerV2._check_bench_drafter_temperature(
                    _dflash_worker(_is_domino=True)
                )

    def test_a_megakernel_runtime_keeps_running_and_is_named(self):
        # Its drafts are its own until the runtime applies the temperature.
        runtime = type("FakeDFlashRuntime", (), {})()
        with self.assertLogs(dfw.logger, "WARNING") as logs:
            self._check(_dflash_worker(_runtime=runtime, _target_tp_rank=0))
        self.assertIn("not applied by the megakernel's DFlash runtime", logs.output[-1])
        self.assertIn("FakeDFlashRuntime", logs.output[-1])
        self.assertFalse(any("BENCHMARK:" in line for line in logs.output))

    def test_rejects_drafts_it_cannot_sample(self):
        cases = {
            "Domino": _dflash_worker(_is_domino=True),
            "selector sampling": _dflash_worker(
                selector=object(), _selector_sampling_enabled=False
            ),
            "SGLANG_ENABLE_LILICORR_SAMPLING": _dflash_worker(lilicorr=object()),
            # A TP32 shard of a 128,832-row padded vocab starts at 4,026 on rank 1.
            "multiples of 4 rows, got 4026": _dflash_worker(
                _target_worker=_target_worker(rows_per_rank=4026)
            ),
        }
        for reason, worker in cases.items():
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(ValueError, reason):
                    self._check(worker)
        with patch.object(dfw, "_is_npu", True):
            with self.assertRaisesRegex(ValueError, "on NPU"):
                self._check(_dflash_worker())
            self._check(
                _dflash_worker(lilicorr=object(), _lilicorr_sampling_enabled=True)
            )


if __name__ == "__main__":
    unittest.main()
