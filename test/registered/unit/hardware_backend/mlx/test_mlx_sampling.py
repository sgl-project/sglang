"""Unit tests for MLX in-graph sampling (hardware_backend/mlx/sampling.py)."""

from __future__ import annotations

import importlib.util
import unittest
from collections import Counter

from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_mlx_ci(est_time=20, suite="stage-a-unit-test-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires mlx"

if _HAS_MLX:
    import mlx.core as mx

    from sglang.srt.hardware_backend.mlx.sampling import (
        DEFAULT_SAMPLING_SEED,
        GREEDY_PARAMS,
        MAX_BOUNDED_TOP_K,
        MlxLogprobSpec,
        MlxSamplingParams,
        _candidate_width,
        _gumbel_noise,
        _murmur_hash32,
        all_greedy,
        compute_logprobs,
        sample_tokens,
        sanitize_logits,
    )


def _reference_murmur3(seed: int, pos: int, col: int) -> int:
    """Pure-Python MurmurHash3 mirroring the Triton kernel in
    sglang/kernels/ops/sampling/murmur_hash.py: blocks seed_low,
    seed_high, position, column; length-16 finalization; fmix32."""

    def mix(h: int, k: int) -> int:
        k = (k * 0xCC9E2D51) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * 0x1B873593) & 0xFFFFFFFF
        h ^= k
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
        return (h * 5 + 0xE6546B64) & 0xFFFFFFFF

    seed &= 0xFFFFFFFFFFFFFFFF
    h = mix(0, seed & 0xFFFFFFFF)
    h = mix(h, (seed >> 32) & 0xFFFFFFFF)
    h = mix(h, pos & 0xFFFFFFFF)
    h = mix(h, col & 0xFFFFFFFF)
    h ^= 16
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


def _params(temperature=1.0, top_k=1 << 30, top_p=1.0, min_p=0.0, seed=None):
    return MlxSamplingParams(
        temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p, seed=seed
    )


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMurmurHashPort(CustomTestCase):
    def test_matches_pure_python_reference(self):
        """Guards the mx uint32 port of the CUDA murmur kernel: any drift in
        wraparound/shift/block-order semantics changes seeded sampling."""
        seeds = [0, 1, 42, 2**31, 2**63 + 12345]
        positions = [0, 7, 1023, 2**31 - 1, 5]
        vocab = 64
        hashed = _murmur_hash32(seeds=seeds, positions=positions, vocab_size=vocab)
        mx.eval(hashed)
        for row, (seed, pos) in enumerate(zip(seeds, positions)):
            for col in (0, 1, vocab // 2, vocab - 1):
                self.assertEqual(
                    int(hashed[row, col].item()),
                    _reference_murmur3(seed, pos, col),
                    msg=f"mismatch at seed={seed} pos={pos} col={col}",
                )


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestSampleTokens(CustomTestCase):
    VOCAB = 32

    def _logits(self, batch_size: int, key_int: int = 0) -> mx.array:
        return (
            mx.random.normal(shape=(batch_size, self.VOCAB), key=mx.random.key(key_int))
            * 3.0
        )

    def _draw(self, logits, params, positions=None, n=200, key_start=100):
        """Sample n times with distinct keys, return per-row token Counters."""
        batch_size = logits.shape[0]
        positions = positions if positions is not None else [5] * batch_size
        counters = [Counter() for _ in range(batch_size)]
        for i in range(n):
            toks = sample_tokens(
                last_logits=logits,
                params=params,
                positions=positions,
                key=mx.random.key(key_start + i),
            )
            mx.eval(toks)
            for row, t in enumerate(toks.tolist()):
                counters[row][int(t)] += 1
        return counters

    @staticmethod
    def _reference_support(probs, top_k, top_p, min_p):
        """Independent replica of the mask, in pure Python, on sorted probs."""
        order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
        keep, cum = [], 0.0
        for rank, idx in enumerate(order):
            p = probs[idx]
            masked = (
                rank >= min(top_k, len(probs))
                or cum > top_p
                or p < probs[order[0]] * min_p
            )
            cum += p
            if not masked:
                keep.append(idx)
        return set(keep)

    def _probs(self, logits):
        probs = mx.softmax(logits.astype(mx.float32), axis=-1)
        mx.eval(probs)
        return probs[0].tolist()

    def test_greedy_rows_match_argmax_in_mixed_batch(self):
        """A greedy row must return exactly argmax even when other rows in
        the batch sample — guards the where() row-select and the sglang
        greedy convention (top_k == 1)."""
        logits = self._logits(3)
        expected = mx.argmax(logits, axis=-1).tolist()
        params = [_params(top_k=1), _params(temperature=0.7), _params(top_k=1)]
        counters = self._draw(logits, params, n=25)
        self.assertEqual(set(counters[0]), {expected[0]})
        self.assertEqual(set(counters[2]), {expected[2]})

    def test_filter_supports_match_reference(self):
        """The sampled support must stay inside the independently computed
        mask for each filter and for their combination — guards the rank
        mask, the nucleus exclusion, the min_p threshold, and the
        sorted->vocab index map."""
        cases = [
            ("top_k=2", dict(top_k=2)),
            ("top_p=0.6", dict(top_p=0.6)),
            ("min_p=0.3", dict(min_p=0.3)),
            ("top_k=8,top_p=0.7,min_p=0.05", dict(top_k=8, top_p=0.7, min_p=0.05)),
        ]
        for label, kwargs in cases:
            with self.subTest(label):
                logits = self._logits(1, key_int=3)
                support = self._reference_support(
                    self._probs(logits),
                    kwargs.get("top_k", 1 << 30),
                    kwargs.get("top_p", 1.0),
                    kwargs.get("min_p", 0.0),
                )
                drawn = set(self._draw(logits, [_params(**kwargs)], n=400)[0])
                self.assertTrue(drawn <= support, f"{label}: extra {drawn - support}")
                self.assertTrue(drawn, f"{label}: nothing sampled")

    def test_seeded_row_is_deterministic_and_key_independent(self):
        """A row with sampling_seed must produce the same token regardless
        of the RNG key or batch composition — the murmur-gumbel path only
        depends on (seed, position, logits)."""
        logits = self._logits(2, key_int=5)
        seeded = _params(temperature=1.0, seed=1234)
        tok_solo = sample_tokens(
            last_logits=logits[:1],
            params=[seeded],
            positions=[9],
            key=mx.random.key(0),
        )
        tok_other_key = sample_tokens(
            last_logits=logits[:1],
            params=[seeded],
            positions=[9],
            key=mx.random.key(999),
        )
        tok_in_batch = sample_tokens(
            last_logits=logits,
            params=[seeded, _params(temperature=0.8)],
            positions=[9, 3],
            key=mx.random.key(7),
        )
        mx.eval(tok_solo, tok_other_key, tok_in_batch)
        self.assertEqual(tok_solo.tolist(), tok_other_key.tolist())
        self.assertEqual(int(tok_in_batch[0].item()), int(tok_solo[0].item()))

    def test_seeded_row_unaffected_by_batchmate_filtering(self):
        """A seeded row's token must not change when a batchmate triggers
        the top-k/top-p sort path — guards the vocab-id-space noise
        contract (regression: noise was applied in sorted-rank space when
        any row filtered, so batch composition changed seeded tokens)."""
        # Near-uniform seeded row: the Gumbel noise decides the token, so
        # a change of noise index space is guaranteed to show up.
        seeded_logits = (
            mx.random.normal(shape=(1, self.VOCAB), key=mx.random.key(8)) * 0.05
        )
        mate_logits = (
            mx.random.normal(shape=(1, self.VOCAB), key=mx.random.key(9)) * 3.0
        )
        logits = mx.concatenate([seeded_logits, mate_logits], axis=0)
        seeded = _params(seed=4321)
        solo = sample_tokens(
            last_logits=logits[:1],
            params=[seeded],
            positions=[6],
            key=mx.random.key(0),
        )
        with_filtering_mate = sample_tokens(
            last_logits=logits,
            params=[seeded, _params(temperature=1.2, top_k=2)],
            positions=[6, 11],
            key=mx.random.key(55),
        )
        mx.eval(solo, with_filtering_mate)
        self.assertEqual(int(with_filtering_mate[0].item()), int(solo[0].item()))

    def test_bounded_top_k_picks_the_same_token_as_the_full_vocab_chain(self):
        """The bounded top-K chain is an optimization, not a policy change:
        a seeded row must pick the same token whether or not the batch is
        eligible for it.  A batchmate without a finite top_k pushes the
        whole batch back onto the full-vocab chain, so the same seeded row
        is sampled both ways here."""
        logits = self._logits(2, key_int=11)
        seeded = _params(top_k=4, seed=2024)
        bounded = sample_tokens(
            last_logits=logits[:1],
            params=[seeded],
            positions=[6],
            key=mx.random.key(0),
        )
        # min_p alone leaves top_k at TOP_K_ALL, so this batch falls back.
        full_vocab = sample_tokens(
            last_logits=logits,
            params=[seeded, _params(min_p=0.1)],
            positions=[6, 2],
            key=mx.random.key(3),
        )
        mx.eval(bounded, full_vocab)
        self.assertEqual(int(full_vocab[0].item()), int(bounded[0].item()))

    def test_candidate_width_gates_the_bounded_chain(self):
        """Only a batch whose widest top_k fits inside both the bound and
        the vocabulary may shrink the chain; anything else must return the
        full vocab size (which selects the scatter-back path)."""
        vocab = 4096
        for label, params, expected in [
            ("under the bound", [_params(top_k=64)], 64),
            ("at the bound", [_params(top_k=MAX_BOUNDED_TOP_K)], MAX_BOUNDED_TOP_K),
            ("past the bound", [_params(top_k=MAX_BOUNDED_TOP_K + 1)], vocab),
            ("no top_k (TOP_K_ALL)", [_params()], vocab),
            ("top_k == vocab", [_params(top_k=vocab)], vocab),
            ("widest row wins", [_params(top_k=4), _params(top_k=64)], 64),
            ("one unbounded row", [_params(top_k=4), _params(top_p=0.9)], vocab),
        ]:
            with self.subTest(label):
                self.assertEqual(_candidate_width(params, vocab), expected)

    def test_seeded_row_varies_with_position(self):
        """Positions feed the hash, so a fixed seed must not freeze the
        distribution across steps: over many positions the sampled tokens
        must not all collapse to one value (vocab of near-uniform probs)."""
        logits = mx.zeros((1, self.VOCAB))  # uniform distribution
        seeded = [_params(seed=77)]
        toks = set()
        for pos in range(40):
            t = sample_tokens(
                last_logits=logits,
                params=seeded,
                positions=[pos],
                key=mx.random.key(0),
            )
            mx.eval(t)
            toks.add(int(t[0].item()))
        self.assertGreater(len(toks), 5, toks)

    def test_seeded_noise_is_finite(self):
        """The uniform draw is clamped to [2**-32, 1 - 2**-24] before the
        double log: uint32(0xFFFFFFFF) rounds UP to 2**32 in float32, so an
        unclamped u can exceed 1 and make log(-log u) NaN — and u == 1 gives
        +inf, which would deterministically force that token."""
        # Sanity-check the hazard the clamp exists for.
        u_max = mx.array([0xFFFFFFFF], dtype=mx.uint32).astype(mx.float32) / float(
            0xFFFFFFFF
        )
        mx.eval(u_max)
        self.assertGreaterEqual(float(u_max.item()), 1.0)
        self.assertFalse(bool(mx.isfinite(-mx.log(-mx.log(u_max))).item()))

        noise = _gumbel_noise(
            params=[_params(seed=1), _params(seed=2**63 - 1)],
            positions=[0, 4096],
            shape=(2, 1 << 16),
            key=mx.random.key(0),
        )
        mx.eval(noise)
        self.assertTrue(bool(mx.all(mx.isfinite(noise)).item()))

    def test_seed_with_min_p_is_supported(self):
        """seed + min_p is well defined here (the pytorch backend asserts on
        the combination): Gumbel-max over unnormalized masked weights is
        invariant to the missing renormalization, which is exactly the TODO
        at layers/sampler.py's multinomial_with_seed path."""
        logits = self._logits(1, key_int=4)
        support = self._reference_support(self._probs(logits), 1 << 30, 1.0, 0.3)
        tok = sample_tokens(
            last_logits=logits,
            params=[_params(seed=1234, min_p=0.3)],
            positions=[9],
            key=mx.random.key(0),
        )
        mx.eval(tok)
        self.assertIn(int(tok[0].item()), support)

    def test_temperature_sharpens_distribution(self):
        """Lower temperature must concentrate mass on the argmax token —
        guards the per-row temperature division (e.g. broadcasting bugs
        that apply one row's temperature to all rows)."""
        logits = self._logits(2, key_int=6)
        expected0 = int(mx.argmax(logits[0]).item())
        params = [_params(temperature=0.05), _params(temperature=5.0)]
        counters = self._draw(logits, params, n=200)
        self.assertGreater(counters[0][expected0] / 200.0, 0.95)
        self.assertGreater(len(counters[1]), 5, "high temp should spread mass")

    def test_greedy_helpers(self):
        self.assertTrue(all_greedy([GREEDY_PARAMS, _params(top_k=1)]))
        self.assertFalse(all_greedy([GREEDY_PARAMS, _params(temperature=0.9)]))


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestSanitizeAndLogprobs(CustomTestCase):
    VOCAB = 16

    def test_sanitize_matches_nan_to_num_semantics(self):
        """Guards the port of sanitize_nan_logits' exact replacement values
        (+-1e30, not dtype extremes — temperature division would overflow
        dtype extremes back to inf and softmax them to NaN)."""
        import struct

        def f32(v):
            return struct.unpack("f", struct.pack("f", v))[0]

        x = mx.array([[1.0, float("nan"), float("inf"), -float("inf")]])
        out = sanitize_logits(x)
        mx.eval(out)
        self.assertEqual(out.tolist(), [[1.0, f32(-1e30), f32(1e30), f32(-1e30)]])

    def test_logprobs_match_reference_and_row_shapes(self):
        """compute_logprobs must equal log_softmax(logits/temp) per row and
        cut top-k / token-ids to each row's requested shape — guards the
        per-row temperature broadcast and the spec row alignment."""
        import math

        logits = mx.random.normal(shape=(2, self.VOCAB), key=mx.random.key(11))
        params = [_params(temperature=0.5), _params(temperature=2.0)]
        tokens = mx.array([3, 7], dtype=mx.uint32)
        spec = MlxLogprobSpec(top_ks=(2, 0), token_ids=(None, (1, 4)))
        lp = compute_logprobs(logits, params, tokens, spec)
        mx.eval(*[a for a in [lp.chosen, lp.top_val, lp.top_idx] if a is not None])

        raw = logits.tolist()
        for row, temp in ((0, 0.5), (1, 2.0)):
            scaled = [v / temp for v in raw[row]]
            m = max(scaled)
            lse = m + math.log(sum(math.exp(v - m) for v in scaled))
            ref = [v - lse for v in scaled]
            chosen_token = int(tokens[row].item())
            self.assertAlmostEqual(
                float(lp.chosen[row].item()), ref[chosen_token], places=4
            )
            if row == 0:
                expect_top = sorted(ref, reverse=True)[:2]
                got = lp.top_val[row].tolist()[:2]
                for a, b in zip(got, expect_top):
                    self.assertAlmostEqual(a, b, places=4)
            if row == 1:
                got = lp.token_ids_val[1].tolist()
                self.assertAlmostEqual(got[0], ref[1], places=4)
                self.assertAlmostEqual(got[1], ref[4], places=4)
        self.assertIsNone(lp.token_ids_val[0])


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestRunnerSelectTokens(CustomTestCase):
    """_select_tokens_with_logprobs lifecycle on a bare runner (object.__new__)."""

    class _FakeCache:
        def __init__(self, offset):
            self.offset = offset

    class _FakeLayout:
        has_auxiliary_state = False
        first_attention_layer_index = 0

    def _runner(self, enable_sampling):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        runner = object.__new__(MlxModelRunner)
        runner._enable_sampling = enable_sampling
        runner._cache_layout = self._FakeLayout()
        runner._req_sampling = {}
        runner._rng_key = mx.random.key(0) if enable_sampling else None
        return runner

    def test_disabled_and_greedy_paths_consume_no_rng(self):
        """Flag-off and all-greedy batches must return exact argmax and
        leave the RNG key untouched — guards the byte-exact greedy
        contract that the e2e temp=0 test relies on."""
        logits = mx.random.normal(shape=(2, 16), key=mx.random.key(1))
        expected = mx.argmax(logits, axis=-1).tolist()
        caches = [[self._FakeCache(4)], [self._FakeCache(9)]]

        disabled = self._runner(enable_sampling=False)
        toks = disabled._select_tokens_with_logprobs(logits, ["a", "b"], caches)[0]
        self.assertEqual(toks.tolist(), expected)

        enabled = self._runner(enable_sampling=True)
        enabled._req_sampling = {"a": GREEDY_PARAMS, "b": _params(top_k=1)}
        key_before = enabled._rng_key
        toks = enabled._select_tokens_with_logprobs(logits, ["a", "b"], caches)[0]
        self.assertEqual(toks.tolist(), expected)
        self.assertIs(enabled._rng_key, key_before)

    def test_discarded_chunk_without_trunk_stays_greedy_and_consumes_no_rng(self):
        """A needs_logits=False chunk on a model without a headless trunk
        must not sample: consuming RNG for a discarded token would make
        the final output depend on prefill chunking."""

        def full_model_only(input_ids, cache=None):
            return mx.zeros((1, input_ids.shape[1], 16))

        runner = self._runner(enable_sampling=True)
        runner.model = full_model_only  # no .model attr -> no trunk
        runner._req_sampling = {"a": _params(temperature=1.0)}
        key_before = runner._rng_key
        tok, lazy_logprobs = runner._forward_lazy_token(
            mx.array([[3, 4]], dtype=mx.int32),
            [self._FakeCache(2)],
            needs_logits=False,
            req_id="a",
        )
        self.assertIsNone(lazy_logprobs)
        mx.eval(tok)
        self.assertEqual(tok.tolist(), [0])  # argmax of zeros
        self.assertIs(runner._rng_key, key_before)

    def test_logit_edits_gate_greedy_and_sampled_and_logprobs(self):
        """An additive -inf edit row must exclude a token from greedy argmax,
        from sampling, AND from the reported logprob distribution — guards
        the edits-before-selection ordering (a regression that samples raw
        logits would pass every other test on near-uniform inputs)."""
        runner = self._runner(enable_sampling=True)
        runner._req_sampling = {"g": GREEDY_PARAMS, "s": _params(temperature=1.0)}
        caches = [[self._FakeCache(4)], [self._FakeCache(4)]]
        logits = mx.zeros((2, 8))
        logits = mx.put_along_axis(
            logits,
            mx.array([[7], [7]], dtype=mx.uint32),
            mx.array([[5.0], [5.0]]),
            axis=-1,
        )  # token 7 dominates both rows
        edits = mx.zeros((2, 8))
        edits = mx.put_along_axis(
            edits,
            mx.array([[7], [7]], dtype=mx.uint32),
            mx.array([[-float("inf")], [-float("inf")]]),
            axis=-1,
        )  # ...but is masked out for both
        spec = MlxLogprobSpec(top_ks=(1, 1), token_ids=(None, None))
        for _ in range(10):
            tokens, lp = runner._select_tokens_with_logprobs(
                logits, ["g", "s"], caches, edits, spec
            )
            mx.eval(tokens, lp.chosen, lp.top_val)
            self.assertNotIn(7, tokens.tolist())
            self.assertNotIn(7, [row[0] for row in lp.top_idx.tolist()])

    def test_seed_is_gated_on_deterministic_inference(self):
        """Upstream seed contract: SamplingBatchInfo only populates
        sampling_seed under --enable-deterministic-inference, and then seeds
        every row (default 42). A per-request seed outside that flag is
        ignored by every other backend, so it is ignored here too."""
        from types import SimpleNamespace

        def make_req(sampling_seed):
            return SimpleNamespace(
                sampling_params=SimpleNamespace(
                    temperature=0.8,
                    top_k=1 << 30,
                    top_p=1.0,
                    min_p=0.0,
                    sampling_seed=sampling_seed,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    repetition_penalty=1.0,
                )
            )

        self.assertIsNone(MlxSamplingParams.from_req(make_req(7)).seed)
        self.assertIsNone(
            MlxSamplingParams.from_req(make_req(None), deterministic_seeding=False).seed
        )
        self.assertEqual(
            MlxSamplingParams.from_req(make_req(7), deterministic_seeding=True).seed, 7
        )
        self.assertEqual(
            MlxSamplingParams.from_req(make_req(None), deterministic_seeding=True).seed,
            DEFAULT_SAMPLING_SEED,
        )

    def test_chained_decode_keeps_logit_bias(self):
        """A chained decode step must keep applying the batch's static
        logit_bias rows — regression: the chained path passed edits=None,
        silently dropping the bias after the first (fresh) step."""
        runner = self._runner(enable_sampling=True)
        runner._req_sampling = {"a": GREEDY_PARAMS}
        runner._req_caches = {"a": [self._FakeCache(3)]}
        runner._req_token_ids = {"a": [1]}
        # token 2 dominates; the edit row bans it -> argmax must fall to 1
        logits = mx.array([[0.0, 3.0, 5.0, 0.0]])
        runner._decode_with_batched_attention = lambda caches, x, rids: logits
        edits = mx.array([[0.0, 0.0, -float("inf"), 0.0]])

        fresh = runner.decode_batch_start(["a"], edit_rows=edits)
        chained = runner.decode_batch_start_chained(fresh)
        mx.eval(fresh.lazy_tokens, chained.lazy_tokens)
        self.assertEqual(fresh.lazy_tokens.tolist(), [1])
        self.assertEqual(chained.lazy_tokens.tolist(), [1])

    def test_logits_hook_bridge_roundtrip(self):
        """The custom-logit-processor hook must see materialized float32
        logits and its in-place edits must re-enter the graph — guards the
        mx->numpy->mx bridge (a copy-semantics change would drop edits)."""
        runner = self._runner(enable_sampling=True)
        logits = mx.zeros((1, 8), dtype=mx.bfloat16)

        def hook(arr):
            assert arr.dtype.name == "float32"
            arr[0, 5] = 99.0
            return arr

        edited = runner._run_logits_hook(logits, hook)
        mx.eval(edited)
        self.assertEqual(int(mx.argmax(edited, axis=-1)[0].item()), 5)

    def test_sampling_path_advances_rng_key(self):
        """Consecutive sampling builds must consume distinct keys, or every
        chained decode step would draw identical noise."""
        logits = mx.zeros((1, 16))
        runner = self._runner(enable_sampling=True)
        runner._req_sampling = {"a": _params(temperature=1.0)}
        caches = [[self._FakeCache(3)]]
        toks = set()
        for _ in range(20):
            t = runner._select_tokens_with_logprobs(logits, ["a"], caches)[0]
            mx.eval(t)
            toks.add(int(t[0].item()))
        self.assertGreater(len(toks), 3, toks)


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestWorkerSamplingExtras(CustomTestCase):
    """Worker-side builders: logit-edit rows, logprob specs, output assembly."""

    VOCAB = 8

    @classmethod
    def setUpClass(cls):
        from sglang.srt.runtime_context import get_context

        # The worker reads --mlx-enable-sampling off the device config bag,
        # which fails closed before a publish.
        cls._config = get_context().override_server_args(mlx_enable_sampling=True)
        cls._config.install()
        cls.addClassCleanup(cls._config.restore)

    @staticmethod
    def _worker():
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

        return MlxTpModelWorker.__new__(MlxTpModelWorker)

    def _batch(self, sinfo, n=2, return_logprob=False, has_grammar=False):
        from types import SimpleNamespace

        return SimpleNamespace(
            reqs=[
                SimpleNamespace(
                    rid=f"r{i}", return_logprob=return_logprob, grammar=None
                )
                for i in range(n)
            ],
            sampling_info=sinfo,
            return_logprob=return_logprob,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            has_grammar=has_grammar,
        )

    def test_edit_rows_combine_grammar_mask_and_bias(self):
        """The grammar mask must be applied through the backend's own
        apply_vocab_mask on a zeros base and summed with logit_bias —
        guards the backend-agnostic zeros-trick, the combine order, and the
        ForwardBatch.init_new grammars-population mirror (regression: the
        MLX paths never build a ForwardBatch, so sinfo.grammars stayed None
        and live grammar objects produced no mask at all)."""
        from types import SimpleNamespace

        import torch

        class FakeGrammar:
            def apply_vocab_mask(self, logits, vocab_mask):
                logits[0, 3] = -float("inf")  # row 0 forbids token 3

        sinfo = SimpleNamespace(
            grammars=None,  # not yet populated, as on the real MLX path
            logit_bias=torch.zeros(2, self.VOCAB).index_put_(
                (torch.tensor([1]), torch.tensor([5])), torch.tensor([2.5])
            ),
            vocab_size=self.VOCAB,
            grammar_mask=None,
        )

        def update_mask():
            sinfo.grammar_mask = SimpleNamespace(
                grammar=FakeGrammar(), vocab_mask=torch.zeros(2, 1)
            )

        sinfo.update_regex_vocab_mask = update_mask

        batch = self._batch(sinfo, has_grammar=True)
        batch.reqs[0].grammar = object()
        rows = self._worker()._build_logit_edit_rows(batch)
        self.assertEqual(
            [g is not None for g in sinfo.grammars],
            [True, False],
            "worker must mirror ForwardBatch.init_new's grammars population",
        )
        mx.eval(rows["r0"], rows["r1"])
        self.assertEqual(rows["r0"].tolist()[3], -float("inf"))
        self.assertEqual(rows["r1"].tolist()[5], 2.5)
        self.assertIsNone(sinfo.grammar_mask, "mask must be released after use")

    def test_edit_rows_none_when_nothing_to_edit(self):
        from types import SimpleNamespace

        sinfo = SimpleNamespace(grammars=None, logit_bias=None, vocab_size=self.VOCAB)
        self.assertIsNone(self._worker()._build_logit_edit_rows(self._batch(sinfo)))

    def test_logprob_spec_subset_alignment(self):
        """Spec rows must align to the rid subset order, not batch order —
        guards mixed-batch decode sub-batches."""
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

        rows = {"a": (3, None), "c": (0, (7, 9))}
        spec = MlxTpModelWorker._logprob_spec_for(rows, ["c", "b", "a"])
        self.assertEqual(spec.top_ks, (0, 0, 3))
        self.assertEqual(spec.token_ids, ((7, 9), None, None))
        self.assertIsNone(MlxTpModelWorker._logprob_spec_for(rows, ["x"]))

    @unittest.skipUnless(
        importlib.util.find_spec("xgrammar") is not None, "requires xgrammar"
    )
    def test_xgrammar_wrapper_supports_cpu_logits(self):
        """The MLX edit-row builder feeds CPU logits to the grammar
        backend's apply_vocab_mask — regression: the xgrammar wrapper
        raised 'Unsupported device: cpu' (its dispatch stopped at
        cuda/xpu/musa/npu), so every grammar request crashed the worker."""
        import math

        import numpy as np
        import torch

        from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar

        logits = torch.zeros(1, 40)
        blocks = math.ceil(40 / 32)
        bitmask = torch.full((1, blocks), -1, dtype=torch.int32)
        bitmask[0, 0] = int(np.int32(np.uint32(0xFFFFFFFF & ~(1 << 7))))
        XGrammarGrammar.apply_vocab_mask(None, logits, bitmask)
        self.assertEqual(logits[0, 7].item(), -float("inf"))
        self.assertEqual(logits[0, 6].item(), 0.0)

    def test_assemble_logprob_output_matches_scheduler_contract(self):
        """Field shapes must survive the scheduler's move_logprobs_to_cpu
        (`.tolist()` on the batch tensor and on every per-row val/idx entry)
        and add_logprob_return_values indexing — guards the external
        LogitsProcessorOutput consumption contract, including rows without
        logprob requests getting empty-but-tolistable fills."""
        from types import SimpleNamespace

        from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

        step_rows = {"a": (-1.5, [-0.1, -0.2], [4, 2], [-3.0], [9])}
        reqs = [SimpleNamespace(rid="a"), SimpleNamespace(rid="b")]
        out = MlxTpModelWorker._assemble_logprob_output(step_rows, reqs)

        self.assertEqual(out.next_token_logprobs.tolist(), [-1.5, 0.0])
        self.assertEqual(
            [v.tolist() for v in out.next_token_top_logprobs_val],
            [[-0.10000000149011612, -0.20000000298023224], []],
        )
        self.assertEqual(
            [v.tolist() for v in out.next_token_top_logprobs_idx], [[4, 2], []]
        )
        self.assertEqual(
            [v.tolist() for v in out.next_token_token_ids_logprobs_val],
            [[-3.0], []],
        )
        self.assertEqual(out.next_token_token_ids_logprobs_idx, [[9], []])


if __name__ == "__main__":
    unittest.main()
