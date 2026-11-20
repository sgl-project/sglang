"""SynthID lib: hash goldens, slot state, and the watermarked sampling law."""

import functools
import unittest

import torch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.sampling.watermarking.synthid.algorithm import (
    REFERENCE_SYNTHID_KEYS,
    SynthIDConfig,
    compute_context_hash_torch,
    compute_g_values_torch,
    compute_synthid_key_iv,
    watermark_topk_scores_torch,
)
from sglang.srt.sampling.watermarking.synthid.sampler import (
    SynthIDSampler,
    top_k_top_p_min_p_filter_probs,
)
from sglang.srt.sampling.watermarking.synthid.state import SynthIDStatePool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_STATE_FIELDS = ("context", "context_history", "history_cursor", "num_calls")


def _pool(config=None, *, replay=((),)):
    config = config or SynthIDConfig(
        keys=(1, 2), ngram_len=3, top_k=3, context_history_size=4
    )
    pool = SynthIDStatePool(config, len(replay) + 1, "cpu")
    slots = torch.arange(1, len(replay) + 1)
    pool.seed(slots, replay)
    return pool, slots


def _filter_probs(top_ks, top_ps, min_ps):
    return functools.partial(
        top_k_top_p_min_p_filter_probs, top_ks=top_ks, top_ps=top_ps, min_ps=min_ps
    )


def _reference_law(logits, temperatures, context_hashes, apply_mask, keys, cap, filt):
    """The SynthID law of each watermarked row, built step by step.

    Keep the top ``cap`` logits, divide by the temperature, apply the request
    filters, run the tournament, scatter into a ``-inf`` row, and softmax. It
    reuses the tournament and the pool's hashes and masks, so their correctness
    rests on the hash and state goldens.
    """
    scores, token_ids = torch.topk(
        logits, k=min(cap, logits.shape[-1]), dim=-1, sorted=False
    )
    scores = scores / temperatures
    filtered = None
    if filt is not None:
        filtered = filt(torch.softmax(scores, dim=-1))
        scores = filtered.log()
    scores = watermark_topk_scores_torch(
        scores, token_ids, context_hashes, keys, apply_mask
    )
    if filtered is not None:
        scores = scores.masked_fill(filtered == 0, -float("inf"))
    dense = torch.full_like(logits, -float("inf")).scatter_(1, token_ids, scores)
    return torch.softmax(dense, dim=-1)


def _top_k_only(top_ks):
    return _filter_probs(top_ks, torch.ones(len(top_ks)), torch.zeros(len(top_ks)))


def _snapshot(pool):
    return {name: getattr(pool, name).clone() for name in _STATE_FIELDS}


class TestSynthIDAlgorithm(CustomTestCase):
    def test_hash_and_tournament_goldens(self):
        keys = torch.tensor([1, 2])
        iv = compute_synthid_key_iv([1, 2])
        self.assertEqual(iv, 5312871951099227430)
        hashes = compute_context_hash_torch(torch.tensor([[0, 0], [0, 2]]), iv)
        self.assertEqual(hashes.tolist(), [2393652284050965700, -3324819341964999906])
        self.assertEqual(
            compute_g_values_torch(
                hashes, torch.tensor([[0, 1, 2]] * 2), keys
            ).tolist(),
            [[[1, 0], [0, 0], [1, 0]], [[1, 0], [1, 1], [0, 0]]],
        )
        transformed = watermark_topk_scores_torch(
            torch.tensor([[1.0, 0.0, -0.5]], dtype=torch.float64),
            torch.tensor([[1, 0, 2]]),
            hashes[1:],
            keys,
            torch.tensor([True]),
        )
        torch.testing.assert_close(
            transformed,
            torch.tensor(
                [[-0.08367551842228967, -2.594303613204244, -5.189915007942694]],
                dtype=torch.float64,
            ),
            rtol=1e-12,
            atol=1e-12,
        )
        for message, bad in (
            ("distinct", {"keys": (1, 1)}),
            ("1 to 64 keys", {"keys": ()}),
            ("1 to 64 keys", {"keys": tuple(range(65))}),
            ("ngram_len must be at least 2", {"ngram_len": 1}),
            ("top_k must be from 2 to 65536", {"top_k": 1}),
            ("top_k must be from 2 to 65536", {"top_k": 65537}),
            (
                "context_history_size must be from 1 to 2048",
                {"context_history_size": 0},
            ),
            (
                "context_history_size must be from 1 to 2048",
                {"context_history_size": 2049},
            ),
        ):
            with self.subTest(**bad), self.assertRaisesRegex(ValueError, message):
                SynthIDConfig(**{"keys": (1, 2), **bad})


class TestSynthIDStatePool(CustomTestCase):
    def test_repeat_skip_and_bounded_history_projection(self):
        for skip in (False, True):
            config = SynthIDConfig(
                keys=(1, 2),
                ngram_len=3,
                top_k=3,
                context_history_size=4,
                skip_first_ngram_calls=skip,
            )
            pool, slot = _pool(config)
            for step, token in enumerate((7, 8, 7, 8)):
                _, mask = pool.current(slot)
                self.assertEqual(mask.item(), not skip or step >= 2)
                pool.commit(slot, torch.tensor([token]))
            self.assertFalse(pool.current(slot)[1].item())
        pool, slot = _pool(replay=([7],))
        hashes, mask = pool.project(slot, torch.tensor([[7, 8, 9]]))
        expected = compute_context_hash_torch(
            torch.tensor([[0, 7], [7, 8], [8, 9]]), pool.config.hash_iv
        )
        torch.testing.assert_close(hashes, expected)
        self.assertEqual(mask.tolist(), [True] * 3)
        pool.commit(slot, torch.tensor([[8, 9, 0]]), torch.tensor([2]))
        self.assertEqual(pool.context[1].tolist(), [8, 9])
        self.assertEqual(pool.num_calls[1].item(), 3)
        config = SynthIDConfig(
            keys=(1, 2), ngram_len=3, top_k=3, context_history_size=1
        )
        pool, slot = _pool(config)
        pool.context[1] = torch.tensor([1, 2])
        pool.context_history[1, 0] = compute_context_hash_torch(
            torch.tensor([[2, 3]]), config.hash_iv
        )[0]
        self.assertEqual(
            pool.project(slot, torch.tensor([[99, 3]]))[1].tolist(), [True, True]
        )

    def test_commit_and_projection_goldens(self):
        # Pins the state after a fresh and a replayed slot commit single tokens,
        # partial and zero-length chains, and a repeated context, plus the
        # hashes and masks current() and project() read from it.
        expected = {
            False: (
                [
                    [6724582080861368071, 5102677561096368021]
                    + [8327104607655880434, -7160451867295759887],
                    [-2398838170243479245, 2539401162743546112]
                    + [-7560259010295473855, 6324774937861654103],
                ],
                [1, 0],
            ),
            True: (
                [
                    [8327104607655880434, -7160451867295759887]
                    + [6724582080861368071, 0],
                    [-7560259010295473855, 6324774937861654103]
                    + [-2398838170243479245, 2539401162743546112],
                ],
                [3, 2],
            ),
        }
        current = [225878161816462216, -4601071411537562560]
        projected = [
            225878161816462216,
            -7736884646230218570,
            -6760644724296045919,
            -4601071411537562560,
            1117400214478403046,
            -7560259010295473855,
        ]
        for skip, (history, cursor) in expected.items():
            with self.subTest(skip=skip):
                config = SynthIDConfig(
                    keys=REFERENCE_SYNTHID_KEYS[:4],
                    ngram_len=3,
                    top_k=8,
                    context_history_size=4,
                    skip_first_ngram_calls=skip,
                )
                pool = SynthIDStatePool(config, 4, "cpu")
                slots = torch.tensor([1, 3])
                pool.seed(slots, [[], [5, 6, 7, 5, 6]])
                pool.commit(slots, torch.tensor([11, 12]))
                pool.commit(
                    slots,
                    torch.tensor([[13, 14, 15], [16, 17, 0]]),
                    torch.tensor([3, 2]),
                )
                pool.commit(
                    slots, torch.tensor([[11, 12, 13], [5, 6, 7]]), torch.tensor([0, 3])
                )
                pool.commit(slots, torch.tensor([7, 7]))
                self.assertEqual(pool.context[slots].tolist(), [[15, 7], [7, 7]])
                self.assertEqual(pool.context_history[slots].tolist(), history)
                self.assertEqual(pool.history_cursor[slots].tolist(), cursor)
                self.assertEqual(pool.num_calls[slots].tolist(), [5, 12])
                hashes, mask = pool.current(slots)
                self.assertEqual(hashes.tolist(), current)
                self.assertEqual(mask.tolist(), [True, True])
                hashes, mask = pool.project(
                    slots, torch.tensor([[7, 21, 22], [7, 5, 6]])
                )
                self.assertEqual(hashes.tolist(), projected)
                self.assertEqual(mask.tolist(), [True] * 5 + [False])

    def test_seed_resets_reused_slot_and_replays_resumed_request(self):
        tokens = [4, 9, 4, 9, 4, 2, 6]
        for skip in (False, True):
            config = SynthIDConfig(
                keys=(1, 2),
                ngram_len=3,
                top_k=3,
                context_history_size=4,
                skip_first_ngram_calls=skip,
            )
            pool = SynthIDStatePool(config, 3, "cpu")
            pool.seed(torch.tensor([1, 2]), [[], []])
            for token in tokens:
                pool.commit(torch.tensor([1]), torch.tensor([token]))
            pool.seed(torch.tensor([2]), [tokens])
            for name in _STATE_FIELDS:
                state = getattr(pool, name)
                self.assertEqual(state[2].tolist(), state[1].tolist(), name)
            pool.seed(torch.tensor([1]), [[]])
            self.assertEqual(pool.context[1].tolist(), [0, 0])
            self.assertEqual(pool.context_history[1].tolist(), [0] * 4)
            self.assertEqual(pool.history_cursor[1].item(), 0)
            self.assertEqual(pool.num_calls[1].item(), 0)


class TestSynthIDSampler(CustomTestCase):
    def _sampler(self, cap=6, vocab=16, replay=((), (3, 4, 3, 4), (5,), (1, 2, 1))):
        config = SynthIDConfig(
            keys=REFERENCE_SYNTHID_KEYS[:4],
            ngram_len=3,
            top_k=cap,
            context_history_size=4,
        )
        sampler = SynthIDSampler(config, len(replay) + 1, "cpu")
        slots = torch.arange(1, len(replay) + 1)
        sampler.state.seed(slots, replay)
        torch.manual_seed(0)
        logits = torch.randn(len(replay), vocab) * 2
        return sampler, slots, logits

    def _filters(self, batch):
        top_ks = torch.tensor([3, TOP_K_ALL, 4, TOP_K_ALL])[:batch]
        top_ps = torch.tensor([1.0, 0.7, 0.9, 1.0])[:batch]
        min_ps = torch.tensor([0.0, 0.0, 0.2, 0.3])[:batch]
        return {
            "none": None,
            "top_k": (top_ks, torch.ones(batch), torch.zeros(batch)),
            "top_p": (torch.full((batch,), TOP_K_ALL), top_ps, torch.zeros(batch)),
            "min_p": (torch.full((batch,), TOP_K_ALL), torch.ones(batch), min_ps),
            "joint": (top_ks, top_ps, min_ps),
        }

    def test_transform_matches_reference_law_under_a_stock_sampler(self):
        temperatures = torch.tensor([[0.7], [1.0], [1.3], [0.5]])
        for mode, rows in (("all_on", None), ("mixed", torch.tensor([0, 2, 3]))):
            for name, params in self._filters(4).items():
                with self.subTest(mode=mode, filters=name):
                    sampler, slots, logits = self._sampler()
                    on = torch.arange(4) if rows is None else rows
                    _, apply_mask = sampler.state.current(slots[on])
                    if mode == "all_on":
                        # Row 1 replays a repeated context: base law in the cap.
                        self.assertEqual(apply_mask.tolist(), [True, False, True, True])
                    filt = None
                    if params is not None:
                        filt = _filter_probs(*(p[on] for p in params))
                    expected = _reference_law(
                        logits[on],
                        temperatures[on],
                        *sampler.state.current(slots[on]),
                        sampler.keys,
                        sampler.config.top_k,
                        filt,
                    )

                    watermarked = logits.clone()
                    state = _snapshot(sampler.state)
                    sampler.apply(
                        watermarked,
                        temperatures[on],
                        slots[on],
                        rows=rows,
                        filter_probs=filt,
                    )
                    self.assert_state_unchanged(sampler.state, state)
                    # The stock sampler: divide by T, softmax, keep the request
                    # top-k (top_p=1 and min_p=0 on watermarked rows).
                    probs = torch.softmax(watermarked / temperatures, dim=-1)[on]
                    if params is not None:
                        probs = _top_k_only(params[0][on])(probs)
                    torch.testing.assert_close(probs, expected, rtol=0, atol=1e-6)
                    off = [row for row in range(4) if row not in on.tolist()]
                    self.assertTrue(torch.equal(watermarked[off], logits[off]))

    def test_verify_transform_matches_reference_law_per_position(self):
        width = 3
        temperatures = torch.tensor([[0.7], [1.0], [1.3], [0.5]])
        verify_tokens = torch.tensor([[0, 3, 4], [4, 3, 4], [5, 6, 7], [1, 1, 2]])
        for mode, requests in (("all_on", None), ("mixed", torch.tensor([1, 3]))):
            for name, params in self._filters(4).items():
                with self.subTest(mode=mode, filters=name):
                    sampler, slots, _ = self._sampler()
                    logits = torch.randn(4 * width, 16) * 2
                    on = torch.arange(4) if requests is None else requests
                    rows = (on[:, None] * width + torch.arange(width)).view(-1)
                    filt = None
                    if params is not None:
                        filt = _filter_probs(
                            *(p[on].repeat_interleave(width) for p in params)
                        )
                    row_temperatures = temperatures[on].repeat_interleave(width, 0)
                    expected = _reference_law(
                        logits[rows],
                        row_temperatures,
                        *sampler.state.project(slots[on], verify_tokens[on]),
                        sampler.keys,
                        sampler.config.top_k,
                        filt,
                    )

                    watermarked = logits.clone()
                    state = _snapshot(sampler.state)
                    sampler.apply_verify(
                        watermarked,
                        temperatures[on],
                        slots[on],
                        verify_tokens[on],
                        requests=requests,
                        filter_probs=filt,
                    )
                    self.assert_state_unchanged(sampler.state, state)
                    probs = torch.softmax(
                        watermarked / temperatures.repeat_interleave(width, 0), dim=-1
                    )[rows]
                    if params is not None:
                        probs = _top_k_only(params[0][on].repeat_interleave(width))(
                            probs
                        )
                    torch.testing.assert_close(probs, expected, rtol=0, atol=1e-6)
                    off = [r for r in range(4 * width) if r not in rows.tolist()]
                    self.assertTrue(torch.equal(watermarked[off], logits[off]))

    def test_cap_and_filters_bound_the_support(self):
        raw = torch.tensor([[0.4, 0.3, 0.2, 0.1]]).log()
        for cap, requested, support in (
            (3, 2, 2),
            (3, TOP_K_ALL, 3),
            (16, TOP_K_ALL, 4),
        ):
            with self.subTest(cap=cap, requested=requested):
                config = SynthIDConfig(
                    keys=(1, 2), top_k=cap, skip_first_ngram_calls=True
                )
                sampler = SynthIDSampler(config, 2, "cpu")
                slot = torch.tensor([1])
                sampler.state.seed(slot, [[]])
                logits = raw.clone()
                sampler.apply(
                    logits,
                    torch.ones(1, 1),
                    slot,
                    filter_probs=_filter_probs(
                        torch.tensor([requested]), torch.ones(1), torch.zeros(1)
                    ),
                )
                expected = raw.exp()
                expected[:, support:] = 0
                expected /= expected.sum(-1, keepdim=True)
                torch.testing.assert_close(logits.softmax(-1), expected)
        config = SynthIDConfig(keys=(1, 2), ngram_len=3, top_k=3)
        sampler = SynthIDSampler(config, 2, "cpu")
        slot = torch.tensor([1])
        sampler.state.seed(slot, [[]])
        logits = torch.tensor([[0.5, 0.3, 0.2]]).log()
        temperature = torch.tensor([[0.5]])
        sampler.apply(
            logits,
            temperature,
            slot,
            filter_probs=_filter_probs(
                torch.tensor([TOP_K_ALL]), torch.tensor([0.6]), torch.zeros(1)
            ),
        )
        torch.testing.assert_close(
            (logits / temperature).softmax(-1), torch.tensor([[1.0, 0.0, 0.0]])
        )

    def test_request_filter_keeps_the_pytorch_sampler_support(self):
        generator = torch.Generator().manual_seed(0)
        num_rows, vocab = 8, 32
        positions = torch.zeros(num_rows, dtype=torch.int64)
        for _ in range(20):
            # Four probability levels on a 1/64 grid, top-p on that grid, and
            # min-p in quarters: every sum and threshold is exact, so ties and
            # values exactly at the top-k, top-p, and min-p cuts occur.
            probs = torch.randint(1, 5, (num_rows, vocab), generator=generator) / 64
            top_ks = torch.randint(1, vocab + 1, (num_rows,), generator=generator)
            top_ks[0] = TOP_K_ALL
            top_ps = torch.randint(1, 65, (num_rows,), generator=generator) / 64
            min_ps = torch.randint(0, 4, (num_rows,), generator=generator) / 4
            for need_min_p in (False, True):
                filtered = top_k_top_p_min_p_filter_probs(
                    probs.clone(),
                    top_ks=top_ks,
                    top_ps=top_ps,
                    min_ps=min_ps if need_min_p else None,
                )
                _, stock_sort, stock_idx, _ = (
                    top_k_top_p_min_p_sampling_from_probs_torch(
                        probs.clone(),
                        top_ks.to(torch.int32),
                        top_ps,
                        min_ps,
                        need_min_p,
                        None,
                        positions,
                        return_filtered_probs=True,
                    )
                )
                stock = torch.zeros_like(probs).scatter_(
                    1, stock_idx.to(torch.int64), stock_sort
                )
                self.assertTrue(torch.equal(filtered > 0, stock > 0))

    def assert_state_unchanged(self, pool, snapshot):
        for name, before in snapshot.items():
            self.assertTrue(torch.equal(getattr(pool, name), before), name)


if __name__ == "__main__":
    unittest.main()
