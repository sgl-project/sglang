import unittest

import torch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.sampling.watermarking.synthid.algorithm import (
    SynthIDConfig,
    compute_g_values_torch,
    compute_hash_iv,
    hash_contexts,
    synthid_tournament_torch,
)
from sglang.srt.sampling.watermarking.synthid.state import SynthIDStatePool
from sglang.srt.sampling.watermarking.synthid.watermark import SynthIDWatermark
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.synthid_utils import (
    SYNTHID_TEST_KEYS,
    assert_synthid_state_equal,
    synthid_state,
)

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _pool(config=None, *, replay=((),)):
    config = config or SynthIDConfig(
        keys=(1, 2), ngram_len=3, num_candidates=3, context_history_size=4
    )
    pool = SynthIDStatePool(config, len(replay) + 1, "cpu")
    slots = torch.arange(1, len(replay) + 1)
    pool.seed(slots, replay)
    return pool, slots


def _drop_below_half_of_max(probs):
    kept = torch.where(probs >= probs.max(dim=-1, keepdim=True).values / 2, probs, 0)
    return kept / kept.sum(dim=-1, keepdim=True)


def _reference_law(logits, temperatures, context_hashes, apply_mask, keys, cap, filt):
    """The SynthID law of each watermarked row; it guards the ``score * T``
    round trip through the stock sampler."""
    scores, token_ids = torch.topk(
        logits, k=min(cap, logits.shape[-1]), dim=-1, sorted=False
    )
    scores = scores / temperatures
    filtered = None
    if filt is not None:
        filtered = filt(torch.softmax(scores, dim=-1))
        scores = filtered.log()
    scores = synthid_tournament_torch(
        scores, token_ids, context_hashes, keys, apply_mask
    )
    if filtered is not None:
        scores = scores.masked_fill(filtered == 0, -float("inf"))
    dense = torch.full_like(logits, -float("inf")).scatter_(1, token_ids, scores)
    return torch.softmax(dense, dim=-1)


class TestSynthIDAlgorithm(CustomTestCase):
    def test_hash_and_tournament_goldens(self):
        # Goldens computed with google-deepmind/synthid-text@addb4a15.
        keys = torch.tensor([1, 2])
        hashes = hash_contexts(torch.tensor([[0, 0], [0, 2]]), compute_hash_iv([1, 2]))
        self.assertEqual(
            compute_g_values_torch(
                hashes, torch.tensor([[0, 1, 2]] * 2), keys
            ).tolist(),
            [[[1, 0], [0, 0], [1, 0]], [[1, 0], [1, 1], [0, 0]]],
        )
        transformed = synthid_tournament_torch(
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


class TestSynthIDStatePool(CustomTestCase):
    def test_repeat_skip_and_bounded_history_projection(self):
        for skip in (False, True):
            config = SynthIDConfig(
                keys=(1, 2),
                ngram_len=3,
                num_candidates=3,
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
        expected = hash_contexts(
            torch.tensor([[0, 7], [7, 8], [8, 9]]), pool.config.hash_iv
        )
        torch.testing.assert_close(hashes, expected)
        self.assertEqual(mask.tolist(), [True] * 3)
        pool.commit(slot, torch.tensor([[8, 9, 0]]), torch.tensor([2]))
        self.assertEqual(pool.context[1].tolist(), [8, 9])
        self.assertEqual(pool.num_calls[1].item(), 3)
        config = SynthIDConfig(
            keys=(1, 2), ngram_len=3, num_candidates=3, context_history_size=1
        )
        pool, slot = _pool(config)
        pool.context[1] = torch.tensor([1, 2])
        pool.context_history[1, 0] = hash_contexts(
            torch.tensor([[2, 3]]), config.hash_iv
        )[0]
        self.assertEqual(
            pool.project(slot, torch.tensor([[99, 3]]))[1].tolist(), [True, True]
        )

    def test_seed_resets_reused_slot_and_replays_resumed_request(self):
        tokens = [4, 9, 4, 9, 4, 2, 6]
        for skip in (False, True):
            config = SynthIDConfig(
                keys=(1, 2),
                ngram_len=3,
                num_candidates=3,
                context_history_size=4,
                skip_first_ngram_calls=skip,
            )
            pool = SynthIDStatePool(config, 3, "cpu")
            pool.seed(torch.tensor([1, 2]), [[], []])
            for token in tokens:
                pool.commit(torch.tensor([1]), torch.tensor([token]))
            pool.seed(torch.tensor([2]), [tokens])
            for name, state in synthid_state(pool).items():
                self.assertEqual(state[2].tolist(), state[1].tolist(), name)
            pool.seed(torch.tensor([1]), [[]])
            for name, state in synthid_state(pool).items():
                self.assertFalse(state[1].any(), name)


class TestSynthIDWatermark(CustomTestCase):
    _REPLAY = ((), (3, 4, 3, 4), (5,), (1, 2, 1))
    _TEMPERATURES = torch.tensor([[0.7], [1.0], [1.3], [0.5]])

    def _watermark(self):
        config = SynthIDConfig(
            keys=SYNTHID_TEST_KEYS[:4],
            ngram_len=3,
            num_candidates=6,
            context_history_size=4,
        )
        slots = torch.arange(1, len(self._REPLAY) + 1)
        watermark = SynthIDWatermark(config, len(self._REPLAY) + 1, "cpu")
        watermark.state.seed(slots, self._REPLAY)
        reference = SynthIDStatePool(config, len(self._REPLAY) + 1, "cpu")
        reference.seed(slots, self._REPLAY)
        return watermark, reference, slots

    def test_transform_matches_reference_law_under_a_stock_sampler(self):
        for mode, rows, filt in (
            ("all_on", None, None),
            ("mixed", torch.tensor([0, 2, 3]), _drop_below_half_of_max),
        ):
            with self.subTest(mode=mode):
                watermark, reference, slots = self._watermark()
                torch.manual_seed(0)
                logits = torch.randn(4, 16) * 2
                on = torch.arange(4) if rows is None else rows
                context_hashes, apply_mask = reference.current(slots[on])
                if mode == "all_on":
                    # Row 1 replays a repeated context: base law in the cap.
                    self.assertEqual(apply_mask.tolist(), [True, False, True, True])
                expected = _reference_law(
                    logits[on],
                    self._TEMPERATURES[on],
                    context_hashes,
                    apply_mask,
                    watermark.keys,
                    watermark.config.num_candidates,
                    filt,
                )

                watermarked = logits.clone()
                state = synthid_state(watermark.state)
                watermark.apply(
                    watermarked,
                    self._TEMPERATURES[on],
                    slots[on],
                    rows=rows,
                    filter_probs=filt,
                )
                assert_synthid_state_equal(synthid_state(watermark.state), state)
                probs = torch.softmax(watermarked / self._TEMPERATURES, dim=-1)
                torch.testing.assert_close(probs[on], expected, rtol=0, atol=1e-6)
                off = [row for row in range(4) if row not in on.tolist()]
                self.assertTrue(torch.equal(watermarked[off], logits[off]))

    def test_verify_transform_matches_reference_law_per_position(self):
        width = 3
        verify_tokens = torch.tensor([[0, 3, 4], [4, 3, 4], [5, 6, 7], [1, 1, 2]])
        for mode, request_rows, filt in (
            ("all_on", None, None),
            ("mixed", torch.tensor([1, 3]), _drop_below_half_of_max),
        ):
            with self.subTest(mode=mode):
                watermark, reference, slots = self._watermark()
                torch.manual_seed(0)
                logits = torch.randn(4 * width, 16) * 2
                on = torch.arange(4) if request_rows is None else request_rows
                rows = (on[:, None] * width + torch.arange(width)).view(-1)
                row_temperatures = self._TEMPERATURES.repeat_interleave(width, 0)
                expected = _reference_law(
                    logits[rows],
                    row_temperatures[rows],
                    *reference.project(slots[on], verify_tokens[on]),
                    watermark.keys,
                    watermark.config.num_candidates,
                    filt,
                )

                watermarked = logits.clone()
                state = synthid_state(watermark.state)
                watermark.apply_verify(
                    watermarked,
                    self._TEMPERATURES[on],
                    slots[on],
                    verify_tokens[on],
                    request_rows=request_rows,
                    filter_probs=filt,
                )
                assert_synthid_state_equal(synthid_state(watermark.state), state)
                probs = torch.softmax(watermarked / row_temperatures, dim=-1)
                torch.testing.assert_close(probs[rows], expected, rtol=0, atol=1e-6)
                off = [r for r in range(4 * width) if r not in rows.tolist()]
                self.assertTrue(torch.equal(watermarked[off], logits[off]))


if __name__ == "__main__":
    unittest.main()
