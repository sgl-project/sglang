"""Check that QSA's dense prefill guard matches what the indexer actually selects."""

import unittest

import torch

from sglang.srt.layers.attention.qsa.config import QSAProfile
from sglang.srt.layers.attention.qsa.dense_prefill import (
    assert_prefill_selection_exhaustive,
    dense_causal_prefill_reference,
    exhaustive_prefill_budget,
    gather_packed_sequence_kv,
    is_prefill_selection_exhaustive,
    total_visible_tokens,
    visible_prefix_counts,
)
from sglang.srt.layers.attention.qsa.kernel import (
    qsa_fast_topk,
    qsa_sparse_attention_reference,
    torch_expand_qsa_block_indices,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

# The production profile is budget 2048 over ratio 4; these shapes keep the
# reference attention cheap while exercising the same block arithmetic.
_RATIO = 4
_BLOCK_TOPK = 8
_BUDGET = _BLOCK_TOPK * _RATIO
_QUERY_HEADS = 4
_KV_HEADS = 1
_HEAD_DIM = 16


def _select_prefill_tokens(*, sequence_len: int, rows: int, seed: int):
    """Run the indexer's prefill selection for one request on CPU.

    Mirrors ``QSAIndexer.select_prefill_tokens``: score each row's complete
    blocks, keep the top ``block_topk``, then expand to logical token indices.
    ``rows`` trailing query positions are selected for, so a chunk that starts
    mid-sequence is covered too.
    """

    generator = torch.Generator().manual_seed(seed)
    positions = torch.arange(sequence_len - rows, sequence_len, dtype=torch.int64)
    candidates = (positions + 1) // _RATIO
    logits = torch.randn(
        rows, max(int(candidates.max()), 1), generator=generator, dtype=torch.float32
    )
    block_indices = qsa_fast_topk(
        logits,
        torch.zeros(rows, dtype=torch.int32),
        candidates.to(torch.int32),
        topk=_BLOCK_TOPK,
    )
    return torch_expand_qsa_block_indices(
        block_indices,
        positions,
        torch.full((rows,), sequence_len, dtype=torch.int64),
        _RATIO,
        _BUDGET,
    )


def _selected_sets(token_indices: torch.Tensor):
    return [
        set(int(value) for value in row[row >= 0].tolist()) for row in token_indices
    ]


class TestQSADensePrefillGuard(CustomTestCase):
    def test_budget_comes_from_the_profile(self):
        profile = QSAProfile(
            n_heads=4, kv_heads=1, head_dim=128, budget=2048, compress_ratio=4
        )
        self.assertEqual(profile.block_topk, 512)
        self.assertEqual(exhaustive_prefill_budget(profile), 2048)
        self.assertEqual(exhaustive_prefill_budget(None), 0)

    def test_guard_accepts_exactly_the_provable_cases(self):
        cases = {
            # (sequence_lens, extend_lens, query_rows): eligible
            ((_BUDGET,), (_BUDGET,), _BUDGET): True,
            ((_BUDGET + 1,), (_BUDGET + 1,), _BUDGET + 1): False,
            ((_BUDGET, 4), (_BUDGET, 4), _BUDGET + 4): True,
            ((_BUDGET, _BUDGET + 1), (_BUDGET, _BUDGET + 1), 2 * _BUDGET + 1): False,
            # Continuation chunk: only part of the sequence is being extended.
            ((_BUDGET,), (4,), 4): True,
            # DP padding or a speculative layout: rows do not match extend lens.
            ((_BUDGET,), (_BUDGET,), _BUDGET + 8): False,
            # Decode-shaped and empty batches stay on the sparse path.
            ((_BUDGET,), (0,), 0): False,
            ((), (), 0): False,
            # Malformed metadata is rejected rather than guessed at.
            ((_BUDGET, _BUDGET), (_BUDGET,), _BUDGET): False,
            ((2,), (4,), 4): False,
        }
        for (sequence_lens, extend_lens, query_rows), eligible in cases.items():
            with self.subTest(sequence_lens=sequence_lens, extend_lens=extend_lens):
                self.assertEqual(
                    is_prefill_selection_exhaustive(
                        sequence_lens=list(sequence_lens),
                        extend_lens=list(extend_lens),
                        query_rows=query_rows,
                        budget=_BUDGET,
                    ),
                    eligible,
                )

    def test_disabled_budget_never_takes_the_dense_path(self):
        self.assertFalse(
            is_prefill_selection_exhaustive(
                sequence_lens=[4], extend_lens=[4], query_rows=4, budget=0
            )
        )

    def test_selection_covers_the_causal_prefix_within_budget(self):
        for sequence_len in (1, 2, 3, 4, 5, 7, 8, _BUDGET - 1, _BUDGET):
            with self.subTest(sequence_len=sequence_len):
                indices = _select_prefill_tokens(
                    sequence_len=sequence_len, rows=sequence_len, seed=sequence_len
                )
                expected = [set(range(row + 1)) for row in range(sequence_len)]
                self.assertEqual(_selected_sets(indices), expected)

    def test_selection_is_incomplete_past_the_budget(self):
        sequence_len = _BUDGET + _RATIO
        indices = _select_prefill_tokens(
            sequence_len=sequence_len, rows=sequence_len, seed=7
        )
        selected = _selected_sets(indices)
        self.assertEqual(selected[0], {0})
        self.assertNotEqual(selected[-1], set(range(sequence_len)))

    def test_dense_matches_sparse_for_a_full_chunk(self):
        self._assert_dense_matches_sparse(sequence_len=_BUDGET, extend_len=_BUDGET)

    def test_dense_matches_sparse_for_a_continuation_chunk(self):
        # The guard's one untested corner in the original NPU measurements:
        # queries start mid-sequence, so causality is bottom-right aligned.
        for extend_len in (1, 3, _RATIO, _BUDGET // 2):
            with self.subTest(extend_len=extend_len):
                self._assert_dense_matches_sparse(
                    sequence_len=_BUDGET, extend_len=extend_len
                )

    def _assert_dense_matches_sparse(self, *, sequence_len: int, extend_len: int):
        generator = torch.Generator().manual_seed(sequence_len * 1000 + extend_len)
        q = torch.randn(
            extend_len,
            _QUERY_HEADS,
            _HEAD_DIM,
            generator=generator,
            dtype=torch.float32,
        )
        k = torch.randn(
            sequence_len, _KV_HEADS, _HEAD_DIM, generator=generator, dtype=torch.float32
        )
        v = torch.randn(
            sequence_len, _KV_HEADS, _HEAD_DIM, generator=generator, dtype=torch.float32
        )
        scale = _HEAD_DIM**-0.5
        indices = _select_prefill_tokens(
            sequence_len=sequence_len, rows=extend_len, seed=extend_len
        )
        sparse = qsa_sparse_attention_reference(q, k, v, indices, scale)
        dense = dense_causal_prefill_reference(
            q=q,
            k=k,
            v=v,
            extend_lens=[extend_len],
            sequence_lens=[sequence_len],
            scale=scale,
        )
        self.assertEqual(dense.shape, sparse.shape)
        torch.testing.assert_close(dense, sparse, rtol=1e-5, atol=1e-5)

    def test_dense_handles_a_multi_request_batch(self):
        extend_lens = [4, 1, _BUDGET]
        sequence_lens = [_BUDGET, _BUDGET, _BUDGET]
        generator = torch.Generator().manual_seed(11)
        q = torch.randn(
            sum(extend_lens),
            _QUERY_HEADS,
            _HEAD_DIM,
            generator=generator,
            dtype=torch.float32,
        )
        k = torch.randn(
            sum(sequence_lens),
            _KV_HEADS,
            _HEAD_DIM,
            generator=generator,
            dtype=torch.float32,
        )
        v = torch.randn_like(k)
        dense = dense_causal_prefill_reference(
            q=q,
            k=k,
            v=v,
            extend_lens=extend_lens,
            sequence_lens=sequence_lens,
            scale=_HEAD_DIM**-0.5,
        )
        self.assertEqual(dense.shape, q.shape)
        query_start = 0
        key_start = 0
        for extend, total in zip(extend_lens, sequence_lens):
            indices = _select_prefill_tokens(
                sequence_len=total, rows=extend, seed=extend
            )
            sparse = qsa_sparse_attention_reference(
                q[query_start : query_start + extend],
                k[key_start : key_start + total],
                v[key_start : key_start + total],
                indices,
                _HEAD_DIM**-0.5,
            )
            torch.testing.assert_close(
                dense[query_start : query_start + extend],
                sparse,
                rtol=1e-5,
                atol=1e-5,
            )
            query_start += extend
            key_start += total

    def test_total_visible_tokens_bounds_the_selection(self):
        # The closed form must equal the row-by-row sum, and the selection must
        # reach it exactly when exhaustive and fall short otherwise. This is the
        # invariant the default-on device assertion relies on.
        for sequence_lens, extend_lens in (
            ([8], [8]),
            ([8], [3]),
            ([_BUDGET], [1]),
            ([_BUDGET, 8, 5], [_BUDGET, 3, 5]),
            ([4], [0]),
            ([], []),
        ):
            with self.subTest(sequence_lens=sequence_lens, extend_lens=extend_lens):
                counts = visible_prefix_counts(
                    sequence_lens=sequence_lens,
                    extend_lens=extend_lens,
                    device=torch.device("cpu"),
                )
                self.assertEqual(
                    total_visible_tokens(
                        sequence_lens=sequence_lens, extend_lens=extend_lens
                    ),
                    int(counts.sum()),
                )
        for sequence_len, extend_len in ((8, 8), (_BUDGET, 4), (_BUDGET, _BUDGET)):
            with self.subTest(sequence_len=sequence_len, extend_len=extend_len):
                indices = _select_prefill_tokens(
                    sequence_len=sequence_len, rows=extend_len, seed=extend_len
                )
                expected = total_visible_tokens(
                    sequence_lens=[sequence_len], extend_lens=[extend_len]
                )
                self.assertEqual(int((indices >= 0).sum()), expected)
                damaged = indices.clone()
                damaged[-1, 0] = -1
                self.assertLess(int((damaged >= 0).sum()), expected)

    def test_assertion_rejects_an_under_selected_row(self):
        sequence_lens = [8]
        extend_lens = [8]
        counts = visible_prefix_counts(
            sequence_lens=sequence_lens,
            extend_lens=extend_lens,
            device=torch.device("cpu"),
        )
        self.assertEqual(counts.tolist(), [1, 2, 3, 4, 5, 6, 7, 8])
        indices = _select_prefill_tokens(sequence_len=8, rows=8, seed=3)
        assert_prefill_selection_exhaustive(
            token_indices=indices, visible_counts=counts
        )
        damaged = indices.clone()
        damaged[5, 0] = -1
        with self.assertRaisesRegex(RuntimeError, "not exhaustive: row 5"):
            assert_prefill_selection_exhaustive(
                token_indices=damaged, visible_counts=counts
            )

    def test_packed_gather_matches_a_slot_by_slot_gather(self):
        sequence_lens = [3, 5]
        key_cache = torch.arange(64, dtype=torch.float32).reshape(32, _KV_HEADS, 2)
        value_cache = key_cache + 100.0
        slot_table = torch.tensor(
            [[7, 5, 3, 0, 0, 0, 0, 0], [1, 2, 4, 6, 8, 0, 0, 0]], dtype=torch.int32
        )
        keys, values = gather_packed_sequence_kv(
            key_cache=key_cache,
            value_cache=value_cache,
            token_slot_table=slot_table,
            sequence_lens=sequence_lens,
        )
        expected_slots = [7, 5, 3, 1, 2, 4, 6, 8]
        torch.testing.assert_close(keys, key_cache[expected_slots])
        torch.testing.assert_close(values, value_cache[expected_slots])


if __name__ == "__main__":
    unittest.main()
