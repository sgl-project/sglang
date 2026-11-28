"""The HIP prefill indexer row shard runs only when the rows see enough compressed context
on average: a row at position p sees (p + 1) // ratio positions."""

import itertools
import unittest

from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
    INDEXER_ROW_SHARD_MIN_CONTEXT,
    INDEXER_ROW_SHARD_MIN_CONTEXT_BELOW_TP4,
    _compressed_positions_seen,
    indexer_row_shard_pays,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _mean_context_pays(extend_lens, seq_lens, ratio, floor):
    positions = [
        (p + 1) // ratio for n, s in zip(extend_lens, seq_lens) for p in range(s - n, s)
    ]
    return bool(positions) and sum(positions) >= floor * len(positions)


class TestIndexerRowShardGate(CustomTestCase):
    def test_closed_form_matches_per_row_count(self):
        for ratio, n in itertools.product((1, 2, 4, 128), range(300)):
            self.assertEqual(
                _compressed_positions_seen(n, ratio),
                sum(q // ratio for q in range(1, n + 1)),
                f"n={n} ratio={ratio}",
            )

    def test_gate_matches_mean_context_rule(self):
        # Ragged batches around both floors, with an empty request and short tails at long context.
        batches = [
            ([16384], [16384]),
            ([16384], [32768]),
            ([16384], [40960]),
            ([1], [2 * 12288]),
            ([1], [2 * 12288 - 2]),
            ([128] * 30, [92160] * 30),
            ([129, 0, 193, 195], [24704, 30000, 49345, 60000]),
            ([0, 0], [8192, 70000]),
        ]
        for (extend_lens, seq_lens), ratio, world in itertools.product(
            batches, (2, 4), (2, 4, 8)
        ):
            floor = (
                INDEXER_ROW_SHARD_MIN_CONTEXT
                if world >= 4
                else INDEXER_ROW_SHARD_MIN_CONTEXT_BELOW_TP4
            )
            self.assertEqual(
                indexer_row_shard_pays(
                    ratio=ratio, extend_lens=extend_lens, seq_lens=seq_lens, world=world
                ),
                _mean_context_pays(extend_lens, seq_lens, ratio, floor),
                f"extend={extend_lens} seq={seq_lens} ratio={ratio} world={world}",
            )

    def test_tp4_floor_is_inclusive(self):
        # At ratio 2 the last row of a 24,577-token sequence sees exactly 12,288 positions.
        self.assertTrue(
            indexer_row_shard_pays(ratio=2, extend_lens=[1], seq_lens=[24577], world=4)
        )
        self.assertFalse(
            indexer_row_shard_pays(ratio=2, extend_lens=[1], seq_lens=[24575], world=4)
        )
        self.assertFalse(
            indexer_row_shard_pays(ratio=2, extend_lens=[1], seq_lens=[24577], world=2)
        )


if __name__ == "__main__":
    unittest.main()
