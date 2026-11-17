"""Invariants of Muse Spark's shape-only Humming config rule and tables."""

import unittest

from sglang.srt.layers.muse_spark import muse_spark_humming_tuning as tuning
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

EP8 = dict(top_k=8, num_local_experts=32, num_global_experts=256)
UNBOUNDED = 1 << 30


def _table(*rows, use_stream_k=True):
    """Humming-style rows (lo, hi, block_m, block_n, block_k)."""
    return [
        [
            lo,
            hi,
            {
                "block_shape": [block_m, block_n, block_k],
                "warp_shape": [block_m, 32, 64],
                "use_stream_k": use_stream_k,
                "use_f16_accum": False,
                "num_stages": 4,
            },
        ]
        for lo, hi, block_m, block_n, block_k in rows
    ]


class TestEpValidShapeM(CustomTestCase):
    def test_without_ep_is_the_route_count(self):
        """Without EP the rule must equal the runner's num_tokens * top_k."""
        for experts, top_k in ((8, 2), (64, 6), (256, 8)):
            for tokens in (1, 3, 8, 100, 4096):
                self.assertEqual(
                    tuning.ep_valid_shape_m(
                        num_tokens=tokens,
                        top_k=top_k,
                        num_local_experts=experts,
                        num_global_experts=experts,
                    ),
                    tokens * top_k,
                )

    def test_ep8_caps_the_busiest_rank_until_the_expected_share_passes_it(self):
        want = {1: 8, 16: 128, 128: 1024, 256: 2048, 768: 2048, 4096: 4096}
        got = {m: tuning.ep_valid_shape_m(num_tokens=m, **EP8) for m in want}
        self.assertEqual(got, want)

    def test_monotone_between_expected_share_and_busiest_rank(self):
        """CUDA graph buckets must never pick a smaller M for a larger batch."""
        for local, total, top_k in ((32, 256, 8), (16, 128, 4), (4, 64, 8), (1, 8, 2)):
            last = 0
            for tokens in range(1, 20000, 7):
                got = tuning.ep_valid_shape_m(
                    num_tokens=tokens,
                    top_k=top_k,
                    num_local_experts=local,
                    num_global_experts=total,
                )
                self.assertLessEqual(last, got)
                self.assertLessEqual(got, max(1, tokens * min(top_k, local)))
                self.assertGreaterEqual(got, tokens * top_k * local // total)
                last = got


class TestEpTuningTables(CustomTestCase):
    def test_rows_are_open_below_and_closed_above(self):
        """Humming selects a row by min_m < valid_shape_m <= max_m; the routes
        must be aligned to that same row's block M."""
        table = _table((0, 64, 8, 256, 128), (64, UNBOUNDED, 16, 256, 128))
        self.assertEqual(tuning.block_m_for(table, 64), 8)
        self.assertEqual(tuning.block_m_for(table, 65), 16)
        with self.assertRaises(ValueError):
            tuning.config_for(table, 0)

    def test_every_row_drops_stream_k(self):
        """Stream-K sums partial K slices in BF16; Humming defaults it on."""
        table = _table((0, 64, 8, 256, 128), (64, UNBOUNDED, 64, 256, 64))
        keyless = [
            [lo, hi, {k: v for k, v in config.items() if k != "use_stream_k"}]
            for lo, hi, config in table
        ]
        for h200_class in (True, False):
            for source in (table, keyless):
                w13, w2 = tuning.ep_tuning_tables(source, source, h200_class=h200_class)
                self.assertTrue(
                    all(config["use_stream_k"] is False for _, _, config in w13 + w2)
                )
        self.assertTrue(all(config["use_stream_k"] for _, _, config in table))

    def test_h200_class_splices_the_decode_tiles(self):
        """The tiles replace Humming's rows up to their last bound, Humming's
        rows serve above it, and each GEMM keeps its own block M there."""
        w13_heuristic = _table(
            (0, 64, 8, 256, 128),
            (64, 256, 16, 128, 256),
            (256, UNBOUNDED, 64, 256, 64),
        )
        w2_heuristic = _table((0, 128, 8, 256, 128), (128, UNBOUNDED, 72, 256, 64))
        tiles = tuning._H200_DECODE_TILES
        hi = tiles[-1][1]
        w13, w2 = tuning.ep_tuning_tables(w13_heuristic, w2_heuristic, h200_class=True)
        for spliced in (w13, w2):
            bounds = [row[:2] for row in spliced]
            self.assertEqual(bounds[0][0], 0)
            self.assertEqual(bounds[-1][1], UNBOUNDED)
            for (_, prev_hi), (next_lo, _) in zip(bounds, bounds[1:]):
                self.assertEqual(prev_hi, next_lo)
        self.assertEqual([row for row in w13 if row[0] < hi], [t[:3] for t in tiles])
        self.assertEqual(
            [row for row in w2 if row[0] < hi], [t[:2] + t[3:] for t in tiles]
        )
        self.assertEqual(tuning.block_m_for(w13, hi + 1), 64)
        self.assertEqual(tuning.block_m_for(w2, hi + 1), 72)

        untouched, _ = tuning.ep_tuning_tables(
            w13_heuristic, w2_heuristic, h200_class=False
        )
        self.assertEqual(
            untouched,
            [
                [lo, hi, {**config, "use_stream_k": False}]
                for lo, hi, config in w13_heuristic
            ],
        )

    def test_decode_tiles_are_legal_for_muse_gemms(self):
        """Every tile divides Muse Spark's expert GEMMs and meets the rules
        humming-kernels 0.1.18 checks at kernel build (kernel/humming.py:
        WGMMA needs four warps along N and at least three stages)."""
        shapes = {0: (8192, 4096), 1: (4096, 4096)}  # w13, w2: (N, K)
        for row in tuning._H200_DECODE_TILES:
            for sublayer, (shape_n, shape_k) in shapes.items():
                config = row[2 + sublayer]
                block_m, block_n, block_k = config["block_shape"]
                warp_m, warp_n, warp_k = config["warp_shape"]
                with self.subTest(row=row[:2], sublayer=sublayer):
                    # Routes are aligned in blocks of block M; Humming loads
                    # them in 16-byte vectors.
                    self.assertEqual(block_m % 8, 0)
                    self.assertLessEqual(block_m, 256)
                    self.assertEqual(shape_n % block_n, 0)
                    self.assertEqual(shape_k % block_k, 0)
                    self.assertEqual(block_m % warp_m, 0)
                    self.assertEqual(block_n % warp_n, 0)
                    self.assertEqual(block_k % warp_k, 0)
                    self.assertFalse(config["use_stream_k"])
                    if config["mma_type"] == "wgmma":
                        self.assertEqual(block_n // warp_n % 4, 0)
                        self.assertGreaterEqual(config["num_stages"], 3)


if __name__ == "__main__":
    unittest.main()
