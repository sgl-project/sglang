import unittest

from sglang.srt.layers.attention.dsv4.mega_attention import use_mega_paged_prefill
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestMegaPrefillPolicy(CustomTestCase):
    def test_observed_agentx_shapes_and_unmeasured_paths(self):
        # Regressions would restore the full C1 gather on short cached extensions
        # or send large/fresh prefills through the slower paged specialization.
        cases = (
            (558, [134702], 1, 64, True),
            (558, [134702], 2, 64, False),
            (1532, [101884], 1, 64, False),
            (3823, [156571, 115284], 1, 64, False),
            (3047, [3047], 1, 64, False),
            (128, [32768], 1, 64, True),
            (128, [32768], 2, 64, False),
            (512, [49152, 16384], 1, 64, True),
            (512, [49152, 16384], 2, 64, False),
            (1024, [524288], 2, 64, True),
            (1025, [524288], 2, 64, False),
            (1, [32768], 1, 64, True),
            (1, [4096], 1, 64, False),
            (0, [131072], 1, 64, False),
            (558, [], 1, 64, False),
            (558, [134702], 0, 64, False),
            (558, [134702], 4, 64, False),
            (558, [134702], 1, 128, False),
        )
        for queries, lengths, ratio, heads, expected in cases:
            with self.subTest(
                queries=queries, lengths=lengths, ratio=ratio, heads=heads
            ):
                self.assertEqual(
                    use_mega_paged_prefill(queries, lengths, ratio, heads), expected
                )


if __name__ == "__main__":
    unittest.main()
